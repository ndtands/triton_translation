import json
import triton_python_backend_utils as pb_utils
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np

MODEL_PATH = "/workspace/model_repository/pretrain/en2vi"
NUM_BEAMS = 5
EARLY_STOPPING = True

class TritonPythonModel:
    def initialize(self, args):
        """Khởi tạo mô hình khi được tải lên.

        Parameters
        ----------
        args : dict
          Cấu hình mô hình và thông tin instance.
        """
        # Lấy cấu hình mô hình từ args
        self.model_config = json.loads(args["model_config"])

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            src_lang="en_XX"
        )

        # Load mô hình
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.model.half()

        # Lấy tham số từ config
        self.num_beams = NUM_BEAMS
        self.early_stopping = EARLY_STOPPING

        # Lấy cấu hình đầu ra từ model_config
        output_config = pb_utils.get_output_config_by_name(self.model_config, "output_ids")

        # Chuyển đổi kiểu dữ liệu từ Triton sang numpy
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    def execute(self, requests):
        """Xử lý các request đầu vào và tạo bản dịch.

        Parameters
        ----------
        requests : list
          Danh sách các request từ client.

        Returns
        -------
        list
          Danh sách các InferenceResponse chứa tensor đầu ra.
        """
        logger = pb_utils.Logger
        responses = []
        for request in requests:
            # Lấy dữ liệu đầu vào từ tensor "input_ids"
            input_ids = pb_utils.get_input_tensor_by_name(request, "input_ids").as_numpy()
            logger.log_info(f"[EN2VI_MODEL] [Input 1]: {input_ids.shape}")
            input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)

            # Generate bản dịch
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    decoder_start_token_id=self.tokenizer.lang_code_to_id["en_XX"],
                    num_beams=self.num_beams,
                    early_stopping=self.early_stopping
                )

            # Chuyển đổi sang numpy và định dạng kiểu dữ liệu
            output_ids = output_ids.cpu().numpy().astype(self.output_dtype)

            # Tạo tensor đầu ra
            output_tensor = pb_utils.Tensor("output_ids", output_ids)

            # Tạo response và thêm vào danh sách
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses

    def finalize(self):
        """Dọn dẹp khi mô hình bị unload."""
        print("Cleaning up...")