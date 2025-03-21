import json
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer
import numpy as np


MODEL_PATH = "/workspace/model_repository/pretrain/en2vi"

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

        # Khởi tạo tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            src_lang="en_XX"
        )

        # Lấy cấu hình đầu ra từ model_config
        output_config = pb_utils.get_output_config_by_name(self.model_config, "input_ids")

        # Chuyển đổi kiểu dữ liệu từ Triton sang numpy
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    def execute(self, requests):
        """Xử lý các request đầu vào và trả về tensor đã được tiền xử lý.

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
            texts = pb_utils.get_input_tensor_by_name(request, "texts").as_numpy()
            logger.log_info(f"[EN2VI_PREPROCESS] [Input 1]: {texts.shape}")
            texts = [text[0].decode("utf-8") for text in texts]

            # Tokenize văn bản
            inputs = self.tokenizer(
                texts,
                return_tensors="np",
                padding=True,
                truncation=True
            )
            input_ids = inputs["input_ids"].astype(self.output_dtype)

            # Tạo tensor đầu ra
            output_tensor = pb_utils.Tensor("input_ids", input_ids)
            logger.log_info(f"[EN2VI_PREPROCESS] [Output 1]: {output_tensor.shape()}")
            # Tạo response và thêm vào danh sách
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses

    def finalize(self):
        """Dọn dẹp khi mô hình bị unload."""
        print("Cleaning up...")