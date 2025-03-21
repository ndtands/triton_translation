import json
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer
import numpy as np

MODEL_PATH = "/workspace/model_repository/pretrain/vi2en"

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
            src_lang="vi_VN"
        )

        # Lấy cấu hình đầu ra từ model_config
        output_config = pb_utils.get_output_config_by_name(self.model_config, "translated_texts")

        # Chuyển đổi kiểu dữ liệu từ Triton sang numpy
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    def execute(self, requests):
        """Xử lý các request đầu vào và trả về văn bản đã dịch.

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
            # Lấy dữ liệu đầu vào từ tensor "output_ids"
            output_ids = pb_utils.get_input_tensor_by_name(request, "output_ids").as_numpy()
            logger.log_info(f"[VI2EN_POSTPROCESS] [Input 1]: {output_ids.shape}")
            # Giải mã output_ids thành văn bản
            translated_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            translated_texts = np.array(translated_texts, dtype=self.output_dtype)

            # Tạo tensor đầu ra
            output_tensor = pb_utils.Tensor("translated_texts", translated_texts)

            # Tạo response và thêm vào danh sách
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses

    def finalize(self):
        """Dọn dẹp khi mô hình bị unload."""
        print("Cleaning up...")