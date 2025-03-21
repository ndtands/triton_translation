# coding=utf-8
import numpy as np
import tritonclient.grpc as grpcclient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Translation API")

# Cấu hình Triton gRPC client
TRITON_SERVER_URL = "localhost:9091"
VI2EN_MODEL = "vi2en_ensemble"
EN2VI_MODEL = "en2vi_ensemble"

# Khởi tạo client
client = grpcclient.InferenceServerClient(url=TRITON_SERVER_URL, verbose=False)

# Định nghĩa request model
class TranslationRequest(BaseModel):
    texts: List[str]

def translate(texts: List[str], model_name: str) -> List[str]:
    # Chuẩn bị dữ liệu đầu vào
    texts = [[text] for text in texts]
    texts_np = np.array(texts, dtype=object)
    
    # Tạo InferInput
    input_name = "texts"
    input_shape = texts_np.shape
    infer_input = grpcclient.InferInput(input_name, input_shape, "BYTES")
    infer_input.set_data_from_numpy(texts_np)
    
    # Thực hiện inference
    try:
        results = client.infer(
            model_name=model_name,
            inputs=[infer_input]
        )
        translated_texts = results.as_numpy("translated_texts")
        return [item.decode() for item in translated_texts]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

@app.post("/vi2en", response_model=List[str])
async def vietnamese_to_english(request: TranslationRequest):
    """
    Dịch từ tiếng Việt sang tiếng Anh
    """
    return translate(request.texts, VI2EN_MODEL)

@app.post("/en2vi", response_model=List[str])
async def english_to_vietnamese(request: TranslationRequest):
    """
    Dịch từ tiếng Anh sang tiếng Việt
    """
    return translate(request.texts, EN2VI_MODEL)

@app.get("/health")
async def health_check():
    """
    Kiểm tra trạng thái service
    """
    try:
        is_available = client.is_server_live()
        return {"status": "healthy" if is_available else "unhealthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8094)