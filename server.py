from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from newspaper import Article

app = FastAPI()

# Cấu hình CORS để giao diện HTML có thể gọi được API ngầm này
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === SỬA ĐỔI 1: NẠP MODEL LOCAL VÀ PHÂN PHỐI PHẦN CỨNG ĐẦY ĐỦ ===
print("🤖 Đang nạp bộ não mạng Neural BERT đã được huấn luyện chuyên sâu...")
model_name = "fakenews_bert_model"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Khai báo cấu hình phần cứng (Sửa lỗi: name 'device' is not defined)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device) 
print(f"-> Hệ thống đang kích hoạt lõi phần cứng: {device.upper()}")


class ScanRequest(BaseModel):
    type: str  # "url" hoặc "text"
    content: str


def scrape_url(url):
    try:
        # Khởi tạo đối tượng báo chí tự động lọc nhiễu
        article = Article(url)
        article.download()
        article.parse()
        # Chỉ lấy duy nhất nội dung chữ cốt lõi của bài báo, bỏ qua quảng cáo/menu
        return article.text.strip()
    except Exception as e:
        print(f"Lỗi cào nâng cao: {str(e)}")
        return ""

@app.post("/api/scan")
async def scan_news(req: ScanRequest):
    try:
        # 1. Thu thập văn bản ban đầu
        text = req.content if req.type == "text" else scrape_url(req.content)
        
        if not text:
            raise HTTPException(status_code=400, detail="Không thể trích xuất nội dung.")
            
        # Ép kiểu dữ liệu sang chuỗi và làm sạch khoảng trắng một cách tường minh
        text_str = str(text).strip()
        
        if len(text_str) < 10:
            raise HTTPException(status_code=400, detail="Nội dung mục tiêu quá ngắn.")
            
        print(f"\n[HỆ THỐNG] Đang tiến hành quét nội dung: {text_str[:50]}...")

        # 2. Xử lý mã hóa Token mã hóa cho BERT (Biến device đã hoạt động thông suốt)
        inputs = tokenizer(
            text_str, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=256
        ).to(device)
        
        # 3. Đẩy qua mạng Neural BERT dự đoán
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        # 4. Tính toán xác suất phần trăm
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        confidence = float(probs[pred] * 100)
        
        print(f"[KẾT QUẢ AI] Nhãn dự đoán: {pred} | Độ tự tin: {confidence:.2f}%")
        
        return {
            "status": "REAL" if pred == 1 else "FAKE",
            "confidence": round(confidence, 2),
            "preview": text_str[:150] + "..."
        }
        
    except Exception as e:
        print(f"❌ LỖI HỆ THỐNG PHÁT SINH: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)