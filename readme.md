
# Computational Thinking

Đáp ứng yêu cầu:
- Frontend: **Streamlit**
- Dữ liệu bản đồ: **OpenStreetMap** (Nominatim/Overpass), tuyến đường **OSRM**
- Chatbot: **Ollama** (có thể chạy ở Google Colab và công khai qua Cloudflared/Pinggy)
- Tìm kiếm, lọc dữ liệu : **SerpApi**

## NOTE
```bash
Tạo tài khoản SerpApi rồi lấy key gán vào web. Hiện tại do Key SerpApi
có giới hạn lượt sử dụng nên mỗi người muốn chạy chương trình thì tự tạo
cho mình một Key riêng, sau khi hoàn thiện tất cả thì sẽ gán cố định một Key
để đơn giản hóa ứng dụng.
```
Tạo Key tại https://serpapi.com

```

Nếu chọn vị trí/nhập vị trí bị lỗi thì đợi khoảng 1 phút rồi thử lại.

```

Key API để test: fdcea49178237153de98821d877265b20649dadd015fddac5a28c2482873a7d3

```

## 1. Cài đặt

```bash
pip install -r requirements.txt
```

## 2. Chạy ứng dụng

```bash
streamlit run app.py
```

Truy cập URL hiển thị (thường là http://localhost:8501).

## 3. Chạy Ollama ở Google Colab (gợi ý nhanh)

Trong Colab:
```bash
Mở file model_LLM trên colab rồi tạo link pinggy, lựa chọn model bất kì.

## 4. DO OSRM mặc định bị lỗi nên tạo thêm OSRM riêng để chạy mượt hơn
Yêu cầu trước khi chạy:
Lên mạng tải Docker hình con cá voi, tải xong bật nó lên

Tạo một Folder trống.
Bỏ file setup_osrm.py vào.
Bật CMD lên rồi chạy: 
    python setup_osrm.py

Khúc dưới nó tự động làm

Ngồi đợi: Script sẽ tự động:

Tải bản đồ Việt Nam về thư mục osrm-data.

Gọi Docker để xử lý dữ liệu.

Kết quả: Khi chạy xong, script sẽ in ra một dòng lệnh dài 
(bắt đầu bằng docker run -d...). Bạn chỉ cần Copy dòng lệnh đó 
và Paste vào terminal để server bắt đầu chạy.

Để test đã hoạt động hay chưa thì gõ lệnh:
    curl "http://localhost:5000/route/v1/driving/105.854444,21.028511;105.804817,21.028511?steps=true"
Nếu hiện code OK là thành công 

CÓ ẢNH CHO THẤY ĐÃ LÀM THÀNH CÔNG TAO GỬI KÈM ĐÓ.

Sau này muốn bật localhost thì chỉ cần bật docker(hỏi gemini).
```

## 4. Tính năng chính
- **Model** : gán link và tên model để sử dụng chatbot.
- **Chọn vị trí** : chọn từ bản đồ hoặc nhập từ bàn phím(chưa biết làm GPS).
- **Lên lịch trình** : thời gian bảo đảm tính logic.
- **Tìm kiếm và kết quả** : tìm kiếm dựa vào dữ liệu **SerApi** để đưa ra danh sách các gợi ý, OSRM để vẽ các tuyến đường.
- **Lưu và xuất lịch trình** : lưu lịch trình hiện tại và có thể mở lại lịch trình cũ để chỉnh sửa, xuất lịch trình thành file CSV, TXT và JSON.
