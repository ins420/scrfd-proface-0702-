import os
import sqlite3
import base64
import io
import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ==========================================
# 1. 설정 및 SQLite <-> Numpy 변환 어댑터
# ==========================================
IMAGE_DIR = "registered_faces"
DB_PATH = "security_system.db"
os.makedirs(IMAGE_DIR, exist_ok=True)

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

def init_db():
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            auth_group TEXT NOT NULL,
            image_path TEXT NOT NULL,
            vector array NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ==========================================
# 2. AI 모델 로드 & UI 그리기 도우미 함수 (강제 int 변환 적용)
# ==========================================
_SCRFD_DETECTOR = None
_ARCFACE_RECOGNIZER = None

def get_models():
    global _SCRFD_DETECTOR, _ARCFACE_RECOGNIZER
    if _SCRFD_DETECTOR is None or _ARCFACE_RECOGNIZER is None:
        print("🧠 모델 로드 중...")
        app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_thresh=0.6)
        _SCRFD_DETECTOR = app.models['detection']
        _ARCFACE_RECOGNIZER = app.models['recognition']
        print("✅ 모델 로드 완료!")
    return _SCRFD_DETECTOR, _ARCFACE_RECOGNIZER

get_models()

def draw_green_circle(img, center, size=7):
    cx, cy = int(center[0]), int(center[1])
    cv2.circle(img, (cx, cy), int(size), (0, 255, 0), -1)

def draw_yellow_triangle(img, center, size=7):
    cx, cy = int(center[0]), int(center[1])
    s = int(size)
    pts = np.array([
        [cx, cy - s],
        [cx - s, cy + s],
        [cx + s, cy + s]
    ], np.int32)
    cv2.fillPoly(img, [pts], (0, 255, 255))

def draw_red_x(img, center, size=6, thickness=2):
    cx, cy = int(center[0]), int(center[1])
    s = int(size)
    cv2.line(img, (cx - s, cy - s), (cx + s, cy + s), (0, 0, 255), int(thickness))
    cv2.line(img, (cx + s, cy - s), (cx - s, cy + s), (0, 0, 255), int(thickness))

# ==========================================
# 3. FastAPI 웹 서버 및 순서 적용 API
# ==========================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

class RegisterData(BaseModel):
    name: str
    group: str
    image_base64: str

@app.get("/", response_class=HTMLResponse)
async def serve_webpage(request: Request):
    return templates.TemplateResponse(request = request, name="index.html")

@app.post("/api/register")
async def register_person(data: RegisterData):
    try:
        detector, recognizer = get_models()

        header, encoded = data.image_base64.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        bboxes, kpss = detector.detect(frame, max_num=1, metric='default')

        if bboxes is None or len(bboxes) == 0:
            h, w = frame.shape[:2]
            cx, cy = int(w // 2), int(h // 2)
            draw_red_x(frame, (cx, cy), size=50, thickness=10)
            cv2.putText(frame, "No Face Detected", (cx - 120, cy + 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return {"status": "error", "message": "❌ 사진에서 얼굴을 찾을 수 없습니다."}

        # ⭐️ 에러 원인 차단: 모든 좌표를 철저하게 int()로 변환
        x1, y1, x2, y2 = int(bboxes[0, 0]), int(bboxes[0, 1]), int(bboxes[0, 2]), int(bboxes[0, 3])
        conf = float(bboxes[0, 4])
        landmarks = kpss[0]

        left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
        dist_left = np.linalg.norm(left_eye - nose)
        dist_right = np.linalg.norm(right_eye - nose)
        ratio = max(dist_left, dist_right) / (min(dist_left, dist_right) + 1e-5)
        box_width = x2 - x1
        eye_dist = np.linalg.norm(left_eye - right_eye)
        is_extreme_side = (eye_dist / box_width) < 0.25
        is_side_face = (ratio > 1.5) or is_extreme_side

        face_aligned = face_align.norm_crop(frame, landmark=landmarks, image_size=112)
        embedding = recognizer.get_feat(face_aligned)

        if embedding is None:
            return {"status": "error", "message": "❌ 얼굴 특징 벡터를 추출할 수 없습니다."}

        if is_side_face:
            state_text = "Side"
            color = (0, 255, 255)  
        else:
            state_text = "Frontal"
            color = (0, 255, 0)    

        # ⭐️ 에러 원인 차단 2: 텍스트 및 UI 좌표를 명시적 정수로 캐스팅
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        ui_y = int(max(y1 - 12, 15))
        marker_center = (int(x1 + 10), ui_y - 4)
        text_start_x = int(x1 + 25)

        if state_text == "Side":
            draw_yellow_triangle(frame, marker_center)
        else:
            draw_green_circle(frame, marker_center)

        display_text = str(f"{state_text} {conf:.2f}")
        cv2.putText(frame, display_text, (text_start_x, ui_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if state_text == "Frontal":
            file_name = f"{data.name}_정면.jpg"
            msg_tag = "정면"
        else:
            if dist_left < dist_right:
                file_name = f"{data.name}_측면1(좌).jpg"
                msg_tag = "좌측면"
            else:
                file_name = f"{data.name}_측면2(우).jpg"
                msg_tag = "우측면"

       # ... (위쪽 각도 판별 로직은 그대로 둠) ...

        file_path = os.path.join(IMAGE_DIR, file_name)
        cv2.imwrite(file_path, frame)

        # ⭐️ 핵심 추가: DB에 들어갈 이름을 "이름_방향" 형태로 조합합니다.
        db_store_name = f"{data.name}_{msg_tag}"

        # SQLite DB에 데이터 저장
        conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        cursor = conn.cursor()
        
        # ⭐️ 삽입 로직 변경: data.name 대신 db_store_name을 넣습니다.
        cursor.execute(
            "INSERT INTO users (name, auth_group, image_path, vector) VALUES (?, ?, ?, ?)",
            (db_store_name, data.group, file_path, embedding)
        )
        conn.commit()
        conn.close()

        return {"status": "success", "message": f"✅ [{data.name}]님의 [{msg_tag}] 얼굴 등록 성공! ({file_name})"}
    
    except Exception as e:
        return {"status": "error", "message": f"서버 오류: {str(e)}"}
