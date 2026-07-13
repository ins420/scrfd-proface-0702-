"""
CameraProcessor — 카메라 캡처 + SCRFD 탐지 + ArcFace 인식 + INN 익명화
백그라운드 스레드에서 처리 후 MJPEG용 JPEG 버퍼를 유지한다.
"""
import io
import os
import threading
import sqlite3
import time
import queue
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from insightface.app import FaceAnalysis
from insightface.utils import face_align

import config as c
from core.anonymizer import INNAnonymizer

DB_PATH = "security_system.db"

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Ubuntu
    "C:/Windows/Fonts/NanumGothic.ttf",                  # Windows (나눔고딕 설치)
    "C:/Windows/Fonts/malgun.ttf",                        # Windows 맑은 고딕
    "C:/Windows/Fonts/gulim.ttc",                         # Windows 굴림
]


def _load_font(size: int) -> ImageFont.ImageFont:
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _put_text(frame: np.ndarray, text: str, pos: tuple, size: int, color_bgr: tuple) -> np.ndarray:
    b, g, r = color_bgr
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    ImageDraw.Draw(pil).text(pos, text, font=_load_font(size), fill=(r, g, b))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# SQLite ↔ numpy 어댑터
def _adapt_array(arr: np.ndarray) -> sqlite3.Binary:
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    return sqlite3.Binary(buf.read())


def _convert_array(data: bytes) -> np.ndarray:
    buf = io.BytesIO(data)
    buf.seek(0)
    return np.load(buf)


sqlite3.register_adapter(np.ndarray, _adapt_array)
sqlite3.register_converter("array", _convert_array)


def _load_db() -> list:
    try:
        conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        rows = conn.execute("SELECT name, auth_group, vector FROM users").fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"[DB] 로드 실패: {e}")
        return []


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else -1.0


class CameraProcessor:
    """
    단일 카메라를 백그라운드 스레드로 처리.
    - SCRFD 탐지 → ArcFace 인식 → 사원/외부인 분기
    - 외부인: INN 익명화 + 빨간 박스
    - 사원: 권한 컬러 박스 + 이름
    - get_jpeg(): 최신 처리 프레임을 JPEG bytes로 반환
    """

    def _load_models(self):
        """Hailo-8L NPU 가속기 및 CPU 폴백 모델 로드"""
        use_hailo = getattr(c, "USE_HAILO", False)
        if use_hailo:
            try:
                from hailo_infer import HAILO_AVAILABLE, HailoSCRFD, HailoArcFace
                if not HAILO_AVAILABLE:
                    raise RuntimeError("hailo_platform 미설치")
                det = HailoSCRFD(
                    c.SCRFD_HEF_PATH,
                    conf_thresh=getattr(c, "HAILO_DET_THRESH", 0.5),
                )
                rec = HailoArcFace(c.ARCFACE_HEF_PATH)
                print("[CameraProcessor] ⚡ Hailo-8L 가속 사용 (SCRFD+ArcFace)")
                return det, rec
            except Exception as e:
                print(f"[CameraProcessor] Hailo 사용 불가({e}) → insightface 폴백")

        fa = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
        fa.prepare(ctx_id=-1, det_thresh=0.6)
        print("[CameraProcessor] insightface(CPU) 사용")
        return fa.models["detection"], fa.models["recognition"]

    def __init__(self):
        print("[CameraProcessor] 모델 로드 중...")
        self.detector, self.recognizer = self._load_models()

        if c.INN_CHECKPOINT:
            self._anonymizer = INNAnonymizer(checkpoint_path=c.INN_CHECKPOINT)
            print(f"[CameraProcessor] INN 로드: {c.INN_CHECKPOINT}")
        else:
            self._anonymizer = None
            print("[CameraProcessor] INN 체크포인트 없음 → 모자이크 익명화 사용")
        self._password = c.DEMO_PASSWORD

        self._db_lock = threading.Lock()
        self._db_users: list = []
        self.reload_db()

        self._frame_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_raw_jpeg: bytes | None = None  # 익명화 전 원본 (등록용)

        self._tiles_lock = threading.Lock()
        self._latest_tiles: list = []   # [{"tile_f32": ndarray, "crop_box": list}]

        # 디스크 대신 메모리 큐 사용 (INN 대기)
        self._pending_lock = threading.Lock()
        self._pending_records = []
        self._pending_max = int(
            getattr(c, "CHUNK_SECONDS", 60) * getattr(c, "SAVE_FPS", 1)
        ) or 1

        self._stats_lock = threading.Lock()
        self._stats = {"employee_count": 0, "unknown_count": 0, "recording": True}

        self._running = False
        print("[CameraProcessor] 준비 완료")

    # ── 공개 API ──────────────────────────────────────────────────────────

    def reload_db(self):
        with self._db_lock:
            self._db_users = _load_db()
        print(f"[DB] {len(self._db_users)}명 로드됨")

    def start(self, cam_id: int = 0):
        self._running = True
        t = threading.Thread(target=self._loop, args=(cam_id,), daemon=True)
        t.start()
        print(f"[CameraProcessor] 카메라 {cam_id} 시작")

    def stop(self):
        self._running = False

    def get_jpeg(self) -> bytes | None:
        with self._frame_lock:
            return self._latest_jpeg

    def get_raw_jpeg(self) -> bytes | None:
        """익명화 전 원본 프레임 (사원 등록용)."""
        with self._frame_lock:
            return self._latest_raw_jpeg

    def capture_raw_frame(self) -> "np.ndarray | None":
        """현재 원본 프레임을 디코딩해 ndarray로 반환 (등록 처리용)."""
        jpeg = self.get_raw_jpeg()
        if jpeg is None:
            return None
        arr = np.frombuffer(jpeg, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def get_stats(self) -> dict:
        with self._stats_lock:
            return dict(self._stats)

    def get_recording_snapshot(self) -> dict | None:
        """녹화용 스냅샷: 현재 JPEG + INN 타일 목록 반환."""
        with self._frame_lock:
            jpeg = self._latest_jpeg
        if jpeg is None:
            return None
        with self._tiles_lock:
            tiles = list(self._latest_tiles)
        return {"jpeg": jpeg, "tiles": tiles}

    def get_debug_info(self) -> dict:
        with self._frame_lock:
            size = len(self._latest_jpeg) if self._latest_jpeg else 0
        return {
            "running": self._running,
            "jpeg_size": size,
            "has_frame": size > 0,
            "db_users": len(self._db_users),
            "stats": self.get_stats(),
        }

    # ── 전용 프레임 리더 (블로킹 cap.read 격리) ──────────────────────────────

    def _start_frame_reader(self, cap) -> "queue.Queue":
        q: "queue.Queue" = queue.Queue(maxsize=2)

        def _reader():
            while True:
                ret, frame = cap.read()
                try:
                    q.put_nowait((ret, frame))
                except Exception:
                    pass  # 큐가 가득 찬 경우 최신 프레임 우선 → 버림

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        return q

    def _find_camera_index(self, preferred: int = 0) -> int:
        candidates = [preferred] + [i for i in range(9) if i != preferred]
        for idx in candidates:
            try:
                cap = cv2.VideoCapture(idx)
            except Exception:
                continue
            if not cap.isOpened():
                cap.release()
                continue
            found = False
            for _ in range(15):
                ret, f = cap.read()
                if (ret and f is not None and f.ndim == 3
                        and f.shape[2] == 3 and float(f.mean()) > 3):
                    found = True
                    break
            cap.release()
            if found:
                print(f"[Camera] 자동 선택: 인덱스 {idx} (컬러 영상 확인)")
                return idx
        print(f"[Camera] 컬러 카메라 자동탐색 실패 → 인덱스 {preferred} 사용")
        return preferred

    # ── 캡처 루프 ─────────────────────────────────────────────────────────

    def _loop(self, cam_id: int):
        if getattr(c, "FORCE_VIDEO", False):
            fallback = getattr(c, "VIDEO_FALLBACK", None)
            if fallback and os.path.exists(fallback):
                print(f"[Camera] FORCE_VIDEO=True → 영상 재생: {fallback}")
                self._video_loop(fallback)
                return

        if getattr(c, "CAMERA_TYPE", "webcam") == "realsense":
            self._realsense_loop()
            return

        cam_id = self._find_camera_index(cam_id)
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print(f"[Camera] 카메라 {cam_id} 열기 실패 → 5초 후 재시도")
            self._show_reconnecting()
            time.sleep(5)
            self._loop(cam_id)
            return
            
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        print(f"[Camera] 카메라 {cam_id} 열림")

        state = {"frame": None, "run": True, "first": True}
        rlock = threading.Lock()

        save_fps = getattr(c, "SAVE_FPS", 1)
        save_dt = (1.0 / save_fps) if save_fps and save_fps > 0 else 0.0

        def _reader():
            last_save = 0.0
            while state["run"] and self._running:
                ret, f = cap.read()
                if not ret or f is None:
                    continue
                f = cv2.flip(f, 1)
                
                _okr, _bufr = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if _okr:
                    with self._frame_lock:
                        self._latest_raw_jpeg = _bufr.tobytes()
                with rlock:
                    state["frame"] = f

                now = time.time()
                with self._pending_lock:
                    if (save_dt == 0.0 or (now - last_save) >= save_dt) \
                            and len(self._pending_records) < self._pending_max:
                        last_save = now
                        self._pending_records.append((f.copy(), now))

        threading.Thread(target=_reader, daemon=True).start()

        max_fps = getattr(c, "PROCESS_MAX_FPS", 15)
        min_dt = (1.0 / max_fps) if max_fps and max_fps > 0 else 0.0
        last_proc = 0.0
        while self._running:
            with rlock:
                frame = state["frame"]
                state["frame"] = None
                
            if frame is None:
                time.sleep(0.005)
                continue

            if state["first"]:
                state["first"] = False
                print(f"[Camera] 첫 프레임 수신 {frame.shape}")

            now = time.time()
            if min_dt > 0 and (now - last_proc) < min_dt:
                time.sleep(0.002)
                continue
            last_proc = now

            frame = self._maybe_downscale(frame)
            try:
                frame, emp, unk = self._process(frame)
            except Exception as e:
                print(f"[Camera] _process 오류 (건너뜀): {e}")
                emp, unk = 0, 0

            with self._stats_lock:
                self._stats["employee_count"] = emp
                self._stats["unknown_count"] = unk

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                with self._frame_lock:
                    self._latest_jpeg = buf.tobytes()

        state["run"] = False
        cap.release()
        print(f"[Camera] 카메라 {cam_id} 종료")

    def _realsense_loop(self):
        try:
            import pyrealsense2 as rs
        except ImportError:
            print("[Camera] pyrealsense2 미설치 → 'pip install pyrealsense2'")
            self._show_reconnecting()
            return

        W = getattr(c, "REALSENSE_WIDTH", 640)
        H = getattr(c, "REALSENSE_HEIGHT", 480)
        FPS = getattr(c, "REALSENSE_FPS", 30)

        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)

        try:
            pipeline.start(cfg)
        except Exception as e:
            print(f"[Camera] RealSense 시작 실패: {e} → 5초 후 재시도")
            self._show_reconnecting()
            time.sleep(5)
            self._realsense_loop()
            return

        print(f"[Camera] RealSense 시작 ({W}x{H} @ {FPS}fps)")
        every_n = max(1, getattr(c, "PROCESS_EVERY_N", 1))
        frame_count = 0
        try:
            while self._running:
                try:
                    frames = pipeline.wait_for_frames(2000)
                except Exception:
                    continue
                color = frames.get_color_frame()
                if not color:
                    continue

                frame = np.asanyarray(color.get_data())
                frame_count += 1
                if frame_count == 1:
                    print(f"[Camera] RealSense 첫 프레임 {frame.shape}")

                _okr, _bufr = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if _okr:
                    with self._frame_lock:
                        self._latest_raw_jpeg = _bufr.tobytes()

                if frame_count % every_n != 0:
                    continue

                frame = self._maybe_downscale(frame)
                try:
                    frame, emp, unk = self._process(frame)
                except Exception as e:
                    print(f"[Camera] _process 오류 (건너뜀): {e}")
                    emp, unk = 0, 0

                with self._stats_lock:
                    self._stats["employee_count"] = emp
                    self._stats["unknown_count"] = unk

                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    with self._frame_lock:
                        self._latest_jpeg = buf.tobytes()
        finally:
            pipeline.stop()
            print("[Camera] RealSense 종료")

    def _video_loop(self, video_path: str):
        fps = getattr(c, "VIDEO_FALLBACK_FPS", 25)
        delay = 1.0 / max(1, fps)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Video] 영상 열기 실패: {video_path} → 더미 모드")
            self._dummy_loop()
            return

        print(f"[Video] 폴백 영상 재생 시작 (목표 fps={fps})")
        frame_count = 0
        while self._running:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_count += 1
            try:
                frame, emp, unk = self._process(frame)
            except Exception as e:
                print(f"[Video] _process 오류 (건너뜀): {e}")
                emp, unk = 0, 0

            cv2.putText(frame, f"DEMO (video) F{frame_count}",
                        (10, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

            with self._stats_lock:
                self._stats["employee_count"] = emp
                self._stats["unknown_count"] = unk

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                with self._frame_lock:
                    self._latest_jpeg = buf.tobytes()

            elapsed = time.time() - t0
            if elapsed < delay:
                time.sleep(delay - elapsed)

        cap.release()
        print("[Video] 폴백 영상 종료")

    def _show_reconnecting(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (30, 30, 40)
        cv2.putText(frame, "Reconnecting...", (160, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 140, 200), 2)
        cv2.putText(frame, "Camera disconnected. Retrying in 5s", (60, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 100), 1)
        _, buf = cv2.imencode(".jpg", frame)
        with self._frame_lock:
            self._latest_jpeg = buf.tobytes()

    def _dummy_loop(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (30, 30, 40)
        cv2.putText(frame, "No Camera", (200, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 120), 2)
        cv2.putText(frame, "Connect webcam & restart server", (70, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 100), 1)
        _, buf = cv2.imencode(".jpg", frame)
        jpeg = buf.tobytes()
        with self._frame_lock:
            self._latest_jpeg = jpeg
        while self._running:
            time.sleep(1)

    # ── 모자이크 익명화 (Gaussian Blur) ──────────────────────────────

    def _mosaic(self, frame: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        bx1, by1 = max(0, int(x1)), max(0, int(y1))
        bx2, by2 = min(w, int(x2)), min(h, int(y2))
        
        if bx2 > bx1 and by2 > by1:
            roi = out[by1:by2, bx1:bx2]
            if roi.size > 0:
                blurred = cv2.GaussianBlur(roi, (99, 99), 30)
                out[by1:by2, bx1:bx2] = blurred
        return out

    # ── 프레임 처리 ───────────────────────────────────────────────────────

    def _maybe_downscale(self, frame: np.ndarray) -> np.ndarray:
        pw = getattr(c, "PROCESS_WIDTH", 0)
        if pw and frame.shape[1] > pw:
            scale = pw / frame.shape[1]
            frame = cv2.resize(frame, (pw, int(frame.shape[0] * scale)))
        return frame

    def _process(self, frame: np.ndarray) -> tuple[np.ndarray, int, int]:
        bboxes, kpss = self.detector.detect(frame, max_num=0, metric="default")
        
        # 사람이 없을 때 불필요한 연산 방지
        if bboxes is None or len(bboxes) == 0:
            return frame, 0, 0

        anonymize_all = getattr(c, "ANONYMIZE_ALL", False)
        emp, unk = 0, 0
        for i in range(bboxes.shape[0]):
            x1, y1, x2, y2 = bboxes[i, :4].astype(int)
            lm = kpss[i]
            aligned = face_align.norm_crop(frame, landmark=lm, image_size=112)
            emb = self.recognizer.get_feat(aligned)
            name, group, sim = self._match(emb)

            if name == "Unknown":
                unk += 1
            else:
                emp += 1

            if name == "Unknown" or anonymize_all:
                frame = self._mosaic(frame, x1, y1, x2, y2)
            frame = self._draw(frame, x1, y1, x2, y2, name, group, sim)
        return frame, emp, unk

    # ── pending 메모리 큐 API ──────────────────────────────────────────────

    def pop_pending(self):
        """가장 오래된 완성 pending 항목의 (frame, ts) 반환 후 삭제. 없으면 None."""
        with self._pending_lock:
            if not self._pending_records:
                return None
            return self._pending_records.pop(0)

    def pending_size(self) -> int:
        with self._pending_lock:
            return len(self._pending_records)

    def oldest_pending_ts(self) -> float | None:
        """아직 처리 안 된 가장 오래된 대기 프레임의 촬영 시각. 큐 비면 None."""
        with self._pending_lock:
            if not self._pending_records:
                return None
            return self._pending_records[0][1]

    def make_protected(self, frame: np.ndarray) -> tuple[np.ndarray, list]:
        bboxes, kpss = self.detector.detect(frame, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0:
            return frame, []
        anonymize_all = getattr(c, "ANONYMIZE_ALL", False)
        out = frame
        tiles = []
        for i in range(bboxes.shape[0]):
            x1, y1, x2, y2 = bboxes[i, :4].astype(int)
            lm = kpss[i]
            aligned = face_align.norm_crop(out, landmark=lm, image_size=112)
            emb = self.recognizer.get_feat(aligned)
            name, group, sim = self._match(emb)
            if name == "Unknown" or anonymize_all:
                if self._anonymizer is not None:
                    try:
                        out, tile_f32, crop_box = self._anonymizer.protect_roi(
                            out, [x1, y1, x2, y2], self._password
                        )
                        tiles.append({"tile_f32": tile_f32, "crop_box": crop_box})
                    except Exception as e:
                        print(f"[INN] make_protected 실패 → 모자이크: {e}")
                        out = self._mosaic(out, x1, y1, x2, y2)
                else:
                    out = self._mosaic(out, x1, y1, x2, y2)
            out = self._draw(out, x1, y1, x2, y2, name, group, sim)
        return out, tiles

    def _match(self, emb) -> tuple[str, str, float]:
        best_name, best_group, best_sim = "Unknown", "비허가", -1.0
        with self._db_lock:
            if emb is not None and self._db_users:
                for db_name, db_group, db_vec in self._db_users:
                    s = _cosine_sim(emb, db_vec)
                    if s > best_sim:
                        best_sim = s
                        if s > c.MATCH_THRESHOLD:
                            best_name = db_name
                            best_group = db_group
        return best_name, best_group, best_sim

    def _draw(self, frame: np.ndarray, x1, y1, x2, y2, name, group, sim) -> np.ndarray:
        # 상태에 따른 텍스트와 색상 설정 (Unknown 기준 판별)
        if name != "Unknown":
            color = (0, 200, 0) # 초록색
            # 개인 이름 대신 '허가자'로 통일하고 유사도 표시
            label = f"허가자 ({sim:.2f})" 
        else:
            color = (0, 0, 220) # 빨간색
            # '외부인' 대신 '비허가자'로 변경하고 유사도 표시
            label = f"비허가자 ({sim:.2f})"

        # 네모 테두리 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 텍스트 위치 계산 및 기존 _put_text 함수로 한글 출력
        text_y = max(y1 - 28, 5)
        frame = _put_text(frame, label, (x1, text_y), 18, color)
        
        return frame