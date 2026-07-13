"""
PSFRecorder — 1시간 단위 PSF 청크 녹화 (Phase 3)

청크 구조:
  recordings/
    YYYY-MM-DD_HH/        ← 1시간 단위 청크
      manifest.json
      000001/             ← 스냅샷 (N초 간격)
        frame.jpg         ← 익명화된 프레임
        face_0.npy        ← float32 (3,256,256) 복원 타일
        face_0_box.json   ← crop_box [x1,y1,x2,y2]
      000002/
        ...
"""

import hashlib
import json
import os
import threading
import time
import shutil
from datetime import datetime
import cv2
import numpy as np
import config as c

RECORDINGS_DIR = getattr(c, "RECORD_RAM_DIR", "recordings")


# ── JSON 유틸 ──────────────────────────────────────────────────────────────

def _load_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# ── 그림자 백업 (Shadow Backup) ──────────────────────────────────────────
def backup_chunk_to_sd(ram_chunk_path: str):
    """메인 시스템에 부하를 주지 않고 뒤에서 조용히 SD카드로 복사하는 thread"""
    ram_base = getattr(c, "RECORD_RAM_DIR", "recordings")
    sd_base = getattr(c, "RECORD_SD_DIR", "recordings")
    
    # 윈도우/맥처럼 RAM과 SD 경로가 동일하면 복사할 필요 없음
    if ram_base == sd_base:
        return

    def _copy_task():
        t_start = time.time()

        try:
            # RAM 경로에서 상대 경로(예: 2026-07/14/오후/14시/14-00-00)만 추출
            rel_path = os.path.relpath(ram_chunk_path, ram_base)
            sd_path = os.path.join(sd_base, rel_path)
            
            # SD카드 쪽에 폴더 만들고 통째로 덮어쓰기 복사
            os.makedirs(os.path.dirname(sd_path), exist_ok=True)
            shutil.copytree(ram_chunk_path, sd_path, dirs_exist_ok=True)
            # 💡 [추가] 복사 소요 시간 계산
            elapsed = time.time() - t_start
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            
            # 💡 [추가] 요구하신 "26-07-15 13시 00분" 포맷 생성
            now_str = datetime.now().strftime("%y-%m-%d %H시 %M분")
            print(f"[Backup] 💾 쉐도우 백업 완료 (SD카드 저장됨): {sd_path}")
        except Exception as e:
            print(f"[Backup Error] ❌ 쉐도우 백업 실패: {e}")

    # 데몬 스레드로 실행 (메인 서버가 꺼지면 같이 종료됨)
    threading.Thread(target=_copy_task, daemon=True).start()

def _json_default(o):
    """numpy 정수/실수/배열을 JSON 직렬화 가능 타입으로 변환."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def _save_json(path: str, data: dict):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
    os.replace(tmp_path, path)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _first_frame_jpg(chunk_path: str) -> str | None:
    for name in sorted(os.listdir(chunk_path)):
        fdir = os.path.join(chunk_path, name)
        fpath = os.path.join(fdir, "frame.jpg")
        if os.path.isdir(fdir) and os.path.exists(fpath):
            return fpath
    return None


# 계층 폴더(월/일/오전오후/시/10분청크)와 안전한 chunk_id 간 변환.
# 실제 경로:  recordings/2026-06/29/오후/14시/14-00
# chunk_id : "2026-06__29__오후__14시__14-00"  (슬래시 → __, URL 안전)
def _path_to_id(chunk_path: str) -> str:
    """
    RAM이나 SD 어디서든 기준 디렉토리를 떼어내어 순수한 상대 경로로 ID 생성.
    """
    # RAM 디스크 경로에 속해 있다면 RECORD_RAM_DIR 기준으로 상대경로 추출
    if getattr(c, "RECORD_RAM_DIR", "recordings") in chunk_path:
        rel = os.path.relpath(chunk_path, getattr(c, "RECORD_RAM_DIR", "recordings"))
    # SD 카드 경로에 속해 있다면 RECORD_SD_DIR 기준으로 추출
    else:
        rel = os.path.relpath(chunk_path, getattr(c, "RECORD_SD_DIR", "recordings"))
    return rel.replace(os.sep, "__").replace("/", "__")


def _id_to_path(chunk_id: str) -> str:
    """
    1순위로 RAM 디스크(오늘 데이터)에 파일이 있는지 확인하고,
    없으면 2순위로 SD 카드(과거 데이터)에서 파일을 찾아 반환.
    """
    rel = chunk_id.replace("__", os.sep)
    ram_path = os.path.join(getattr(c, "RECORD_RAM_DIR", "recordings"), rel)
    sd_path = os.path.join(getattr(c, "RECORD_SD_DIR", "recordings"), rel)
    return ram_path if os.path.exists(ram_path) else sd_path


# ── PSFRecorder ────────────────────────────────────────────────────────────

class PSFRecorder:
    """
    CameraProcessor에서 N초 간격으로 스냅샷을 받아 PSF 청크로 저장.
    1시간마다 새 청크 폴더 자동 생성.
    """

    def __init__(self, camera, interval_sec: int = 5):
        self._camera = camera
        self._interval = interval_sec
        self._running = False
        os.makedirs(RECORDINGS_DIR, exist_ok=True)

    def start(self):
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        print(f"[Recorder] 녹화 시작 (간격={self._interval}s)")

    def stop(self):
        self._running = False

    # ── 내부 루프 ──────────────────────────────────────────────────────────

    def _chunk_dir(self, ts: float | None = None) -> str:
        """
        월/일/오전오후/시/HH-MM-SS 청크 계층 폴더 경로를 만들고 반환.
        ts(촬영 시각)를 주면 그 시각 기준으로 청크를 분류 (recorder가
        뒤처져 나중에 처리해도 프레임이 올바른 시간대 청크에 저장됨).
        """
        import config as c
        secs = getattr(c, "CHUNK_SECONDS", 60)
        now = datetime.fromtimestamp(ts) if ts else datetime.now()
        month = now.strftime("%Y-%m")            # 2026-06
        day = now.strftime("%d")                 # 29
        ampm = "오전" if now.hour < 12 else "오후"
        hour = f"{now.hour:02d}시"               # 14시
        # 시(hour) 내 초 단위 버킷 → HH-MM-SS
        sec_of_hour = now.minute * 60 + now.second
        b = (sec_of_hour // secs) * secs
        chunk = f"{now.hour:02d}-{b // 60:02d}-{b % 60:02d}"  # 14-00-20
        path = os.path.join(RECORDINGS_DIR, month, day, ampm, hour, chunk)
        os.makedirs(path, exist_ok=True)
        return path

    def _loop(self):
        frame_id = 0
        last_chunk = None
        save_count = 0
        t_report = time.time()
        proc_ms = 0.0
        
        # 💡 [추가] 현재 청크의 시작 시간 기록
        chunk_start_time = time.time()
        
        while self._running:
            if self._interval > 0:
                time.sleep(self._interval)

            popped = self._camera.pop_pending()
            if popped is None:
                time.sleep(0.05)
                continue
            raw, ts = popped
            _t0 = time.time()
            anon_frame, tiles = self._camera.make_protected(raw)
            proc_ms = (time.time() - _t0) * 1000
            save_count += 1

            if time.time() - t_report >= 5.0:
                fps = save_count / (time.time() - t_report)
                qsize = self._camera.pending_size()
                print(f"[Recorder] 저장 {fps:.1f}fps (INN {proc_ms:.0f}ms/frame) "
                      f"대기 큐 {qsize}장")
                save_count = 0
                t_report = time.time()

            chunk = self._chunk_dir(ts)
            
            if chunk != last_chunk:
                if last_chunk is not None:
                    # 💡 [추가] RAM 디스크 녹화 완료 소요 시간 계산
                    elapsed_ram = time.time() - chunk_start_time
                    r_mins = int(elapsed_ram // 60)
                    r_secs = int(elapsed_ram % 60)
                    now_str = datetime.now().strftime("%y-%m-%d %H시 %M분")
                    
                    self._mark_complete(last_chunk)
                    # 💡 [추가] RAM 청크 완료 알림 출력
                    print(f"[{now_str}] 🐏 RAM 청크 저장 완료, {r_mins}분 {r_secs}초 소요")
                    
                    backup_chunk_to_sd(last_chunk)  # 섀도우 백업 시작
                
                # 💡 [추가] 다음 청크를 위한 타이머 리셋
                chunk_start_time = time.time()
                frame_id = 0
                last_chunk = chunk
                
            frame_id += 1
            snap_dir = os.path.join(chunk, f"{frame_id:06d}")
            tmp_dir = snap_dir + ".tmp"
            os.makedirs(tmp_dir, exist_ok=True)

            cv2.imwrite(os.path.join(tmp_dir, "frame.jpg"), anon_frame)
            _save_json(os.path.join(tmp_dir, "meta.json"), {"ts": ts})
            for i, td in enumerate(tiles):
                np.save(os.path.join(tmp_dir, f"face_{i}.npy"), td["tile_f32"])
                _save_json(os.path.join(tmp_dir, f"face_{i}_box.json"), td["crop_box"])
            os.replace(tmp_dir, snap_dir)

            mpath = os.path.join(chunk, "manifest.json")
            m = _load_json(mpath) or {
                "chunk_id": _path_to_id(chunk),
                "start_time": datetime.now().isoformat(),
                "frame_count": 0,
                "total_faces": 0,
            }
            m["frame_count"] += 1
            m["total_faces"] += len(tiles)
            m["last_update"] = datetime.now().isoformat()
            _save_json(mpath, m)

    def _mark_complete(self, chunk_path: str):
        """청크의 10분이 끝나 다음 청크로 넘어갈 때 완료 표시 (복원 허용)."""
        mpath = os.path.join(chunk_path, "manifest.json")
        m = _load_json(mpath) or {}
        m["complete"] = True
        _save_json(mpath, m)
        print(f"[Recorder] 청크 완료: {_path_to_id(chunk_path)}")

    @staticmethod
    def _chunk_end_ts(chunk_id: str) -> float | None:
        """청크의 끝 시각(epoch). chunk_id에서 파싱."""
        import config as c
        from datetime import timedelta
        try:
            parts = chunk_id.split("__")
            year, month = (int(x) for x in parts[0].split("-"))
            day = int(parts[1])
            hms = parts[4].split("-")   # "14-00-20"
            hh = int(hms[0]); mm = int(hms[1]); ss = int(hms[2]) if len(hms) > 2 else 0
            start = datetime(year, month, day, hh, mm, ss)
            end = start + timedelta(seconds=getattr(c, "CHUNK_SECONDS", 60))
            return end.timestamp()
        except Exception:
            return None

    def _is_complete(self, chunk_id: str) -> bool:
        """
        청크가 완료됐는지 = 그 청크의 모든 프레임이 이미 저장됨(pending에 없음).
        판정: 청크 끝시각 <= 아직 대기 중인 가장 오래된 프레임의 촬영시각.
        (recorder가 그 청크 시간대를 다 지나쳐 처리했다는 뜻)
        """
        end = self._chunk_end_ts(chunk_id)
        if end is None:
            return False
        oldest = self._camera.oldest_pending_ts()
        if oldest is None:
            # 대기 큐가 비었으면, 시간대가 지난 청크는 완료
            return end < datetime.now().timestamp()
        return end <= oldest

    # ── 공개 API ──────────────────────────────────────────────────────────

    def list_chunks(self) -> list[dict]:
        """
        RAM 디스크와 SD 카드를 모두 스캔하여 중복 없이 하나의 통합 청크 목록을 반환.
        """
        merged_chunks = {}
        ram_base = getattr(c, "RECORD_RAM_DIR", "recordings")
        sd_base = getattr(c, "RECORD_SD_DIR", "recordings")

        # 헬퍼 함수: 특정 디렉토리를 긁어 중복 제거하며 딕셔너리에 추가
        def scan_directory(base_dir: str):
            if not os.path.exists(base_dir):
                return
            
            # 계층 폴더 스캔 시작
            for month_dir in sorted(os.listdir(base_dir), reverse=True):
                m_path = os.path.join(base_dir, month_dir)
                if not os.path.isdir(m_path): continue
                
                for day_dir in sorted(os.listdir(m_path), reverse=True):
                    d_path = os.path.join(m_path, day_dir)
                    if not os.path.isdir(d_path): continue

                    for ampm in ["오후", "오전"]:
                        a_path = os.path.join(d_path, ampm)
                        if not os.path.exists(a_path): continue
                        
                        for hour_dir in sorted(os.listdir(a_path), reverse=True):
                            h_path = os.path.join(a_path, hour_dir)
                            if not os.path.isdir(h_path): continue
                            
                            for chunk_name in sorted(os.listdir(h_path), reverse=True):
                                c_path = os.path.join(h_path, chunk_name)
                                mpath = os.path.join(c_path, "manifest.json")
                                if os.path.exists(mpath):
                                    m = _load_json(mpath) or {}
                                    cid = _path_to_id(c_path)
                                    m["chunk_id"] = cid
                                    m["has_thumb"] = _first_frame_jpg(c_path) is not None
                                    m["complete"] = m.get("complete", False) or self._is_complete(cid)
                                    
                                    # 이미 딕셔너리에 동일한 cid가 등록되어 있다면 건너뛰거나 덮어쓰기
                                    # (RAM 스캔을 나중에 돌릴 것이므로 RAM 데이터가 자연스럽게 우선 적용됩니다)
                                    merged_chunks[cid] = m

        # 1단계: 영구 보관용 SD 카드 먼저 스캔 (과거 데이터 로드)
        if ram_base != sd_base:
            scan_directory(sd_base)
            
        # 2단계: 실시간 RAM 디스크 스캔 (오늘 데이터로 덮어쓰기 및 추가)
        scan_directory(ram_base)

        # 결과 리스트 변환 및 최신순 정렬
        result = list(merged_chunks.values())
        result.sort(key=lambda x: x.get("chunk_id", ""), reverse=True)
        
        # 넉넉하게 최근 200개 반환 (성능을 조율하기 위해 필요에 따라 조절)
        return result

    def get_chunk_detail(self, chunk_id: str) -> dict | None:
        path = _id_to_path(chunk_id)
        mpath = os.path.join(path, "manifest.json")
        if not os.path.exists(mpath):
            return None
        m = _load_json(mpath) or {}
        m["chunk_id"] = chunk_id
        m["complete"] = m.get("complete", False) or self._is_complete(chunk_id)
        frames = []
        if os.path.isdir(path):
            for fname in sorted(os.listdir(path)):
                fdir = os.path.join(path, fname)
                if not (os.path.isdir(fdir) and fname.isdigit()):
                    continue
                files = os.listdir(fdir)
                npys = [f for f in files if f.endswith(".npy")]
                frames.append({
                    "frame_id": fname,
                    "face_count": len(npys),
                    "has_faces": len(npys) > 0,
                })
        m["frames"] = frames
        return m

    def get_thumb_jpeg(self, chunk_id: str) -> bytes | None:
        path = _id_to_path(chunk_id)
        jpg = _first_frame_jpg(path)
        if jpg is None:
            return None
        with open(jpg, "rb") as f:
            return f.read()

    def get_frame_jpeg(self, chunk_id: str, frame_id: str) -> bytes | None:
        p = os.path.join(_id_to_path(chunk_id), frame_id, "frame.jpg")
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            return f.read()

    def restore_chunk_video(self, chunk_id: str, password: str) -> str | None:
        """
        청크 내 모든 프레임을 복원해 mp4로 합쳐 경로 반환.
        원본 이미지는 사용하지 않고 저장된 tile_f32(보호본)에서 복원.
        """
        import config as c
        from core.anonymizer import INNAnonymizer

        path = _id_to_path(chunk_id)
        if not os.path.isdir(path):
            return None
        frame_dirs = sorted(d for d in os.listdir(path) if d.isdigit())
        if not frame_dirs:
            return None

        anon = None
        if c.INN_CHECKPOINT is not None:
            anon = INNAnonymizer(checkpoint_path=c.INN_CHECKPOINT)

        # 출력 재생 fps (부드러움용). 실제 시간 길이는 타임스탬프로 맞춤.
        out_fps = max(1, getattr(c, "RESTORE_VIDEO_FPS", 10))
        raw_path = os.path.join(path, "restored_raw.mp4")  # mp4v (브라우저 비호환)
        out_path = os.path.join(path, "restored.mp4")       # H.264 (브라우저 호환)
        writer = None

        # 각 프레임의 촬영 시각(ts) 수집 → 프레임 간 실제 간격 계산
        ts_list = []
        for fid in frame_dirs:
            meta = _load_json(os.path.join(path, fid, "meta.json"))
            ts_list.append(meta.get("ts", 0.0) if meta else 0.0)

        for idx, fid in enumerate(frame_dirs):
            snap = os.path.join(path, fid)
            frame = cv2.imread(os.path.join(snap, "frame.jpg"))
            if frame is None:
                continue

            if anon is not None:
                for i in range(20):
                    npy_path = os.path.join(snap, f"face_{i}.npy")
                    box_path = os.path.join(snap, f"face_{i}_box.json")
                    if not os.path.exists(npy_path):
                        break
                    try:
                        tile_f32 = np.load(npy_path)  # 깨진 파일이면 예외
                        crop_box = _load_json(box_path)
                        frame = anon.restore_roi(frame, tile_f32, crop_box, password)
                    except Exception as e:
                        print(f"[RestoreVideo] {fid} face_{i} 건너뜀: {e}")
                        continue
            else:
                cv2.putText(frame, "INN checkpoint required", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    raw_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h)
                )

            # 실제 시간 반영: 다음 프레임까지의 간격만큼 이 화면을 유지(hold)
            if idx < len(frame_dirs) - 1 and ts_list[idx] > 0 and ts_list[idx + 1] > 0:
                dur = ts_list[idx + 1] - ts_list[idx]
            else:
                chunk_sec = getattr(c, "CHUNK_SECONDS", 600)
                if len(ts_list) > 0 and ts_list[0] > 0:
                    elapsed = ts_list[-1] - ts_list[0]
                    dur = max(1.0 / out_fps, chunk_sec - elapsed)
                else:
                    dur = 1.0 / out_fps
            
            hold = max(1, round(dur * out_fps))
            for _ in range(hold):
                writer.write(frame)

        if writer is None:
            return None
        writer.release()

        # ffmpeg으로 H.264 + yuv420p 변환 (브라우저 재생 호환)
        import shutil
        import subprocess
        ffmpeg = getattr(c, "FFMPEG_PATH", None)
        if not ffmpeg or not os.path.exists(ffmpeg):
            ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            try:
                subprocess.run(
                    [ffmpeg, "-y", "-i", raw_path,
                     "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                     "-movflags", "+faststart", out_path],
                    check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                print(f"[RestoreVideo] {chunk_id} H.264 변환 완료: {out_path}")
                return out_path
            except Exception as e:
                print(f"[RestoreVideo] ffmpeg 변환 실패 → mp4v 반환: {e}")
                return raw_path
        print("[RestoreVideo] ffmpeg 없음 → mp4v 반환 (브라우저 재생 안 될 수 있음)")
        return raw_path

    def restore_frame(
        self, chunk_id: str, frame_id: str, password: str
    ) -> bytes | None:
        """INN 역변환으로 익명화 얼굴 복원 → JPEG bytes 반환."""
        import config as c
        from core.anonymizer import INNAnonymizer

        snap_dir = os.path.join(_id_to_path(chunk_id), frame_id)
        frame_path = os.path.join(snap_dir, "frame.jpg")
        if not os.path.exists(frame_path):
            return None

        frame = cv2.imread(frame_path)
        if frame is None:
            return None

        if c.INN_CHECKPOINT is None:
            # 체크포인트 없음 → 원본 익명화 프레임 그대로 반환 + 워터마크
            cv2.putText(frame, "INN checkpoint required for restoration",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
            ok, buf = cv2.imencode(".jpg", frame)
            return buf.tobytes() if ok else None

        anon = INNAnonymizer(checkpoint_path=c.INN_CHECKPOINT)
        for i in range(20):
            npy_path = os.path.join(snap_dir, f"face_{i}.npy")
            box_path = os.path.join(snap_dir, f"face_{i}_box.json")
            if not os.path.exists(npy_path):
                break
            tile_f32 = np.load(npy_path)
            crop_box = _load_json(box_path)
            try:
                frame = anon.restore_roi(frame, tile_f32, crop_box, password)
            except Exception as e:
                print(f"[Restore] face_{i} 복원 실패: {e}")

        ok, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes() if ok else None
