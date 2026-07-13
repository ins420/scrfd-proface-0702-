"""
INN 익명화 래퍼 — ONNXRuntime 최적화 버전 (Phase 3)
SCRFD가 미리 탐지한 bbox를 받아 처리.
"""
import warnings
import numpy as np
import torch
import onnxruntime as ort

import config as c
from utils.key_gen import generate_key, make_key_rec
from utils.image_processing import Obfuscator, to_tensor, to_numpy
from detection.yolo_detector import expand_bbox_square, crop_and_resize, paste_back

class INNAnonymizer:
    def __init__(self, checkpoint_path=None, device=None, obf_type="blur"):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        if checkpoint_path and checkpoint_path.endswith(".onnx"):
            self.session = ort.InferenceSession(
                checkpoint_path, 
                sess_options=opts, 
                providers=['CPUExecutionProvider']
            )
            self.input_names = [ipt.name for ipt in self.session.get_inputs()]
            print(f"[INNAnonymizer] 최적화 ONNX 로드 성공: {checkpoint_path}")
        else:
            raise ValueError(
                f"[INNAnonymizer] 올바른 ONNX 모델 경로가 필요합니다: {checkpoint_path}\n"
            )

        from models.modules import DWT
        self.dwt = DWT().to(self.device)
        self.obfuscator = Obfuscator(obf_type=obf_type)
        print(f"[INNAnonymizer] 준비 완료 (ONNX 엔진 가동)")

    def _prepare_onnx_inputs(self, in_1_tensor, in_2_tensor, skey_tensor, rev: bool) -> dict:
        in_1_np = in_1_tensor.cpu().numpy()
        in_2_np = in_2_tensor.cpu().numpy()
        skey_np = skey_tensor.cpu().numpy()

        inputs = {
            self.input_names[0]: in_1_np,
            self.input_names[1]: in_2_np,
            self.input_names[2]: skey_np
        }

        if len(self.input_names) >= 4:
            rev_name = self.input_names[3]
            rev_type = self.session.get_inputs()[3].type
            if "bool" in rev_type:
                inputs[rev_name] = np.array([rev], dtype=np.bool_)
            elif "int" in rev_type:
                inputs[rev_name] = np.array([1 if rev else 0], dtype=np.int32)
            else:
                inputs[rev_name] = np.array([1.0 if rev else 0.0], dtype=np.float32)
                
        return inputs

    def protect_roi(self, frame: np.ndarray, bbox: list, password) -> tuple:
        H, W = frame.shape[:2]
        crop_box = expand_bbox_square(list(bbox), H, W)
        face_np, _ = crop_and_resize(frame, crop_box, c.NORM_RESOLUTION)

        xa = to_tensor(face_np, device=self.device)
        xa_obfs = self.obfuscator(xa)

        skey = generate_key(
            password, bs=1, w=c.NORM_RESOLUTION, h=c.NORM_RESOLUTION
        ).to(self.device)
        skey_dwt = self.dwt(skey.float())

        onnx_inputs = self._prepare_onnx_inputs(xa, xa_obfs, skey_dwt, rev=False)
        outputs = self.session.run(None, onnx_inputs)
        
        xa_proc_np = outputs[1]
        xa_proc = torch.from_numpy(xa_proc_np).to(self.device)

        ya_hat_np = to_numpy(xa_proc.cpu())
        modified_frame = paste_back(frame, ya_hat_np, crop_box)
        tile_f32 = xa_proc.cpu().squeeze(0).numpy()

        return modified_frame, tile_f32, crop_box

    def restore_roi(
        self, frame: np.ndarray, tile_f32: np.ndarray, crop_box: list, password
    ) -> np.ndarray:
        norm_res = c.NORM_RESOLUTION
        xa_proc = torch.from_numpy(tile_f32).unsqueeze(0).to(self.device)

        skey = generate_key(
            password, bs=1, w=norm_res, h=norm_res
        ).to(self.device)
        skey_dwt = self.dwt(skey.float())
        key_rec = make_key_rec(skey_dwt)

        onnx_inputs = self._prepare_onnx_inputs(key_rec, xa_proc, skey_dwt, rev=True)
        outputs = self.session.run(None, onnx_inputs)
        
        xa_rev_np = outputs[0]
        xa_rev = torch.from_numpy(xa_rev_np).to(self.device)

        x_rec_np = to_numpy(xa_rev.cpu())
        return paste_back(frame, x_rec_np, crop_box)

