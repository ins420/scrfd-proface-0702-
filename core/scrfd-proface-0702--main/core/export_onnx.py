import sys
import os
import torch
import warnings

# 상위 폴더(SCRFD-PROFACE-MAIN)의 모듈들을 정상적으로 불러오도록 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config as c
from models.embedder import ModelDWT
from models.modules import DWT
from utils.key_gen import generate_key

# 1. ONNX 변환용 래퍼(Wrapper) 클래스
class INNProtectWrapper(torch.nn.Module):
    def __init__(self, embedder):
        super().__init__()
        self.embedder = embedder

    def forward(self, xa, xa_obfs, skey_dwt):
        xa_out_z, xa_proc = self.embedder(xa, xa_obfs, skey_dwt, rev=False)
        return xa_proc

def convert_to_onnx():
    checkpoints_dir = os.path.join(parent_dir, "checkpoints")
    
    # 👇 수정된 부분: 올려주신 실제 가중치 파일명으로 정확히 변경했습니다.
    pth_filename = "hybridAll_inv3_recTypeRandom_secretAsNoise_TripMargin1.2_ep12_iter15000.pth"
    
    pth_path = os.path.join(checkpoints_dir, pth_filename)
    onnx_path = os.path.join(checkpoints_dir, "inn_protect.onnx")

    device = torch.device("cpu")
    print(f"\n[ONNX 추출기] 모델 가중치 로드 중...\n ➔ 경로: {pth_path}")
    
    # 모델 뼈대 생성 및 가중치 삽입
    embedder = ModelDWT(n_blocks=c.INV_BLOCKS).to(device)
    
    try:
        state = torch.load(pth_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"❌ 오류: 가중치 파일을 찾을 수 없습니다! ({pth_path})")
        print(f"checkpoints 폴더 안에 '{pth_filename}' 파일이 있는지 확인해 주세요.")
        return

    if isinstance(state, dict):
        state = state.get("state_dict", state.get("model", state))
    if state and all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    
    missing, _ = embedder.load_state_dict(state, strict=False)
    if missing:
        warnings.warn(f"Missing keys: {missing[:3]}")
        
    # 래퍼로 모델 감싸기
    export_model = INNProtectWrapper(embedder)
    export_model.eval()

    print("[ONNX 추출기] 더미 데이터 생성 중...")
    res = c.NORM_RESOLUTION
    dummy_xa = torch.randn(1, 3, res, res, device=device)
    dummy_xa_obfs = torch.randn(1, 3, res, res, device=device)
    
    skey = generate_key("dummy_password", bs=1, w=res, h=res).to(device)
    dwt = DWT().to(device)
    dummy_skey_dwt = dwt(skey.float())

    print("[ONNX 추출기] ONNX 포맷으로 추출(Export) 중...")
    torch.onnx.export(
        export_model,
        (dummy_xa, dummy_xa_obfs, dummy_skey_dwt), 
        onnx_path,
        export_params=True,
        opset_version=11,               
        do_constant_folding=True,       
        input_names=['xa', 'xa_obfs', 'skey_dwt'],
        output_names=['xa_proc']
    )
    print(f"✅ 성공적으로 변환되었습니다!\n ➔ 저장된 위치: {onnx_path}")

if __name__ == "__main__":
    convert_to_onnx()