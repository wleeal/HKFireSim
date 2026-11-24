import time
import torch
import pytest
from src.fire_pinn_sim import FirePINN

@pytest.fixture
def model():
    return FirePINN(hidden_dim=64, num_layers=4)

def test_forward_shape_and_physical_bounds(model):
    """입출력 텐서 차원 및 물리적 출력 범위(온도 >= 0, 연료 0~1) 검증"""
    coords = torch.rand((128, 4), requires_grad=True)
    out = model(coords)

    assert out.shape == (128, 2), f"출력 형상 오류: expected (128, 2), got {out.shape}"
    assert not torch.isnan(out).any(), "Forward 출력에 NaN이 포함되었습니다."

    t_norm, y_fuel = out[:, 0], out[:, 1]
    assert (t_norm >= 0.0).all(), "무차원 온도는 음수가 될 수 없습니다."
    assert (y_fuel >= 0.0).all() and (y_fuel <= 1.0).all(), "연료 질량 분율은 [0, 1] 범위여야 합니다."

def test_autograd_pde_residual_graph(model):
    """자동 미분을 통한 2차 편미분(라플라시안) 계산 및 역전파 그래프 연결 검증"""
    coords = torch.rand((64, 4), requires_grad=True)
    res_T, res_Y = model.compute_pde_residual(coords)

    assert res_T.shape == (64,) or res_T.shape == (64, 1), f"res_T 형상 오류: {res_T.shape}"
    assert res_Y.shape == (64,) or res_Y.shape == (64, 1), f"res_Y 형상 오류: {res_Y.shape}"
    assert not torch.isnan(res_T).any(), "에너지 PDE 잔차에 NaN이 발생했습니다."
    assert not torch.isnan(res_Y).any(), "연료 PDE 잔차에 NaN이 발생했습니다."

    # 잔차의 제곱합으로부터 모델 파라미터까지 그래디언트가 정상 전파되는지 확인 (2차 미분 그래프 검증)
    loss = (res_T ** 2).mean() + (res_Y ** 2).mean()
    loss.backward()

    has_grad = any(p.grad is not None and torch.abs(p.grad).sum() > 0 for p in model.parameters())
    assert has_grad, "PDE 잔차 손실로부터 신경망 가중치로 그래디언트가 전파되지 않습니다 (create_graph=True 누락 의심)."

def test_training_convergence_and_speed(model):
    """실제 학습 스텝 수행 시 손실 감소 여부 및 연산 속도 벤치마크"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    collocation_pts = torch.rand((256, 4))
    # 초기 조건 (t = 0)
    ic_pts = torch.rand((64, 4))
    ic_pts[:, 3] = 0.0
    # 경계 조건 (x, y, z 경계)
    bc_pts = torch.rand((64, 4))
    bc_pts[:, 0] = 0.0

    initial_loss = None
    start_time = time.perf_counter()

    for step in range(15):
        loss_dict = model.train_step(collocation_pts, ic_pts, bc_pts, optimizer)
        assert "total_loss" in loss_dict, "train_step 반환 딕셔너리에 'total_loss' 키가 없습니다."
        total_loss = loss_dict["total_loss"]
        assert not torch.isnan(torch.tensor(total_loss)), f"Step {step}에서 손실이 NaN으로 발산했습니다."
        if step == 0:
            initial_loss = total_loss

    elapsed = time.perf_counter() - start_time
    final_loss = loss_dict["total_loss"]

    assert final_loss < initial_loss, f"학습이 진행되어도 손실이 감소하지 않았습니다 ({initial_loss:.4f} -> {final_loss:.4f})."
    assert elapsed < 5.0, f"15 스텝 PINN 학습 시간이 너무 깁니다: {elapsed:.2f}s"
