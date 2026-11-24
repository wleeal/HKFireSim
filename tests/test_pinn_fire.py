import time
import torch
import pytest
from src.fire_pinn_sim import FirePINN

def multi_compartment_sdf(xyz: torch.Tensor) -> torch.Tensor:
    """
    테스트용 다중 구획 실내 공간 SDF (단위 정육면체 [0, 1]^3 내부, 중앙 x=0.5에 칸막이 벽 및 문 개구부 존재)
    - phi > 0: 실내 공기 영역
    - phi = 0: 외벽 및 중앙 칸막이 벽 표면
    - phi < 0: 벽체 내부
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # 1. 외벽 경계 (x, y, z in [0, 1]) 내부 거리
    dist_outer = torch.minimum(
        torch.minimum(torch.minimum(x, 1.0 - x), torch.minimum(y, 1.0 - y)),
        torch.minimum(z, 1.0 - z)
    )

    # 2. 중앙 칸막이 벽 (x = 0.5, 두께 0.04 -> |x - 0.5| <= 0.02)
    # 단, y in [0.35, 0.65], z in [0.0, 0.7] 구간에는 문(Doorway) 개구부가 뚫려 있음
    wall_x_dist = torch.abs(x - 0.5) - 0.02
    door_y_dist = torch.maximum(0.35 - y, y - 0.65)
    door_z_dist = z - 0.7
    door_opening = torch.maximum(door_y_dist, door_z_dist)

    # 칸막이 벽체까지의 부호 거리 (벽 내부일 때 음수, 공기 영역일 때 양수)
    dist_partition = torch.maximum(wall_x_dist, -door_opening)

    # 최종 실내 공기 영역 SDF (외벽 안쪽이면서 칸막이 벽 바깥쪽)
    sdf = torch.minimum(dist_outer, dist_partition)
    return sdf.unsqueeze(-1)

@pytest.fixture
def model():
    return FirePINN(hidden_dim=64, num_layers=4)

def test_forward_with_sdf_bounds(model):
    """SDF 입력 포함 시 입출력 형상 및 물리적 범위 검증"""
    coords = torch.rand((128, 4), requires_grad=True)
    sdf_val = multi_compartment_sdf(coords[:, :3])
    out = model(coords, sdf_val=sdf_val)

    assert out.shape == (128, 2), f"출력 형상 오류: expected (128, 2), got {out.shape}"
    assert not torch.isnan(out).any(), "Forward 출력에 NaN이 포함되었습니다."

    t_norm, y_fuel = out[:, 0], out[:, 1]
    assert (t_norm >= 0.0).all(), "무차원 온도는 음수가 될 수 없습니다."
    assert (y_fuel >= 0.0).all() and (y_fuel <= 1.0).all(), "연료 질량 분율은 [0, 1] 범위여야 합니다."

def test_sdf_pde_and_wall_bc_autograd(model):
    """SDF 기반 에너지·연료 PDE 잔차 및 벽면 법선 경계조건(res_bc) 자동 미분 그래프 검증"""
    coords = torch.rand((64, 4), requires_grad=True)
    res_T, res_Y, res_bc = model.compute_pde_residual(coords, sdf_fn=multi_compartment_sdf)

    for name, res in [("res_T", res_T), ("res_Y", res_Y), ("res_bc", res_bc)]:
        assert res.shape in [(64,), (64, 1)], f"{name} 형상 오류: {res.shape}"
        assert not torch.isnan(res).any(), f"{name}에 NaN이 발생했습니다."

    loss = (res_T ** 2).mean() + (res_Y ** 2).mean() + (res_bc ** 2).mean()
    loss.backward()

    has_grad = any(p.grad is not None and torch.abs(p.grad).sum() > 0 for p in model.parameters())
    assert has_grad, "PDE 및 SDF 경계조건 손실로부터 가중치로 그래디언트가 전파되지 않습니다."

def test_multi_compartment_training_convergence(model):
    """다중 구획 환경에서 콜로케이션, 초기조건, 벽면 샘플 학습 수렴 및 속도 검증"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    collocation_pts = torch.rand((256, 4))
    ic_pts = torch.rand((64, 4))
    ic_pts[:, 3] = 0.0

    # 중앙 칸막이 벽면(x = 0.48 또는 0.52) 및 외벽 근처 샘플 생성
    wall_pts = torch.rand((64, 4))
    wall_pts[:32, 0] = 0.48
    wall_pts[32:, 0] = 0.0

    initial_loss = None
    start_time = time.perf_counter()

    for step in range(15):
        loss_dict = model.train_step(
            collocation_pts=collocation_pts,
            ic_pts=ic_pts,
            wall_pts=wall_pts,
            sdf_fn=multi_compartment_sdf,
            optimizer=optimizer
        )
        assert "total_loss" in loss_dict, "train_step 반환 딕셔너리에 'total_loss' 키가 없습니다."
        total_loss = loss_dict["total_loss"]
        assert not torch.isnan(torch.tensor(total_loss)), f"Step {step}에서 손실이 NaN으로 발산했습니다."
        if step == 0:
            initial_loss = total_loss

    elapsed = time.perf_counter() - start_time
    final_loss = loss_dict["total_loss"]

    assert final_loss < initial_loss, f"학습이 진행되어도 손실이 감소하지 않았습니다 ({initial_loss:.4f} -> {final_loss:.4f})."
    assert elapsed < 6.0, f"15 스텝 SDF-PINN 학습 시간이 너무 깁니다: {elapsed:.2f}s"
