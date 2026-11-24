import time
import torch
import pytest
from src.fire_spread_sim import FireSpreadModel

def test_simulation_stability_and_physics():
    model = FireSpreadModel(grid_size=(32, 32, 16))
    # 초기 발화점 설정
    temp = torch.full((1, 1, 32, 32, 16), 293.15)
    temp[:, :, 15:17, 15:17, 0:2] = 800.0
    fuel = torch.ones((1, 1, 32, 32, 16))

    start = time.perf_counter()
    for _ in range(10):
        temp, fuel = model(temp, fuel, dt=0.1)
    elapsed = time.perf_counter() - start

    # 1. 발산(NaN/Inf) 검사
    assert not torch.isnan(temp).any(), "온도장에 NaN이 발생했습니다."
    assert not torch.isinf(temp).any(), "온도장이 발산(Inf)했습니다."
    # 2. 물리적 한계 검사 (연료량은 0~1 사이, 절대영도 이상)
    assert (fuel >= 0.0).all() and (fuel <= 1.0).all(), "연료 밀도 범위가 물리 법칙에 위배됩니다."
    assert (temp >= 250.0).all(), "비정상적인 극저온이 발생했습니다."
    # 3. 속도 기준 검사
    assert elapsed < 2.0, f"연산 속도가 너무 느립니다: {elapsed:.3f}s"
