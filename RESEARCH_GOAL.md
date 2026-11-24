# 연구 목표: PINN(물리 정보 신경망) 기반 3D/2D 화재 확산 시뮬레이션 구축 및 최적화

## 1. 아키텍처 및 입출력 인터페이스 요구사항
- 명시적 격자(Grid/Voxel) 차분법 대신, 시공간 연속 좌표 $\mathbf{x} = (x, y, z, t)$를 입력받아 물리 상태장을 출력하는 신경망 모델 `FirePINN`을 `src/fire_pinn_sim.py`에 구현한다.
- **입력:** `coords` — 형상 `(N, 4)`의 PyTorch 텐서 (`x, y, z, t`, `requires_grad=True` 활성화 가능해야 함).
- **출력:** `state` — 형상 `(N, 2)`의 PyTorch 텐서 `[T_norm, Y_fuel]`
  - `T_norm`: 정규화된 무차원 온도 ($\theta \ge 0$, 실제 온도 $T = T_{amb} + \Delta T \cdot \theta$)
  - `Y_fuel`: 잔여 연료 질량 분율 ($0 \le Y_{fuel} \le 1$, Sigmoid 등으로 물리적 범위 보장)
- **필수 메서드:**
  1. `forward(self, coords: torch.Tensor) -> torch.Tensor`: 시공간 좌표에 대한 온도 및 연료 분율 예측.
  2. `compute_pde_residual(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`: `torch.autograd.grad`를 사용해 에너지 보존(열전달+연소발열) PDE 잔차 $R_T$와 연료 소모 ODE/PDE 잔차 $R_Y$를 반환.
  3. `train_step(self, collocation_pts, ic_pts, bc_pts, optimizer) -> dict`: PDE 잔차 손실, 초기조건(IC) 손실, 경계조건(BC) 손실을 결합하여 1 스텝 학습을 수행하고 손실 딕셔너리 반환.

## 2. 지배 방정식 (Governing PDEs)
에이전트는 `compute_pde_residual` 내부에 다음 물리 법칙의 잔차를 정확히 구현해야 한다.
1. **에너지 방정식 (이류-확산-반응):**
   $$\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T - \alpha \nabla^2 T - \dot{\omega}_T(T, Y_{fuel}) = 0$$
   - 부력에 의한 상승 기류 $\mathbf{u} = (0, 0, w(T))$ (Boussinesq 근사: $w \propto \max(0, T - T_{amb})$) 반영.
   - 아레니우스(Arrhenius) 반응 속도 항: $\dot{\omega} = A \cdot Y_{fuel} \cdot \exp\left(-\frac{E_a}{R T}\right) \cdot H(T - T_{ign})$
2. **연료 소모 방정식:**
   $$\frac{\partial Y_{fuel}}{\partial t} + \dot{\omega}_Y(T, Y_{fuel}) = 0$$

## 3. 자율 개선 우선순위
1. **미분 그래프 무결성 및 안정성:** 고차 편미분($\nabla^2 T$) 계산 시 `create_graph=True`가 누락되어 역전파가 끊기거나, $\exp(-E_a/RT)$에서 분모가 0이 되어 `NaN`이 발생하는 현상을 원천 차단할 것.
2. **고주파 물리 현상 포착 (Spectral Bias 극복):** 화염 전선(Flame front)의 급격한 온도 구배를 표현할 수 있도록 Fourier Feature Embedding, SIREN, 또는 Adaptive Activation(예: `SiLU`, `Tanh` + 학습 가능한 스케일 인자)을 적용할 것.
3. **손실 균형 (Loss Balancing):** PDE 잔차 손실과 초기/경계 조건 손실 간의 스케일 불균형을 해결하기 위한 가중치 조절(NTK 기반 또는 Adaptive Loss Weighting) 및 인과성 가중치(Causal Training) 기법을 도입할 것.
