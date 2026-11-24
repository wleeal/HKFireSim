import torch
import torch.nn as nn


class FourierFeatureEmbedding(nn.Module):
    """화염 전선 및 개구부의 급격한 온도 구배 포착을 위한 Gaussian Fourier Feature 매핑"""
    def __init__(self, in_features: int, mapping_size: int = 32, scale: float = 2.0):
        super().__init__()
        self.register_buffer("B", torch.randn((in_features, mapping_size)) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * torch.pi * (x @ self.B)
        return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)


class FirePINN(nn.Module):
    """
    SDF(Signed Distance Function) 기반 다중 구획 실내 화재 확산 PINN 모델
    - 입력: coords (N, 4) -> [x, y, z, t], 선택적 sdf_val (N, 1)
    - 출력: state  (N, 2) -> [T_norm (>= 0), Y_fuel (0 ~ 1)]
    """
    def __init__(self, hidden_dim: int = 64, num_layers: int = 4, fourier_dim: int = 16):
        super().__init__()
        self.fourier = FourierFeatureEmbedding(in_features=4, mapping_size=fourier_dim, scale=1.5)
        # 입력 차원: 원본 coords(4) + sin/cos(2 * fourier_dim) + sdf_val(1)
        in_dim = 4 + 2 * fourier_dim + 1

        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, 2))

        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus(beta=5.0)
        self.sigmoid = nn.Sigmoid()

        # 무차원 물리 파라미터 (지배 방정식 스케일)
        self.alpha = 0.02        # 열확산 계수 (Thermal diffusivity)
        self.buoyancy_c = 0.5    # 부력 상승 속도 계수
        self.damkohler = 1.5     # 연소 반응 속도 계수 (Damkohler number)
        self.act_energy = 1.0    # 무차원 활성화 에너지 (E_a / R)
        self.heat_release = 2.0  # 연소 발열 계수
        self.h_wall = 0.05       # 벽면 대류 열손실 계수 (0이면 완전 단열)

    def forward(self, coords: torch.Tensor, sdf_val: torch.Tensor | None = None) -> torch.Tensor:
        if sdf_val is None:
            sdf_val = torch.ones((coords.shape[0], 1), device=coords.device, dtype=coords.dtype)
        elif sdf_val.ndim == 1:
            sdf_val = sdf_val.unsqueeze(-1)

        emb = self.fourier(coords)
        inp = torch.cat([emb, sdf_val], dim=-1)
        raw_out = self.net(inp)

        # 물리적 출력 범위 강제: T_norm >= 0, Y_fuel in [0, 1]
        t_norm = self.softplus(raw_out[:, 0:1])
        y_fuel = self.sigmoid(raw_out[:, 1:2])

        # 벽면 내부(sdf < 0)에서는 연료 존재 억제
        fluid_mask = torch.sigmoid(20.0 * sdf_val)
        y_fuel = y_fuel * fluid_mask

        return torch.cat([t_norm, y_fuel], dim=-1)

    def compute_pde_residual(
        self, coords: torch.Tensor, sdf_fn: callable
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        자동 미분을 이용한 에너지 보존 PDE 잔차(res_T), 연료 소모 잔차(res_Y),
        및 SDF 벽면 법선 열경계조건 잔차(res_bc) 계산
        """
        if not coords.requires_grad:
            coords = coords.clone().detach().requires_grad_(True)

        xyz = coords[:, :3]
        sdf_val = sdf_fn(xyz)
        if sdf_val.ndim == 1:
            sdf_val = sdf_val.unsqueeze(-1)

        state = self.forward(coords, sdf_val=sdf_val)
        t_norm = state[:, 0:1]
        y_fuel = state[:, 1:2]

        # 1차 편미분 (dT/dx, dT/dy, dT/dz, dT/dt) 및 (dY/dt)
        grad_T = torch.autograd.grad(
            t_norm, coords, grad_outputs=torch.ones_like(t_norm), create_graph=True
        )[0]
        dT_dx, dT_dy, dT_dz, dT_dt = grad_T[:, 0:1], grad_T[:, 1:2], grad_T[:, 2:3], grad_T[:, 3:4]

        grad_Y = torch.autograd.grad(
            y_fuel, coords, grad_outputs=torch.ones_like(y_fuel), create_graph=True
        )[0]
        dY_dt = grad_Y[:, 3:4]

        # 2차 편미분 (라플라시안: d2T/dx2 + d2T/dy2 + d2T/dz2)
        d2T_dx2 = torch.autograd.grad(
            dT_dx, coords, grad_outputs=torch.ones_like(dT_dx), create_graph=True
        )[0][:, 0:1]
        d2T_dy2 = torch.autograd.grad(
            dT_dy, coords, grad_outputs=torch.ones_like(dT_dy), create_graph=True
        )[0][:, 1:2]
        d2T_dz2 = torch.autograd.grad(
            dT_dz, coords, grad_outputs=torch.ones_like(dT_dz), create_graph=True
        )[0][:, 2:3]
        laplacian_T = d2T_dx2 + d2T_dy2 + d2T_dz2

        # SDF 기반 유동 차폐 마스킹 (벽면 근처에서 속도 감쇠)
        flow_mask = torch.sigmoid(25.0 * sdf_val)
        # Boussinesq 부력에 의한 수직 상승 속도 w(T)
        w_vel = flow_mask * self.buoyancy_c * torch.clamp(t_norm, min=0.0, max=5.0)
        advection = w_vel * dT_dz

        # 아레니우스 연소 반응 속도 (0으로 나누기 및 발산 방지 clamp 적용)
        safe_temp = torch.clamp(t_norm, min=1e-3, max=10.0)
        ignition_gate = torch.sigmoid(10.0 * (t_norm - 0.15))
        omega = (
            flow_mask
            * self.damkohler
            * y_fuel
            * torch.exp(-self.act_energy / safe_temp)
            * ignition_gate
        )

        # 내부 공기 영역(sdf > 0) 지배 방정식 잔차
        res_T = flow_mask * (dT_dt + advection - self.alpha * laplacian_T - self.heat_release * omega)
        res_Y = flow_mask * (dY_dt + omega)

        # 벽면 법선 벡터 n = grad(sdf) / ||grad(sdf)|| 계산 및 열경계조건 잔차
        grad_sdf = torch.autograd.grad(
            sdf_val, coords, grad_outputs=torch.ones_like(sdf_val), create_graph=True
        )[0][:, :3]
        norm_sdf = torch.norm(grad_sdf, dim=-1, keepdim=True).clamp(min=1e-6)
        normal_vec = grad_sdf / norm_sdf

        grad_T_spatial = grad_T[:, :3]
        dT_dn = (grad_T_spatial * normal_vec).sum(dim=-1, keepdim=True)
        res_bc = dT_dn + self.h_wall * t_norm

        return res_T.squeeze(-1), res_Y.squeeze(-1), res_bc.squeeze(-1)

    def _initial_condition_target(self, ic_pts: torch.Tensor, sdf_val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """t = 0 초기 상태 정의: 방 1 내부(x=0.25, y=0.5, z=0.15) 국소 발화원"""
        dist_sq = (
            (ic_pts[:, 0:1] - 0.25) ** 2
            + (ic_pts[:, 1:2] - 0.50) ** 2
            + (ic_pts[:, 2:3] - 0.15) ** 2
        )
        t0 = 1.2 * torch.exp(-dist_sq / 0.015)
        # 공기 영역(sdf > 0)에만 가연성 연료 배치
        y0 = torch.where(sdf_val > 0.0, torch.ones_like(sdf_val), torch.zeros_like(sdf_val))
        return t0, y0

    def train_step(
        self,
        collocation_pts: torch.Tensor,
        ic_pts: torch.Tensor,
        wall_pts: torch.Tensor,
        sdf_fn: callable,
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        optimizer.zero_grad()

        # 1. 내부 시공간 콜로케이션 포인트 PDE 잔차 손실
        res_T, res_Y, _ = self.compute_pde_residual(collocation_pts, sdf_fn)
        loss_pde = (res_T ** 2).mean() + (res_Y ** 2).mean()

        # 2. 초기 조건 (t = 0) 손실
        sdf_ic = sdf_fn(ic_pts[:, :3])
        if sdf_ic.ndim == 1:
            sdf_ic = sdf_ic.unsqueeze(-1)
        pred_ic = self.forward(ic_pts, sdf_val=sdf_ic)
        target_t0, target_y0 = self._initial_condition_target(ic_pts, sdf_ic)
        loss_ic = ((pred_ic[:, 0:1] - target_t0) ** 2).mean() + ((pred_ic[:, 1:2] - target_y0) ** 2).mean()

        # 3. 벽면 경계 조건 (phi ~ 0) 손실
        _, _, res_bc_wall = self.compute_pde_residual(wall_pts, sdf_fn)
        loss_bc = (res_bc_wall ** 2).mean()

        total_loss = loss_pde + 5.0 * loss_ic + 2.0 * loss_bc
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        return {
            "total_loss": float(total_loss.item()),
            "loss_pde": float(loss_pde.item()),
            "loss_ic": float(loss_ic.item()),
            "loss_bc": float(loss_bc.item()),
        }
