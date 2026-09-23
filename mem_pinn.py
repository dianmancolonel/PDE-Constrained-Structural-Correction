import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
import time


@dataclass
class MemristorPINNConfig:
    """
    Memristor PINN config class - ODE mismatch experiment
    """
    L: float = 1.0
    T: float = 1.0

    u_hidden: List[int] = field(default_factory=lambda: [128, 128, 128, 128])
    w_hidden: List[int] = field(default_factory=lambda: [64, 64, 64, 64])
    use_fourier: bool = True
    num_freq: int = 8

    # WNet-specific Fourier feature config
    w_use_fourier: bool = True
    w_num_freq: int = 10
    w_output_mode: str = 'clamp'  # 'clamp', 'tanh', 'sigmoid'

    alpha_min: float = 0.1
    alpha_max: float = 1.0
    w_init: float = 0.5
    memristor_mu: float = 0.8
    memristor_V: float = 2.0
    memristor_model: str = 'biolek'
    memristor_p: int = 4

    # Voltage waveform parameters
    voltage_waveform: str = 'sinedown'  # 'sine', 'square', 'sinedown'
    voltage_freq: float = 2.0
    voltage_decay: float = 1.0

    # ODE mismatch experiment parameters
    ode_mismatch_enabled: bool = True
    ode_mismatch_delta_mu: float = 0.3
    ode_mismatch_type: str = 'additive'  # 'additive' or 'multiplicative'

    # w observation sampling strategy
    n_w_obs: int = 2
    w_obs_sampling: str = 'uniform'  # 'uniform', 'random', 'endpointrandom'

    n_u_obs: int = 300
    n_ic: int = 300
    n_bc: int = 300
    n_pde: int = 10000

    epochs: int = 50000
    lr: float = 1e-4
    lr_w: float = 1e-3
    use_amp: bool = True

    # Loss term weights
    w_pde = 1.0
    w_ic = 10.0
    w_bc = 10.0
    w_u_obs = 1.0
    w_physics = 0.0002
    w_w_obs = 1.0

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print_every: int = 200
    seed: int = 3

    def get_physics_params(self, use_mismatch: bool = False) -> Dict:
        """
        Get physics parameters, with optional ODE mismatch
        """
        params = {
            'mu': self.memristor_mu,
            'V': self.memristor_V,
            'window_type': self.memristor_model,
            'p': self.memristor_p,
            'voltage_waveform': self.voltage_waveform,
            'voltage_freq': self.voltage_freq,
            'voltage_decay': self.voltage_decay,
        }

        # Apply ODE mismatch: perturb true μ
        if use_mismatch and self.ode_mismatch_enabled:
            if self.ode_mismatch_type == 'additive':
                params['mu'] += self.ode_mismatch_delta_mu
            elif self.ode_mismatch_type == 'multiplicative':
                params['mu'] *= 1.0 + self.ode_mismatch_delta_mu
            params['is_mismatched'] = True
        else:
            params['is_mismatched'] = False

        return params


class MemristorPhysics:
    @staticmethod
    def window_function(w, window_type='biolek', p=10):
        """
        Window function: prevents state variable w from exceeding physical bounds [0,1]
        """
        w = torch.clamp(w, 0.001, 0.999)
        if window_type == 'biolek':
            return 1 - (2 * w - 1) ** (2 * p)
        elif window_type == 'joglekar':
            return 1 - ((w - 0.5) * 2) ** (2 * p)
        elif window_type == 'linear':
            return 4 * w * (1 - w)
        else:
            return torch.ones_like(w)

    @staticmethod
    def voltage_function_np(t, V0, waveform='sine', freq=1.0, decay=2.0):
        """Voltage function - NumPy version"""
        if waveform == 'sine':
            return V0 * np.sin(2 * np.pi * freq * t)
        elif waveform == 'square':
            return V0 * np.sign(np.sin(2 * np.pi * freq * t))
        elif waveform == 'sinedown':
            # Amplitude-decaying sine wave: V(t) = V₀·exp(-decay·t)·sin(2πft)
            return V0 * np.exp(-decay * t) * np.sin(2 * np.pi * freq * t)
        else:
            return np.ones_like(t) * V0

    @staticmethod
    def voltage_function_torch(t, V0, waveform='sine', freq=1.0, decay=2.0):
        """Voltage function - PyTorch version"""
        if waveform == 'sine':
            return V0 * torch.sin(2 * np.pi * freq * t)
        elif waveform == 'square':
            return V0 * torch.sign(torch.sin(2 * np.pi * freq * t))
        elif waveform == 'sinedown':
            return V0 * torch.exp(-decay * t) * torch.sin(2 * np.pi * freq * t)
        else:
            return torch.ones_like(t) * V0

    @staticmethod
    def generate_ground_truth(config, nt=2000):
        """
        Generate w(t) reference solution using true ODE parameters
        """
        from scipy.integrate import odeint

        t_np = np.linspace(0, config.T, nt)
        params = config.get_physics_params(use_mismatch=False)
        mu_true = params['mu']
        V0 = params['V']
        window_type = params['window_type']
        p = params['p']
        waveform = params['voltage_waveform']
        freq = params['voltage_freq']
        decay = params['voltage_decay']

        def dwdt(w, t):
            """True ODE"""
            w = np.clip(w, 0.001, 0.999)
            w_t = torch.tensor(w, dtype=torch.float32).reshape(-1)
            f_w = MemristorPhysics.window_function(w_t, window_type, p).numpy()
            V_t = MemristorPhysics.voltage_function_np(np.array([t]), V0, waveform, freq, decay)[0]
            return mu_true * V_t * f_w

        w_np = odeint(dwdt, config.w_init, t_np).flatten()
        w_np = np.clip(w_np, 0.0, 1.0)

        t_tensor = torch.tensor(t_np, dtype=torch.float32, device=config.device).reshape(-1, 1)
        w_tensor = torch.tensor(w_np, dtype=torch.float32, device=config.device).reshape(-1, 1)

        return t_tensor, w_tensor

    @staticmethod
    def generate_ground_truth_mismatch(config, nt=2000):
        """
        Generate w(t) using mismatched ODE parameters for comparison
        """
        from scipy.integrate import odeint

        t_np = np.linspace(0, config.T, nt)
        params = config.get_physics_params(use_mismatch=True)
        mu_mismatch = params['mu']
        V0 = params['V']
        window_type = params['window_type']
        p = params['p']
        waveform = params['voltage_waveform']
        freq = params['voltage_freq']
        decay = params['voltage_decay']

        def dwdt(w, t):
            """Mismatched ODE"""
            w = np.clip(w, 0.001, 0.999)
            w_t = torch.tensor(w, dtype=torch.float32).reshape(-1)
            f_w = MemristorPhysics.window_function(w_t, window_type, p).numpy()
            V_t = MemristorPhysics.voltage_function_np(np.array([t]), V0, waveform, freq, decay)[0]
            return mu_mismatch * V_t * f_w

        w_np = odeint(dwdt, config.w_init, t_np).flatten()
        w_np = np.clip(w_np, 0.0, 1.0)

        t_tensor = torch.tensor(t_np, dtype=torch.float32, device=config.device).reshape(-1, 1)
        w_tensor = torch.tensor(w_np, dtype=torch.float32, device=config.device).reshape(-1, 1)

        return t_tensor, w_tensor

    @staticmethod
    def alpha_mapping(w, alpha_min=0.1, alpha_max=2.0):
        """
        Map state variable w(t) ∈ [0,1] to diffusion coefficient α(w)
        """
        return alpha_min + (alpha_max - alpha_min) * w ** 2

class FourierFeatures(nn.Module):
    """
    Fourier feature encoding: maps input to high-dim frequency space to improve fitting capacity
    """

    def __init__(self, input_dim, num_freq=8):
        super().__init__()
        B = torch.randn(input_dim, num_freq) * 2 * np.pi
        self.register_buffer('B', B)

    def forward(self, x):
        x_proj = x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class UNet(nn.Module):
    """Neural network for temperature field u(x,t)"""

    def __init__(self, config):
        super().__init__()
        self.use_fourier = config.use_fourier

        if self.use_fourier:
            self.fourier = FourierFeatures(2, config.num_freq)
            input_dim = 2 * config.num_freq
        else:
            input_dim = 2

        layers = []
        prev_dim = input_dim
        for h_dim in config.u_hidden:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.Tanh()])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        if self.use_fourier:
            xt = self.fourier(xt)
        return self.net(xt)


class WNet(nn.Module):
    """
    Neural network for memristor state variable w(t)

    Design highlights:
      Fourier features improve fitting of temporal dynamics
      No activation on output layer to avoid gradient vanishing
      Three output mapping modes: clamp (recommended), tanh, sigmoid
    """

    def __init__(self, config):
        super().__init__()

        self.use_fourier = config.w_use_fourier

        if self.use_fourier:
            self.num_freq = config.w_num_freq
            # Random Fourier frequency matrix, fixed (not trained)
            B = torch.randn(1, self.num_freq) * 2 * np.pi
            self.register_buffer('B', B)
            input_dim = 2 * self.num_freq
        else:
            input_dim = 1

        layers = []
        prev_dim = input_dim
        for h_dim in config.w_hidden:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.Tanh()])
            prev_dim = h_dim
        # Output layer has no activation
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)
        self.output_mode = config.w_output_mode
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def fourier_encode(self, t):
        """
        Fourier feature encoding: [N,1] -> [N, 2*num_freq]
        """
        if not self.use_fourier:
            return t
        B = self.B.to(t.device)
        t_proj = 2 * np.pi * t @ B
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)

    def forward(self, t):
        """
        Forward pass, outputs w(t) ∈ [0,1]
        """
        t_encoded = self.fourier_encode(t)
        w_raw = self.net(t_encoded)

        # Map raw network output to [0,1] range
        if self.output_mode == 'clamp':
            w = torch.clamp(w_raw, 0.0, 1.0)
        elif self.output_mode == 'tanh':
            w = 0.5 * (torch.tanh(w_raw) + 1.0)
        elif self.output_mode == 'sigmoid':
            w = torch.sigmoid(w_raw)
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")

        return w


class MemristorPINN(nn.Module):
    """
    Memristor PINN model trained with mismatched ODE prior
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.u_net = UNet(config)
        self.w_net = WNet(config)
        self.to(config.device)

    def predict_w(self, t):
        return self.w_net(t)

    def predict_u(self, x, t):
        return self.u_net(x, t)

    def compute_alpha(self, w):
        return MemristorPhysics.alpha_mapping(
            w, self.config.alpha_min, self.config.alpha_max
        )

    def compute_pde_residual(self, x, t):
        """
        Compute PDE residual: u_t - ∂_x(α(w)·u_x) = 0
        """
        x.requires_grad_(True)
        t.requires_grad_(True)

        u = self.predict_u(x, t)
        w = self.predict_w(t)
        alpha = self.compute_alpha(w)

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        flux = alpha * u_x
        flux_x = torch.autograd.grad(flux, x, grad_outputs=torch.ones_like(flux), create_graph=True)[0]

        residual = u_t - flux_x
        return residual

    def compute_ode_residual(self, t):
        """
        Compute ODE residual using mismatched μ (intentionally deviated from true value)

        Note: mismatched prior is intentional, to test if PDE constraints can correct it
        """
        t.requires_grad_(True)
        w = self.predict_w(t)

        w_t = torch.autograd.grad(w, t, grad_outputs=torch.ones_like(w), create_graph=True)[0]

        params = self.config.get_physics_params(use_mismatch=True)
        mu = params['mu']
        V0 = params['V']
        window_type = params['window_type']
        p = params['p']
        waveform = params['voltage_waveform']
        freq = params['voltage_freq']
        decay = params['voltage_decay']

        V_t = MemristorPhysics.voltage_function_torch(t, V0, waveform, freq, decay)
        f_w = MemristorPhysics.window_function(w, window_type, p)

        dwdt_physics = mu * V_t * f_w
        residual = w_t - dwdt_physics

        return residual


def generate_data(config: MemristorPINNConfig) -> Dict[str, torch.Tensor]:
    """
    Generate training data and ground truth reference solutions
    """
    device = config.device

    # Ground truth w(t) from true ODE
    print("\nGenerating ground truth data...")
    t_gt, w_gt = MemristorPhysics.generate_ground_truth(config, nt=2000)

    # Mismatched ODE solution for comparison
    if config.ode_mismatch_enabled:
        print("Generating mismatched ODE solution...")
        t_mismatch, w_mismatch = MemristorPhysics.generate_ground_truth_mismatch(config, nt=2000)
    else:
        t_mismatch, w_mismatch = None, None

    # Sample w observation points
    if config.n_w_obs > 0:
        if config.w_obs_sampling == 'uniform':
            w_obs_indices = torch.linspace(0, len(t_gt) - 1, config.n_w_obs).long()
        elif config.w_obs_sampling == 'random':
            w_obs_indices = torch.randperm(len(t_gt))[:config.n_w_obs].sort()[0]
        elif config.w_obs_sampling == 'endpointrandom':
            # Fix endpoints + random interior sampling
            if config.n_w_obs == 1:
                w_obs_indices = torch.tensor([0], dtype=torch.long)
            elif config.n_w_obs == 2:
                w_obs_indices = torch.tensor([0, len(t_gt) - 1], dtype=torch.long)
            else:
                endpoint_indices = torch.tensor([0, len(t_gt) - 1], dtype=torch.long)
                n_middle = config.n_w_obs - 2
                middle_pool = torch.arange(1, len(t_gt) - 1)
                middle_indices = middle_pool[torch.randperm(len(middle_pool))[:n_middle]]
                w_obs_indices = torch.cat([endpoint_indices, middle_indices]).sort()[0]
        else:
            raise ValueError(f"Unknown w_obs_sampling: {config.w_obs_sampling}")

        t_w_obs = t_gt[w_obs_indices]
        w_obs = w_gt[w_obs_indices]
    else:
        t_w_obs = torch.empty(0, 1, device=device)
        w_obs = torch.empty(0, 1, device=device)

    # Solve PDE with true w(t) to generate u(x,t) ground truth field
    print("Solving PDE to generate u(x,t) ground truth...")
    nx, nt = 100, 100
    x_np = np.linspace(0, config.L, nx)
    t_np = np.linspace(0, config.T, nt)

    from scipy.integrate import solve_ivp

    def pde_system(t_val, u_flat):
        u = u_flat.reshape(nx)
        idx = np.argmin(np.abs(t_gt.cpu().numpy().flatten() - t_val))
        w_val = w_gt[idx].item()
        alpha = MemristorPhysics.alpha_mapping(
            torch.tensor([w_val]), config.alpha_min, config.alpha_max
        ).item()

        dx = x_np[1] - x_np[0]
        u_xx = np.zeros(nx)
        u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2
        u_xx[0] = 0
        u_xx[-1] = 0

        dudt = alpha * u_xx
        return dudt.flatten()

    u0 = np.sin(np.pi * x_np / config.L)
    sol = solve_ivp(pde_system, [0, config.T], u0, t_eval=t_np, method='RK45')
    u_gt_np = sol.y.T

    X_np, T_np = np.meshgrid(x_np, t_np)
    X_gt = torch.tensor(X_np.flatten(), dtype=torch.float32, device=device).reshape(-1, 1)
    T_gt = torch.tensor(T_np.flatten(), dtype=torch.float32, device=device).reshape(-1, 1)
    U_gt = torch.tensor(u_gt_np.flatten(), dtype=torch.float32, device=device).reshape(-1, 1)

    # Sample u observation points
    if config.n_u_obs > 0:
        obs_indices = torch.randperm(len(X_gt))[:config.n_u_obs]
        x_u_obs = X_gt[obs_indices]
        t_u_obs = T_gt[obs_indices]
        u_obs = U_gt[obs_indices]
    else:
        x_u_obs = torch.empty(0, 1, device=device)
        t_u_obs = torch.empty(0, 1, device=device)
        u_obs = torch.empty(0, 1, device=device)

    # PDE collocation points (random sampling)
    x_pde = torch.rand(config.n_pde, 1, device=device) * config.L
    t_pde = torch.rand(config.n_pde, 1, device=device) * config.T

    # Initial condition points
    x_ic = torch.rand(config.n_ic, 1, device=device) * config.L
    t_ic = torch.zeros(config.n_ic, 1, device=device)
    u_ic = torch.sin(np.pi * x_ic / config.L)

    # Boundary condition points (x=0 and x=L)
    t_bc = torch.rand(config.n_bc, 1, device=device) * config.T
    x_bc_0 = torch.zeros(config.n_bc // 2, 1, device=device)
    x_bc_L = torch.ones(config.n_bc // 2, 1, device=device) * config.L
    x_bc = torch.cat([x_bc_0, x_bc_L], dim=0)
    t_bc = torch.cat([t_bc[:config.n_bc // 2], t_bc[config.n_bc // 2:]], dim=0)

    data = {
        't_gt': t_gt, 'w_gt': w_gt,
        'X_gt': X_gt, 'T_gt': T_gt, 'U_gt': U_gt,
        't_mismatch': t_mismatch, 'w_mismatch': w_mismatch,
        't_w_obs': t_w_obs, 'w_obs': w_obs,
        'x_u_obs': x_u_obs, 't_u_obs': t_u_obs, 'u_obs': u_obs,
        'x_pde': x_pde, 't_pde': t_pde,
        'x_ic': x_ic, 't_ic': t_ic, 'u_ic': u_ic,
        'x_bc': x_bc, 't_bc': t_bc,
    }

    print(f"Data generation complete!")
    print(f"  w_gt shape: {w_gt.shape}")
    print(f"  w_obs shape: {w_obs.shape}")
    print(f"  u_obs shape: {u_obs.shape}")

    return data


def train(model: MemristorPINN, config: MemristorPINNConfig, data: Dict) -> Dict:
    """
    Train the PINN model

    UNet and WNet use separate learning rates for balanced convergence
    """
    optimizer = torch.optim.Adam([
        {'params': model.u_net.parameters(), 'lr': config.lr},
        {'params': model.w_net.parameters(), 'lr': config.lr_w}
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=600,
        min_lr=2e-6
    )
    scaler = GradScaler() if config.use_amp else None

    history = {
        'u_obs': [], 'pde': [], 'ic': [], 'bc': [], 'physics': [], 'w_obs': []
    }

    print("\n" + "=" * 80)
    print("Training PINN")
    print("=" * 80)

    start_time = time.time()

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()

        with autocast(enabled=config.use_amp):
            # u observation loss
            if config.w_u_obs > 0 and data['u_obs'].numel() > 0:
                u_pred_obs = model.predict_u(data['x_u_obs'], data['t_u_obs'])
                loss_u_obs = torch.mean((u_pred_obs - data['u_obs']) ** 2)
            else:
                loss_u_obs = torch.tensor(0.0, device=config.device)

            # PDE residual loss
            if config.w_pde > 0:
                pde_res = model.compute_pde_residual(data['x_pde'], data['t_pde'])
                loss_pde = torch.mean(pde_res ** 2)
            else:
                loss_pde = torch.tensor(0.0, device=config.device)

            # Initial condition loss
            if config.w_ic > 0:
                u_pred_ic = model.predict_u(data['x_ic'], data['t_ic'])
                loss_ic = torch.mean((u_pred_ic - data['u_ic']) ** 2)
            else:
                loss_ic = torch.tensor(0.0, device=config.device)

            if config.w_bc > 0:
                u_pred_bc = model.predict_u(data['x_bc'], data['t_bc'])
                loss_bc = torch.mean(u_pred_bc ** 2)
            else:
                loss_bc = torch.tensor(0.0, device=config.device)

            if config.w_physics > 0:
                ode_res = model.compute_ode_residual(data['t_pde'])
                loss_physics = torch.mean(ode_res ** 2)
            else:
                loss_physics = torch.tensor(0.0, device=config.device)

            # w observation loss
            if config.w_w_obs > 0 and data['w_obs'].numel() > 0:
                w_pred_obs = model.predict_w(data['t_w_obs'])
                loss_w_obs = torch.mean((w_pred_obs - data['w_obs']) ** 2)
            else:
                loss_w_obs = torch.tensor(0.0, device=config.device)

            # Weighted total loss
            loss = (config.w_u_obs * loss_u_obs +
                    config.w_pde * loss_pde +
                    config.w_ic * loss_ic +
                    config.w_bc * loss_bc +
                    config.w_physics * loss_physics +
                    config.w_w_obs * loss_w_obs)

        if config.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step(loss.item())

        history['u_obs'].append(loss_u_obs.item())
        history['pde'].append(loss_pde.item())
        history['ic'].append(loss_ic.item())
        history['bc'].append(loss_bc.item())
        history['physics'].append(loss_physics.item())
        history['w_obs'].append(loss_w_obs.item())

        if (epoch + 1) % config.print_every == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch + 1}/{config.epochs} | "
                  f"Loss: {loss.item():.6e} | "
                  f"u_obs: {loss_u_obs.item():.4e} | "
                  f"PDE: {loss_pde.item():.4e} | "
                  f"Physics: {loss_physics.item():.4e} | "
                  f"w_obs: {loss_w_obs.item():.4e} | "
                  f"Time: {elapsed:.1f}s")

    print(f"\nTraining complete! Total time: {time.time() - start_time:.1f}s")

    return history


def visualize(model, data, config, history):
    """Visualize results"""
    model.eval()

    params_true = config.get_physics_params(use_mismatch=False)
    params_train = config.get_physics_params(use_mismatch=True)

    with torch.no_grad():
        t_np = data['t_gt'].cpu().numpy().flatten()
        w_true_np = data['w_gt'].cpu().numpy().flatten()
        w_pred = model.predict_w(data['t_gt'])
        w_pred_np = w_pred.cpu().numpy().flatten()

        if config.ode_mismatch_enabled and data['w_mismatch'] is not None:
            w_mismatch_np = data['w_mismatch'].cpu().numpy().flatten()
        else:
            w_mismatch_np = None

        u_pred = model.predict_u(data['X_gt'], data['T_gt'])
        u_pred_np = u_pred.cpu().numpy().reshape(100, 100)
        u_true = data['U_gt'].cpu().numpy().reshape(100, 100)

    w_mae = np.mean(np.abs(w_pred_np - w_true_np))
    w_rel = np.linalg.norm(w_pred_np - w_true_np) / np.linalg.norm(w_true_np)

    u_mae = np.mean(np.abs(u_pred_np - u_true))
    u_rel = np.linalg.norm(u_pred_np - u_true) / np.linalg.norm(u_true)

    if w_mismatch_np is not None:
        w_mismatch_mae = np.mean(np.abs(w_mismatch_np - w_true_np))
        w_mismatch_rel = np.linalg.norm(w_mismatch_np - w_true_np) / np.linalg.norm(w_true_np)
    else:
        w_mismatch_mae = None
        w_mismatch_rel = None

    x_np = np.linspace(0, config.L, 100)
    X_np, T_np = np.meshgrid(x_np, t_np[:100])

    fig = plt.figure(figsize=(24, 10))

    ax1 = plt.subplot(2, 4, 1)
    ax1.plot(t_np, w_true_np, 'b-', lw=3, label='True w(t)', alpha=0.8)
    ax1.plot(t_np, w_pred_np, 'r--', lw=2.5, label='PINN w(t)', alpha=0.9)

    if w_mismatch_np is not None:
        ax1.plot(t_np, w_mismatch_np, 'm:', lw=2.5, label='Mismatched ODE w(t)', alpha=0.7)

    if data['w_obs'].numel() > 0:
        t_obs_np = data['t_w_obs'].cpu().numpy().flatten()
        w_obs_np = data['w_obs'].cpu().numpy().flatten()
        ax1.scatter(t_obs_np, w_obs_np, c='green', s=100, marker='^',
                    edgecolors='black', label='w Observations', zorder=5)

    ax1.text(0.05, 0.95, f'MAE: {w_mae:.6f}\nRel: {w_rel:.4f} ({w_rel * 100:.2f}%)',
             transform=ax1.transAxes, va='top', bbox=dict(facecolor='wheat', alpha=0.7))
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('w(t)')
    ax1.set_title('Memristor State Variable w(t)', fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 4, 2)
    alpha_true = config.alpha_min + (config.alpha_max - config.alpha_min) * w_true_np ** 2
    alpha_pred = config.alpha_min + (config.alpha_max - config.alpha_min) * w_pred_np ** 2

    ax2.plot(t_np, alpha_true, 'b-', lw=3, label='True α(w)', alpha=0.8)
    ax2.plot(t_np, alpha_pred, 'r--', lw=2.5, label='PINN α(w)', alpha=0.9)

    if w_mismatch_np is not None:
        alpha_mismatch = config.alpha_min + (config.alpha_max - config.alpha_min) * w_mismatch_np ** 2
        ax2.plot(t_np, alpha_mismatch, 'm:', lw=2.5, label='Mismatched α(w)', alpha=0.7)

    ax2.set_xlabel('Time t')
    ax2.set_ylabel('α(w)')
    ax2.set_title('Diffusion Coefficient α(w) Evolution', fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 4, 3)

    def sample_data(data, interval=100):
        """Downsample data, keeping the last point"""
        data_array = np.array(data)
        indices = np.arange(0, len(data_array), interval)
        if indices[-1] != len(data_array) - 1:
            indices = np.append(indices, len(data_array) - 1)
        return indices, data_array[indices]

    if len(history['u_obs']) > 0:
        total_loss_history = []
        for i in range(len(history['u_obs'])):
            total = (config.w_u_obs * history['u_obs'][i] +
                     config.w_pde * history['pde'][i] +
                     config.w_ic * history['ic'][i] +
                     config.w_bc * history['bc'][i] +
                     config.w_physics * history['physics'][i] +
                     config.w_w_obs * history['w_obs'][i])
            total_loss_history.append(total)

        epochs, losses = sample_data(total_loss_history, interval=100)

        ax3.semilogy(epochs, losses, 'b-', lw=1.5, alpha=0.8)
        ax3.set_xlabel('Epoch', fontsize=10)
        ax3.set_ylabel('Loss (log scale)', fontsize=10)
        ax3.set_title('Total Loss vs Epoch', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.scatter(epochs[::5], losses[::5], c='red', s=10, alpha=0.5, zorder=5)
        final_loss = losses[-1]
        ax3.text(0.98, 0.98, f'Final: {final_loss:.2e}',
                 transform=ax3.transAxes,
                 ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                 fontsize=9)

    ax4 = plt.subplot(2, 4, 4)
    w_error_pinn = np.abs(w_pred_np - w_true_np)
    ax4.plot(t_np, w_error_pinn, 'r-', lw=2, label='PINN Error', alpha=0.8)
    ax4.fill_between(t_np, 0, w_error_pinn, alpha=0.2, color='red')

    if w_mismatch_np is not None:
        w_error_mismatch = np.abs(w_mismatch_np - w_true_np)
        ax4.plot(t_np, w_error_mismatch, 'm--', lw=2, label='Mismatch ODE Error', alpha=0.7)
        ax4.fill_between(t_np, 0, w_error_mismatch, alpha=0.15, color='magenta')

    ax4.set_xlabel('Time t')
    ax4.set_ylabel('Absolute Error |w - w_true|')
    ax4.set_title('w(t) Error Comparison (PINN vs Mismatch ODE)', fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(2, 4, 5)
    c1 = ax5.contourf(T_np, X_np, u_true, levels=50, cmap='viridis')
    ax5.set_xlabel('Time t')
    ax5.set_ylabel('Space x')
    ax5.set_title('u(x,t) Ground Truth', fontweight='bold')
    plt.colorbar(c1, ax=ax5)

    ax6 = plt.subplot(2, 4, 6)
    c2 = ax6.contourf(T_np, X_np, u_pred_np, levels=50, cmap='viridis')
    ax6.set_xlabel('Time t')
    ax6.set_ylabel('Space x')
    ax6.set_title('u(x,t) PINN Prediction', fontweight='bold')
    plt.colorbar(c2, ax=ax6)

    ax7 = plt.subplot(2, 4, 7)
    u_error = np.abs(u_true - u_pred_np)
    c3 = ax7.contourf(T_np, X_np, u_error, levels=50, cmap='Reds')
    ax7.text(0.05, 0.95, f'MAE: {u_mae:.6f}\nRel: {u_rel:.4f} ({u_rel * 100:.2f}%)',
             transform=ax7.transAxes, va='top', bbox=dict(facecolor='wheat', alpha=0.7))
    ax7.set_xlabel('Time t')
    ax7.set_ylabel('Space x')
    ax7.set_title('u(x,t) Absolute Error', fontweight='bold')
    plt.colorbar(c3, ax=ax7)

    ax8 = plt.subplot(2, 4, 8)
    nt = len(t_np[:100])
    time_slices = [0, nt // 4, nt // 2, 3 * nt // 4, nt - 1]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    for idx, ti in enumerate(time_slices):
        ax8.plot(x_np, u_true[ti, :], c=colors[idx], ls='-', lw=2.5, label=f't={t_np[ti]:.2f}')
        ax8.plot(x_np, u_pred_np[ti, :], c=colors[idx], ls='--', lw=2, alpha=0.7)
    ax8.set_xlabel('Space x')
    ax8.set_ylabel('u(x,t)')
    ax8.set_title('Spatial Profiles (solid: true, dash: PINN)', fontweight='bold')
    ax8.legend(fontsize=8, ncol=2)
    ax8.grid(True, alpha=0.3)

    if config.ode_mismatch_enabled:
        title = (f'ODE Mismatch Experiment - Voltage Waveform: {config.voltage_waveform}\n'
                 f'True: μ={params_true["mu"]:.3f} | '
                 f'Training: μ_mismatch={params_train["mu"]:.3f} | '
                 f'Mismatch: {config.ode_mismatch_type} δ={config.ode_mismatch_delta_mu:.3f}')
    else:
        title = f'Memristor PINN - Voltage Waveform: {config.voltage_waveform}\nμ={params_true["mu"]:.3f}'

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('ode_mismatch_result.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n{'=' * 80}")
    print("Error Assessment")
    print(f"{'=' * 80}")
    print(f"w(t) Reconstruction Error:")
    print(f"  MAE:     {w_mae:.6e}")
    print(f"  Rel L2:  {w_rel:.4f} ({w_rel * 100:.2f}%)")
    print(f"\nu(x,t) Reconstruction Error:")
    print(f"  MAE:     {u_mae:.6e}")
    print(f"  Rel L2:  {u_rel:.4f} ({u_rel * 100:.2f}%)")

    if config.ode_mismatch_enabled:
        print(f"\n{'=' * 80}")
        print("ODE Mismatch Analysis")
        print(f"{'=' * 80}")
        print(f"True μ:      {params_true['mu']:.4f}")
        print(f"Training μ:  {params_train['mu']:.4f}")
        print(f"Deviation δ: {config.ode_mismatch_delta_mu:.4f}")
        print(f"Relative Deviation: {abs(params_train['mu'] - params_true['mu']) / params_true['mu'] * 100:.2f}%")

        print(f"\n--- w(t) Error Comparison ---")
        print(f"Mismatched ODE Error (direct solution of mismatched equation):")
        print(f"  MAE:     {w_mismatch_mae:.6e}")
        print(f"  Rel L2:  {w_mismatch_rel:.4f} ({w_mismatch_rel * 100:.2f}%)")
        print(f"\nPINN Prediction Error (after PDE constraint correction):")
        print(f"  MAE:     {w_mae:.6e}")
        print(f"  Rel L2:  {w_rel:.4f} ({w_rel * 100:.2f}%)")

        improvement = (w_mismatch_rel - w_rel) / w_mismatch_rel * 100
        if improvement > 0:
            print(f"\nImprovement: {improvement:.2f}% (relative error reduction)")
        else:
            print(f"\nImprovement: {improvement:.2f}% (relative error increase)")


def plot_training_history(history, sample_interval=100):
    """
    Plot training history curves for each loss term
    """
    import numpy as np

    fig, axes = plt.subplots(2, 3, figsize=(20, 8))

    def sample_data(data, interval=sample_interval):
        """Downsample, keeping the last point"""
        data_array = np.array(data)
        indices = np.arange(0, len(data_array), interval)
        if indices[-1] != len(data_array) - 1:
            indices = np.append(indices, len(data_array) - 1)
        return indices, data_array[indices]

    loss_items = [
        ('u_obs', 'u Observation Loss', 0, 0),
        ('pde', 'PDE Residual Loss', 0, 1),
        ('physics', 'Physics (ODE) Loss', 0, 2),
        ('w_obs', 'w Observation Loss', 1, 0),
        ('ic', 'IC Residual Loss', 1, 1),
        ('bc', 'BC Residual Loss', 1, 2),
    ]

    for key, title, row, col in loss_items:
        if key in history and len(history[key]) > 0:
            epochs, losses = sample_data(history[key])

            axes[row][col].semilogy(epochs, losses, 'b-', lw=1.5, alpha=0.8)
            axes[row][col].set_title(title, fontweight='bold', fontsize=12)
            axes[row][col].set_xlabel('Epoch', fontsize=10)
            axes[row][col].set_ylabel('Loss (log scale)', fontsize=10)
            axes[row][col].grid(True, alpha=0.3)
            axes[row][col].scatter(epochs[::5], losses[::5], c='red', s=10, alpha=0.5, zorder=5)

            final_loss = losses[-1]
            axes[row][col].text(0.98, 0.98, f'Final: {final_loss:.2e}',
                                transform=axes[row][col].transAxes,
                                ha='right', va='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                                fontsize=9)

    plt.suptitle(f'Training History (Sampled every {sample_interval} epochs)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 80)
    print("Training History Summary")
    print("=" * 80)
    print(f"Total epochs: {len(history['u_obs']) if 'u_obs' in history else 0}")
    print(f"Sample interval: {sample_interval}")
    print(f"Points plotted: {len(history['u_obs']) // sample_interval + 1 if 'u_obs' in history else 0}")
    print("\nFinal Losses:")
    for key, title, _, _ in loss_items:
        if key in history and len(history[key]) > 0:
            print(f"  {title:<30}: {history[key][-1]:.6e}")
    print("=" * 80)


def main():
    """Main program - ODE mismatch experiment"""

    config = MemristorPINNConfig()

    # Experiment setup
    config.voltage_waveform = 'sine'
    config.voltage_freq = 2.0

    config.ode_mismatch_enabled = True
    config.ode_mismatch_delta_mu = 0.3
    config.ode_mismatch_type = 'additive'

    config.n_w_obs = 3
    config.w_obs_sampling = 'random'

    config.w_use_fourier = True
    config.w_num_freq = 10
    config.w_output_mode = 'clamp'

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print("=" * 80)
    print("Memristor State Variable Identification - ODE Mismatch Experiment")
    print("=" * 80)
    print(f"\nExperimental Setup:")
    print(f"  Voltage Waveform: {config.voltage_waveform}")
    print(f"  Voltage Frequency: {config.voltage_freq:.2f} Hz")
    print(f"  ODE Mismatch: {'Enabled' if config.ode_mismatch_enabled else 'Disabled'}")
    print(f"  w Observation Points: n={config.n_w_obs}, Sampling={config.w_obs_sampling}")
    print(f"  WNet Fourier: {'Enabled' if config.w_use_fourier else 'Disabled'}")
    if config.w_use_fourier:
        print(f"  WNet Fourier Frequencies: {config.w_num_freq}")
    print(f"  WNet Output Mode: {config.w_output_mode}")

    if config.ode_mismatch_enabled:
        params_true = config.get_physics_params(use_mismatch=False)
        params_train = config.get_physics_params(use_mismatch=True)
        print(f"  True μ:     {params_true['mu']:.4f}")
        print(f"  Training μ: {params_train['mu']:.4f}")
        print(f"  Mismatch Type: {config.ode_mismatch_type}")
        print(f"  Deviation δ: {config.ode_mismatch_delta_mu:.4f}")
        print(f"\n  Experimental Objective: Verify whether PDE constraints can correct incorrect ODE physics model")

    print(f"  Device: {config.device}")
    print(f"  Training Epochs: {config.epochs}")

    data = generate_data(config)
    model = MemristorPINN(config)
    history = train(model, config, data)

    # Visualization
    print("\n" + "=" * 80)
    print("Visualization results")
    print("=" * 80)
    visualize(model, data, config, history)
    plot_training_history(history)


if __name__ == '__main__':
    main()