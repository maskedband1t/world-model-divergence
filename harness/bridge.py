"""Bridge to DIAMOND: load the pretrained agent, drive the real emulator and the
world model off one shared action sequence, and measure how far apart they drift.

Verified against eloialonso/diamond @ main:
  - obs are (B, 3, 64, 64) in [-1, 1]              (envs/env.py TorchEnv)
  - denoiser conditions on 4 frames                (config/agent/default.yaml)
  - DiffusionSampler.sample(prev_obs, prev_act) -> (next_obs, trajectory)
  - default sampler has s_churn = 0.0, so the ONLY stochasticity is the initial
    latent draw. Sample spread therefore comes from the torch seed per rollout.
  - *NoFrameskip-v4 is DETERMINISTIC (no sticky actions). The noise floor must be
    produced by explicitly enabling repeat_action_probability. See PREDICTIONS.md
    amendment 2026-09-24.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


# --------------------------------------------------------------------------------------
# DIAMOND import plumbing
# --------------------------------------------------------------------------------------

def add_diamond_to_path(diamond_root: Path) -> None:
    src = Path(diamond_root).expanduser().resolve() / "src"
    if not src.is_dir():
        raise SystemExit(f"--diamond-root {diamond_root} has no src/ (clone eloialonso/diamond first)")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


@dataclass
class Loaded:
    agent: "object"
    num_actions: int
    num_steps_conditioning: int
    device: torch.device
    make_env: "object"


def load_pretrained(game: str, diamond_root: Path, device: torch.device):
    """Download the Atari-100k checkpoint for `game` and build agent + env factory.

    Mirrors src/play.py::prepare_play_mode but never prompts and lets us set
    repeat_action_probability, which play.py does not expose.
    """
    add_diamond_to_path(diamond_root)

    from huggingface_hub import hf_hub_download
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from agent import Agent  # noqa: E402

    def dl(fn: str) -> Path:
        return Path(hf_hub_download(repo_id="eloialonso/diamond", filename=fn))

    ckpt = dl(f"atari_100k/models/{game}.pt")
    cfg_agent = OmegaConf.load(dl("atari_100k/config/agent/default.yaml"))
    cfg_env = OmegaConf.load(dl("atari_100k/config/env/atari.yaml"))
    cfg_env.train.id = cfg_env.test.id = f"{game}NoFrameskip-v4"

    # agent/default.yaml interpolates ${env.train.size} and ${agent...}, so the two
    # must live under one root before anything is resolved.
    cfg = OmegaConf.create({"agent": cfg_agent, "env": cfg_env})

    def make_env(num_envs: int = 1, sticky: float = 0.0):
        return make_atari_env_sticky(
            id=cfg.env.test.id,
            num_envs=num_envs,
            device=device,
            done_on_life_loss=bool(cfg.env.test.done_on_life_loss),
            size=int(cfg.env.test.size),
            max_episode_steps=cfg.env.test.max_episode_steps,
            repeat_action_probability=sticky,
        )

    probe = make_env(1)
    num_actions = probe.num_actions
    probe.close()

    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device).eval()
    agent.load(ckpt)

    return Loaded(
        agent=agent,
        num_actions=num_actions,
        num_steps_conditioning=int(cfg.agent.denoiser.inner_model.num_steps_conditioning),
        device=device,
        make_env=make_env,
    )


def make_atari_env_sticky(
    id: str,
    num_envs: int,
    device: torch.device,
    done_on_life_loss: bool,
    size: int,
    max_episode_steps: Optional[int],
    repeat_action_probability: float,
):
    """DIAMOND's make_atari_env with repeat_action_probability exposed.

    repeat_action_probability = 0.0  -> deterministic, the setting DIAMOND trained on
    repeat_action_probability = 0.25 -> the ALE standard (Machado et al. 2018); used
                                        ONLY for the control run that sets the noise floor
    """
    import ale_py  # noqa: F401  (registers the ALE envs)
    import gymnasium
    from gymnasium.vector import AsyncVectorEnv

    from envs.atari_preprocessing import AtariPreprocessing  # noqa: E402
    from envs.env import DoneOnLifeLoss, TorchEnv  # noqa: E402

    def env_fn():
        try:
            env = gymnasium.make(
                id,
                full_action_space=False,
                frameskip=1,
                render_mode="rgb_array",
                max_episode_steps=max_episode_steps,
                repeat_action_probability=repeat_action_probability,
            )
        except TypeError as e:
            raise SystemExit(
                "This ale_py/gymnasium does not accept repeat_action_probability as a "
                f"make() kwarg ({e}). The noise-floor control needs it; pin ale_py>=0.9."
            )
        env = AtariPreprocessing(env=env, noop_max=30, frame_skip=4, screen_size=size)
        return env

    env = AsyncVectorEnv([env_fn for _ in range(num_envs)])
    if done_on_life_loss:
        env = DoneOnLifeLoss(env)
    return TorchEnv(env, device)


# --------------------------------------------------------------------------------------
# Rollouts
# --------------------------------------------------------------------------------------

@dataclass
class RealRollout:
    obs: Tensor          # (T+1, 3, 64, 64) in [-1, 1]; obs[i] is the frame BEFORE act[i]
    act: Tensor          # (T,) long
    ended_at: Optional[int]  # step index where the episode died, else None
    hx: Optional[Tensor] = None  # (T, lstm_dim) actor-critic state AFTER seeing obs[i]
    cx: Optional[Tensor] = None  # only populated when the policy acted (actions=None)


@torch.no_grad()
def rollout_real(
    loaded: Loaded,
    steps: int,
    seed: int,
    actions: Optional[Tensor] = None,
    sticky: float = 0.0,
) -> RealRollout:
    """Run the real emulator for `steps`.

    actions=None -> the pretrained policy chooses, closed-loop. Use this ONCE, as a
                    pilot, to produce an on-distribution action sequence.
    actions given -> open-loop replay of that exact sequence.

    Every comparison in this repo replays a pilot sequence open-loop. A closed-loop
    arm compared against an open-loop arm would differ by the policy's ability to
    react as well as by the effect under test -- see PREDICTIONS.md amendment 3.
    """
    env = loaded.make_env(1, sticky=sticky)
    obs, _ = env.reset(seed=seed)
    hx = torch.zeros(1, loaded.agent.actor_critic.lstm_dim, device=loaded.device)
    cx = torch.zeros_like(hx)

    obs_log: List[Tensor] = [obs[0].cpu().clone()]
    act_log: List[int] = []
    hx_log: List[Tensor] = []
    cx_log: List[Tensor] = []
    ended_at: Optional[int] = None

    for t in range(steps):
        if actions is None:
            logits, _, (hx, cx) = loaded.agent.actor_critic.predict_act_value(obs, (hx, cx))
            act = torch.distributions.Categorical(logits=logits).sample()
            hx_log.append(hx[0].cpu().clone())
            cx_log.append(cx[0].cpu().clone())
        else:
            act = actions[t].view(1).to(loaded.device)

        obs, _, end, trunc, _ = env.step(act)
        obs_log.append(obs[0].cpu().clone())
        act_log.append(int(act.item()))

        if bool(end.any()) or bool(trunc.any()):
            ended_at = t
            break

    env.close()
    return RealRollout(
        obs=torch.stack(obs_log),
        act=torch.tensor(act_log, dtype=torch.long),
        ended_at=ended_at,
        hx=torch.stack(hx_log) if hx_log else None,
        cx=torch.stack(cx_log) if cx_log else None,
    )


@torch.no_grad()
def rollout_world_model(
    loaded: Loaded,
    init_obs: Tensor,     # (num_steps_conditioning, 3, 64, 64) real frames up to and including t0
    init_act: Tensor,     # (num_steps_conditioning,) actions aligned with init_obs
    future_act: Tensor,   # (H,) the actions the real env took from t0 onward
    num_steps_denoising: int,
    seed: int,
    init_hx_cx: Optional[Tuple[Tensor, Tensor]] = None,
    probe_features: bool = False,
) -> Tuple[Tensor, Optional[dict]]:
    """Predict H frames from the world model under a GIVEN action sequence.

    Returns ((H, 3, 64, 64), features). Frame k predicts real obs[t0 + 1 + k].

    With probe_features=True, also returns Arm B's inputs (docs/PROBE.md), computed
    ONLY from what a deployed system has at step k: its own generated frames and the
    actions. Never the real frames, never the label.

    The actor-critic is carried forward from the real burn-in via init_hx_cx, so its
    recurrent state reflects the trajectory rather than restarting from zeros. It was
    trained on real frames, which is the whole point -- it is the auditor, not the
    generator (amendment 6).
    """
    from models.diffusion import DiffusionSampler, DiffusionSamplerConfig  # noqa: E402

    sampler = DiffusionSampler(
        loaded.agent.denoiser,
        DiffusionSamplerConfig(num_steps_denoising=num_steps_denoising),
    )

    # s_churn is 0 by default, so the initial latent is the only source of randomness.
    # Seeding here is what makes N independent samples differ.
    #
    # NOTE: DiffusionSampler.sample calls torch.randn with no generator argument, so the
    # global RNG is the only lever available. This function therefore has a side effect
    # on global torch RNG state -- callers must finish every real-env rollout BEFORE
    # calling this, or the real rollouts stop being reproducible from their own seeds.
    torch.manual_seed(seed)

    obs_buf = init_obs.unsqueeze(0).to(loaded.device).clone()   # (1, T, 3, 64, 64)
    act_buf = init_act.unsqueeze(0).to(loaded.device).clone()   # (1, T)

    ac = loaded.agent.actor_critic
    if probe_features:
        if init_hx_cx is None:
            hx = torch.zeros(1, ac.lstm_dim, device=loaded.device)
            cx = torch.zeros_like(hx)
        else:
            hx = init_hx_cx[0].view(1, -1).to(loaded.device).clone()
            cx = init_hx_cx[1].view(1, -1).to(loaded.device).clone()
    feats: dict = {"pi_entropy": [], "value": [], "delta": [],
                   "pix_mean": [], "pix_std": [], "embed": []}

    out: List[Tensor] = []
    H = future_act.numel()
    for k in range(H):
        prev_frame = obs_buf[:, -1].clone()
        # act_buffer[:, -1] is the action taken FROM the newest frame (WorldModelEnv.step)
        act_buf[:, -1] = future_act[k].to(loaded.device)
        next_obs, _ = sampler.sample(obs_buf, act_buf)
        out.append(next_obs[0].cpu().clone())

        if probe_features:
            logits, val, (hx, cx) = ac.predict_act_value(next_obs, (hx, cx))
            ent = torch.distributions.Categorical(logits=logits).entropy()
            emb = ac.encoder(next_obs).mean(dim=(2, 3))  # (1, C) channel means
            feats["pi_entropy"].append(float(ent.item()))
            feats["value"].append(float(val.item()))
            feats["delta"].append(float((next_obs - prev_frame).flatten(1).pow(2).mean(1).sqrt().item()))
            feats["pix_mean"].append(float(next_obs.mean().item()))
            feats["pix_std"].append(float(next_obs.std().item()))
            feats["embed"].append(emb[0].cpu().numpy())

        obs_buf = obs_buf.roll(-1, dims=1)
        act_buf = act_buf.roll(-1, dims=1)
        obs_buf[:, -1] = next_obs

    return torch.stack(out), (feats if probe_features else None)


# --------------------------------------------------------------------------------------
# Distance
# --------------------------------------------------------------------------------------

class Distance:
    """Per-step distance between two frame sequences.

    L2 is the dumb metric and is always reported. LPIPS is the perceptual one.
    PREDICTIONS.md: if the two disagree on a headline claim, that disagreement is
    the finding and both get reported.
    """

    def __init__(self, device: torch.device, use_lpips: bool = True) -> None:
        self.device = device
        self.lpips = None
        if use_lpips:
            try:
                import lpips as lpips_pkg
                self.lpips = lpips_pkg.LPIPS(net="alex").to(device).eval()
            except Exception as e:  # noqa: BLE001
                print(f"[warn] LPIPS unavailable ({e}); reporting L2 only", file=sys.stderr)

    @torch.no_grad()
    def __call__(self, a: Tensor, b: Tensor) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """a, b: (H, 3, 64, 64) in [-1, 1]. Returns (l2_per_step, lpips_per_step)."""
        assert a.shape == b.shape, f"{a.shape} vs {b.shape}"
        a = a.to(self.device).float()
        b = b.to(self.device).float()
        l2 = (a - b).flatten(1).pow(2).mean(1).sqrt().cpu().numpy()
        lp = None
        if self.lpips is not None:
            lp = self.lpips(a, b).flatten().cpu().numpy()
        return l2, lp
