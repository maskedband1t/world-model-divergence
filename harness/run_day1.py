#!/usr/bin/env python3
"""Day 1: the noise floor, then the divergence curve.

Run the control FIRST. Nothing downstream is interpretable until the band exists.

  # 1. control: two REAL rollouts, same action sequence, sticky actions on
  python harness/run_day1.py control --game Breakout --diamond-root ../diamond

  # 2. divergence: world model vs real emulator, same action sequence
  python harness/run_day1.py diverge --game Breakout --diamond-root ../diamond --num-samples 8

Writes JSONL to results/. One record per (run, step).
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from bridge import Distance, load_pretrained, rollout_real, rollout_world_model


def provenance(diamond_root: Path) -> dict:
    def git(*a):
        try:
            return subprocess.check_output(["git", "-C", str(diamond_root), *a], text=True).strip()
        except Exception:  # noqa: BLE001
            return "unknown"
    return {
        "diamond_commit": git("rev-parse", "HEAD"),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write(path: Path, header: dict, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(json.dumps({"_header": header}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {path}")


# --------------------------------------------------------------------------------------

def cmd_control(args: argparse.Namespace) -> None:
    """The noise floor: two REAL rollouts of ONE pilot action sequence, sticky on.

    *NoFrameskip-v4 is deterministic, so with sticky actions off two real rollouts of the
    same sequence are bit-identical and the floor would be exactly zero.
    repeat_action_probability = 0.25 is the ALE standard (Machado et al. 2018).

    Both arms replay the pilot OPEN-LOOP. An earlier version had arm A acting closed-loop;
    that floor measured the policy's ability to react as well as the sticky noise, which
    inflates it. See PREDICTIONS.md amendment 3.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loaded = load_pretrained(args.game, Path(args.diamond_root), device)
    dist = Distance(device, use_lpips=not args.no_lpips)

    rows: list[dict] = []
    skipped = 0
    for seed in range(args.seeds):
        # pilot: policy acts, deterministic env. This is the SAME sequence the
        # divergence run uses, so the floor and the thing it calibrates are matched.
        pilot = rollout_real(loaded, steps=args.burnin + args.horizon, seed=1000 + seed, sticky=0.0)
        if pilot.act.numel() < args.burnin + args.horizon:
            print(f"[skip] seed {seed}: pilot ended at {pilot.ended_at}")
            skipped += 1
            continue

        a = rollout_real(loaded, steps=args.burnin + args.horizon, seed=3000 + seed,
                         actions=pilot.act, sticky=args.sticky)
        b = rollout_real(loaded, steps=args.burnin + args.horizon, seed=4000 + seed,
                         actions=pilot.act, sticky=args.sticky)
        if min(a.act.numel(), b.act.numel()) < args.burnin + args.horizon:
            print(f"[skip] seed {seed}: a control arm ended early")
            skipped += 1
            continue

        t0 = args.burnin
        l2, lp = dist(a.obs[t0 + 1 : t0 + 1 + args.horizon],
                      b.obs[t0 + 1 : t0 + 1 + args.horizon])
        for k in range(len(l2)):
            rows.append({
                "run": "control", "game": args.game, "seed": seed, "step": k + 1,
                "l2": float(l2[k]), "lpips": None if lp is None else float(lp[k]),
            })
        print(f"seed {seed}: control done")

    write(
        Path(args.out) / f"control_{args.game}.jsonl",
        {"mode": "control", "game": args.game, "sticky": args.sticky,
         "burnin": args.burnin, "horizon": args.horizon, "seeds": args.seeds,
         "seeds_skipped": skipped, "open_loop_both_arms": True,
         **provenance(Path(args.diamond_root))},
        rows,
    )


def cmd_diverge(args: argparse.Namespace) -> None:
    """World model vs real emulator under ONE shared action sequence.

    Sticky actions are OFF here: the real env is deterministic, so every bit of the
    distance below is model error with no environment noise mixed in.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loaded = load_pretrained(args.game, Path(args.diamond_root), device)
    dist = Distance(device, use_lpips=not args.no_lpips)
    T = loaded.num_steps_conditioning

    rows: list[dict] = []
    embeds: list[np.ndarray] = []
    embed_index: list[tuple] = []
    skipped = 0
    for seed in range(args.seeds):
        # seed 1000+seed matches the control run's pilot, so both are measured on the
        # same action sequence. Real rollout FIRST: rollout_world_model reseeds global RNG.
        real = rollout_real(loaded, steps=args.burnin + args.horizon, seed=1000 + seed, sticky=0.0)
        if real.act.numel() < args.burnin + args.horizon:
            print(f"[skip] seed {seed}: episode ended at {real.ended_at}")
            skipped += 1
            continue

        t0 = args.burnin
        # conditioning window: the T real frames ending at t0, and their actions
        init_obs = real.obs[t0 - T + 1 : t0 + 1]
        init_act = real.act[t0 - T + 1 : t0 + 1]
        future_act = real.act[t0 : t0 + args.horizon]
        target = real.obs[t0 + 1 : t0 + 1 + args.horizon]

        # actor-critic state at t0, carried into the dream so the auditor's recurrent
        # state reflects the real burn-in rather than restarting from zeros
        init_hx_cx = None
        if real.hx is not None and real.hx.shape[0] > t0 - 1:
            init_hx_cx = (real.hx[t0 - 1], real.cx[t0 - 1])

        samples = []
        for s in range(args.num_samples):
            pred, feats = rollout_world_model(
                loaded, init_obs, init_act, future_act,
                num_steps_denoising=args.denoising_steps,
                seed=10_000 * seed + s,
                init_hx_cx=init_hx_cx,
                probe_features=not args.no_probe,
            )
            samples.append(pred)
            if feats is not None:
                embeds.append(np.asarray(feats["embed"], dtype=np.float32))
                embed_index.append((seed, s))
                for k in range(len(feats["pi_entropy"])):
                    rows.append({
                        "run": "probe", "game": args.game, "seed": seed, "sample": s,
                        "step": k + 1,
                        "pi_entropy": feats["pi_entropy"][k],
                        "value": feats["value"][k],
                        "delta": feats["delta"][k],
                        "pix_mean": feats["pix_mean"][k],
                        "pix_std": feats["pix_std"][k],
                    })
            l2, lp = dist(pred, target)
            for k in range(len(l2)):
                rows.append({
                    "run": "diverge", "game": args.game, "seed": seed, "sample": s,
                    "step": k + 1, "l2": float(l2[k]),
                    "lpips": None if lp is None else float(lp[k]),
                })

        # sample spread: the model's own confidence signal (see PREDICTIONS.md P3/P4).
        # s_churn = 0, so this spread comes purely from the initial latent draw.
        if args.num_samples > 1:
            stack = torch.stack(samples)  # (S, H, 3, 64, 64)
            mean = stack.mean(0, keepdim=True)
            spread = (stack - mean).flatten(2).pow(2).mean(-1).sqrt().mean(0)  # (H,)
            for k in range(spread.numel()):
                rows.append({
                    "run": "spread", "game": args.game, "seed": seed,
                    "step": k + 1, "spread": float(spread[k]),
                })
        print(f"seed {seed}: {args.num_samples} sample(s) done")

    write(
        Path(args.out) / f"diverge_{args.game}.jsonl",
        {"mode": "diverge", "game": args.game, "sticky": 0.0,
         "burnin": args.burnin, "horizon": args.horizon, "seeds": args.seeds,
         "num_samples": args.num_samples, "denoising_steps": args.denoising_steps,
         "num_steps_conditioning": T, "seeds_skipped": skipped,
         "probe_features": not args.no_probe,
         **provenance(Path(args.diamond_root))},
        rows,
    )

    if embeds:
        out_npz = Path(args.out) / f"embed_{args.game}.npz"
        np.savez_compressed(
            out_npz,
            embed=np.stack(embeds),                              # (n_rollouts, H, C)
            index=np.asarray(embed_index, dtype=np.int64),       # (n_rollouts, 2) seed, sample
        )
        print(f"wrote embeddings {np.stack(embeds).shape} -> {out_npz}")


# --------------------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn in (("control", cmd_control), ("diverge", cmd_diverge)):
        s = sub.add_parser(name)
        s.add_argument("--game", default="Breakout", help="Atari-100k game, e.g. Breakout Pong MsPacman Boxing")
        s.add_argument("--diamond-root", required=True, help="path to a clone of eloialonso/diamond")
        s.add_argument("--burnin", type=int, default=40, help="real steps before the comparison starts")
        s.add_argument("--horizon", type=int, default=80, help="steps compared; > 50 so H=50 is interior")
        s.add_argument("--seeds", type=int, default=10)
        s.add_argument("--out", default="results")
        s.add_argument("--no-lpips", action="store_true")
        if name == "control":
            s.add_argument("--sticky", type=float, default=0.25, help="ALE repeat_action_probability")
        else:
            s.add_argument("--num-samples", type=int, default=8, help="independent WM rollouts per seed")
            s.add_argument("--denoising-steps", type=int, default=3, help="DIAMOND's Atari default")
            s.add_argument("--no-probe", action="store_true",
                           help="skip Arm B features (docs/PROBE.md); they are nearly free, so this is for debugging only")
        s.set_defaults(func=fn)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
