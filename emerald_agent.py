#!/usr/bin/env python3
"""
Emerald Agent: single-file RL agent to learn and document playing Pokemon Emerald.

Features (single-file, minimal external setup beyond pip):
- Gym Retro integration for GBA emulation (user provides ROM; see usage below)
- Discrete GBA control mapping on top of Retro's MultiBinary action space
- Vision preprocessing (grayscale 84x84) compatible with SB3 CNN policies
- Intrinsic exploration reward via hashed-observation novelty
- PPO training and play modes (Stable-Baselines3)
- Lightweight knowledge base and Markdown guide generation from discovered scenes

IMPORTANT: ROMs are not included and must be imported to Gym Retro locally.

Quickstart (after installing requirements):
1) Import your ROM directory into Gym Retro (one-time):
   python -m retro.import /path/to/your/roms
   # Then list games to find the correct game id
   python - <<'PY'
import retro
print(sorted(retro.list_games())[:50])
PY

2) Train (assumes a game id like 'PokemonEmeraldGBA' or similar; check your list):
   python emerald_agent.py train --retro-game PokemonEmeraldGBA --total-timesteps 2000000

3) Play with a trained model:
   python emerald_agent.py play --retro-game PokemonEmeraldGBA --model-path runs/latest/best_model.zip

4) Generate/update Markdown guide based on the accumulated knowledge JSON:
   python emerald_agent.py guide --knowledge-json runs/latest/knowledge.json --out guide.md

Notes:
- The exact Gym Retro game id and available states depend on your import. Use `retro.list_games()` and `retro.list_states(game)` to verify.
- This script avoids OS-level dependencies (like tesseract). Scene discovery is image-hash based; extend as needed.

"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Soft imports to allow guidance even if packages are not installed yet
try:
    import retro  # Gym Retro emulator
except Exception as e:  # pragma: no cover - optional at import time
    retro = None  # type: ignore

try:
    import cv2  # OpenCV for vision preprocessing
except Exception as e:  # pragma: no cover
    cv2 = None  # type: ignore

# Gymnasium/Stable-Baselines3
try:
    import gymnasium as gym
except Exception:  # pragma: no cover
    gym = None  # type: ignore

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
    from stable_baselines3.common.env_util import make_vec_env
except Exception:  # pragma: no cover
    PPO = None  # type: ignore
    EvalCallback = None  # type: ignore
    CheckpointCallback = None  # type: ignore
    DummyVecEnv = None  # type: ignore
    SubprocVecEnv = None  # type: ignore
    VecFrameStack = None  # type: ignore
    make_vec_env = None  # type: ignore


# ------------------------
# Utilities
# ------------------------

def ensure_packages() -> None:
    missing: List[str] = []
    if retro is None:
        missing.append("retro (gym-retro)")
    if gym is None:
        missing.append("gymnasium")
    if PPO is None:
        missing.append("stable-baselines3")
    if cv2 is None:
        missing.append("opencv-python")
    if missing:
        print("ERROR: Missing required packages:")
        for m in missing:
            print(f" - {m}")
        print("\nInstall with: pip install -r requirements.txt")
        sys.exit(1)


def now_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ------------------------
# Knowledge base and guide
# ------------------------

@dataclasses.dataclass
class SceneRecord:
    first_seen_step: int
    last_seen_step: int
    visit_count: int
    sample_image_relpath: Optional[str] = None


class KnowledgeBase:
    """Tracks discovered scenes and simple statistics to generate a gameplay guide."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.scenes: Dict[str, SceneRecord] = {}
        self.total_steps: int = 0
        self.events: List[Dict[str, Any]] = []  # extensible for richer events later
        self.scene_images_dir = self.run_dir / "scenes"
        self.scene_images_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_path = self.run_dir / "knowledge.json"

    def log_step(self) -> None:
        self.total_steps += 1

    def record_scene(self, scene_hash: str, step: int, frame_bgr: Optional[np.ndarray] = None) -> None:
        if scene_hash not in self.scenes:
            relpath: Optional[str] = None
            if frame_bgr is not None and cv2 is not None:
                # Save a sample image for the guide
                out_path = self.scene_images_dir / f"scene_{scene_hash[:12]}.png"
                try:
                    cv2.imwrite(str(out_path), frame_bgr)
                    relpath = os.path.relpath(out_path, self.run_dir)
                except Exception:
                    relpath = None
            self.scenes[scene_hash] = SceneRecord(
                first_seen_step=step, last_seen_step=step, visit_count=1, sample_image_relpath=relpath
            )
            self.events.append({
                "type": "new_scene",
                "scene_hash": scene_hash,
                "step": step,
                "image": relpath,
                "timestamp": time.time(),
            })
        else:
            rec = self.scenes[scene_hash]
            rec.last_seen_step = step
            rec.visit_count += 1

    def save_json(self) -> None:
        data = {
            "total_steps": self.total_steps,
            "scenes": {k: dataclasses.asdict(v) for k, v in self.scenes.items()},
            "events": self.events,
        }
        with open(self.knowledge_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(json_path: Path) -> "KnowledgeBase":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        kb = KnowledgeBase(run_dir=json_path.parent)
        kb.total_steps = data.get("total_steps", 0)
        kb.scenes = {
            k: SceneRecord(**v) for k, v in data.get("scenes", {}).items()
        }
        kb.events = data.get("events", [])
        return kb

    def write_markdown_guide(self, out_path: Path) -> None:
        lines: List[str] = []
        lines.append("# Pokemon Emerald – Learning Agent Guide\n")
        lines.append(f"Generated: {datetime.utcnow().isoformat()}Z\n")
        lines.append("")
        lines.append("## Overview\n")
        lines.append(f"- Total steps observed: {self.total_steps}")
        lines.append(f"- Unique scenes discovered: {len(self.scenes)}\n")

        if self.scenes:
            lines.append("## Discovered Scenes\n")
            # Sort scenes by first seen
            ordered = sorted(self.scenes.items(), key=lambda kv: kv[1].first_seen_step)
            for scene_hash, rec in ordered:
                lines.append(f"### Scene {scene_hash[:12]}")
                lines.append(f"- First seen at step: {rec.first_seen_step}")
                lines.append(f"- Last seen at step: {rec.last_seen_step}")
                lines.append(f"- Visits: {rec.visit_count}")
                if rec.sample_image_relpath:
                    rel = rec.sample_image_relpath.replace("\\", "/")
                    lines.append(f"- Snapshot: ![]({rel})")
                lines.append("")

        lines.append("## Event Log (truncated)\n")
        for e in self.events[:200]:
            lines.append(f"- {e['type']} at step {e['step']}: {e.get('scene_hash','')} {e.get('image','')}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# ------------------------
# Vision and hashing
# ------------------------

class VisionProcessor:
    """Transforms raw RGB frames to grayscale 84x84 float32 and computes hashes."""

    def __init__(self, output_size: Tuple[int, int] = (84, 84)) -> None:
        self.output_size = output_size
        if cv2 is None:
            raise RuntimeError("opencv-python is required for VisionProcessor")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        # Retro frames are RGB; OpenCV expects BGR for some ops, but we'll handle directly
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, self.output_size, interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    def scene_hash(self, frame: np.ndarray) -> str:
        # Robust-ish hash: downscale + average hash + sha256 for stability
        gray_small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (32, 32), interpolation=cv2.INTER_AREA)
        avg = gray_small.mean()
        bits = (gray_small > avg).astype(np.uint8)
        bitstring = bits.flatten().tobytes()
        return hashlib.sha256(bitstring).hexdigest()

    def novelty_hash(self, processed_obs: np.ndarray) -> int:
        # SimHash of processed observation to track novelty
        x = (processed_obs * 255.0).astype(np.uint8)
        x_small = cv2.resize(x, (16, 16), interpolation=cv2.INTER_AREA)
        vector = x_small.flatten().astype(np.int32)
        # Simple simhash-like: project into 64-bit
        acc = np.zeros(64, dtype=np.int64)
        rng = np.random.RandomState(12345)  # fixed projection for determinism
        projections = rng.randint(0, 2, size=(vector.size, 64), dtype=np.int64) * 2 - 1
        acc += (vector[:, None] * projections).sum(axis=0)
        bits = (acc > 0).astype(np.uint8)
        out = 0
        for b in bits.tolist():
            out = (out << 1) | int(b)
        return out


# ------------------------
# Retro wrappers (Discrete controls, preprocessing, intrinsic reward)
# ------------------------

class GBAActionDiscretizer(gym.ActionWrapper):
    """Map a small discrete action set to Retro's MultiBinary GBA buttons.

    The mapping is derived from env.unwrapped.buttons when available.
    """

    def __init__(self, env: gym.Env, include_diagonals: bool = False) -> None:
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary), "Retro expects MultiBinary action space"
        self.buttons: List[str] = []
        try:
            self.buttons = list(env.unwrapped.buttons)  # type: ignore[attr-defined]
        except Exception:
            # Common GBA order fallback
            self.buttons = [
                "B", "A", "SELECT", "START", "RIGHT", "LEFT", "UP", "DOWN", "R", "L",
            ]
        self.button_index: Dict[str, int] = {b: i for i, b in enumerate(self.buttons)}

        # Define a compact action set
        def btn(*names: str) -> np.ndarray:
            arr = np.zeros(len(self.buttons), dtype=np.int8)
            for n in names:
                if n in self.button_index:
                    arr[self.button_index[n]] = 1
            return arr

        self.discrete_actions: List[np.ndarray] = [
            btn(),  # 0: NOOP
            btn("UP"),
            btn("DOWN"),
            btn("LEFT"),
            btn("RIGHT"),
            btn("A"),
            btn("B"),
            btn("START"),
            btn("SELECT"),
            btn("L"),
            btn("R"),
            btn("UP", "A"),
            btn("DOWN", "A"),
            btn("LEFT", "A"),
            btn("RIGHT", "A"),
            btn("RIGHT", "B"),  # run/right (if applicable)
            btn("LEFT", "B"),   # run/left
            btn("UP", "B"),     # run/up
            btn("DOWN", "B"),   # run/down
            btn("A", "B"),
        ]
        if include_diagonals:
            self.discrete_actions.extend([
                btn("UP", "LEFT"),
                btn("UP", "RIGHT"),
                btn("DOWN", "LEFT"),
                btn("DOWN", "RIGHT"),
            ])

        self.action_space = gym.spaces.Discrete(len(self.discrete_actions))

    def action(self, act: int) -> np.ndarray:
        return self.discrete_actions[int(act)].copy()


class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame; max over last two frames."""

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = []  # type: List[np.ndarray]

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}
        self._obs_buffer.clear()
        for _ in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += float(reward)
            terminated = terminated or term
            truncated = truncated or trunc
            if terminated or truncated:
                break
        max_frame = np.maximum(self._obs_buffer[-1], self._obs_buffer[-2]) if len(self._obs_buffer) >= 2 else self._obs_buffer[-1]
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class PreprocessObs(gym.ObservationWrapper):
    """Convert RGB to grayscale 84x84 float32 for CNN policies."""

    def __init__(self, env: gym.Env, vision: VisionProcessor) -> None:
        super().__init__(env)
        self.vision = vision
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(vision.output_size[1], vision.output_size[0]), dtype=np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return self.vision.preprocess(observation)


class IntrinsicNoveltyReward(gym.Wrapper):
    """Adds an intrinsic reward bonus for novel observations (hashed).

    r_total = r_env + beta * 1/sqrt(visit_count)
    """

    def __init__(self, env: gym.Env, vision: VisionProcessor, beta: float = 0.2) -> None:
        super().__init__(env)
        self.vision = vision
        self.beta = float(beta)
        self.visit_counts: Dict[int, int] = defaultdict(int)
        self._last_obs: Optional[np.ndarray] = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed = obs if obs.ndim == 2 else self.vision.preprocess(obs)
        h = self.vision.novelty_hash(processed)
        self.visit_counts[h] += 1
        intrinsic = self.beta * (1.0 / math.sqrt(self.visit_counts[h]))
        info = dict(info)
        info["intrinsic"] = intrinsic
        return obs, float(reward) + intrinsic, terminated, truncated, info


class SceneDiscoveryLogger(gym.Wrapper):
    """Logs discovered scenes into the knowledge base via image hashing."""

    def __init__(self, env: gym.Env, vision: VisionProcessor, kb: KnowledgeBase) -> None:
        super().__init__(env)
        self.vision = vision
        self.kb = kb
        self._step_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._step_count = 0
        self._log_scene(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        self.kb.log_step()
        # Log scene occasionally or when likely different
        if self._step_count % 5 == 0:
            self._log_scene(obs)
        return obs, reward, terminated, truncated, info

    def _log_scene(self, obs: np.ndarray) -> None:
        # Convert obs (RGB) to BGR for saving
        if obs is None:
            return
        try:
            if obs.ndim == 3 and obs.shape[2] == 3:
                rgb = obs
                bgr = rgb[:, :, ::-1]
            else:
                bgr = None
            scene_h = self.vision.scene_hash(obs if obs.ndim == 3 else np.repeat(obs[:, :, None], 3, axis=2))
            self.kb.record_scene(scene_h, step=self.kb.total_steps, frame_bgr=bgr)
        except Exception:
            pass


# ------------------------
# Environment factory
# ------------------------

def make_emerald_env(
    game: str,
    state: Optional[str],
    scenario: Optional[str],
    seed: int,
    run_dir: Path,
    frame_skip: int = 4,
    intrinsic_beta: float = 0.2,
    include_diagonals: bool = False,
    render_mode: Optional[str] = None,
) -> gym.Env:
    if retro is None or gym is None:
        ensure_packages()
    base_kwargs: Dict[str, Any] = {"game": game}
    if state:
        base_kwargs["state"] = state
    if scenario:
        base_kwargs["scenario"] = scenario
    if render_mode is not None:
        base_kwargs["render_mode"] = render_mode
    env = retro.make(**base_kwargs)
    env.reset(seed=seed)

    vision = VisionProcessor(output_size=(84, 84))
    kb = KnowledgeBase(run_dir=run_dir)

    # Compose wrappers: action -> skip -> scene log -> preprocess -> intrinsic
    env = GBAActionDiscretizer(env, include_diagonals=include_diagonals)
    env = MaxAndSkipEnv(env, skip=frame_skip)
    env = SceneDiscoveryLogger(env, vision=vision, kb=kb)
    env = PreprocessObs(env, vision=vision)
    env = IntrinsicNoveltyReward(env, vision=vision, beta=intrinsic_beta)

    return env


# ------------------------
# Train / Play / Guide
# ------------------------

def train(args: argparse.Namespace) -> None:
    ensure_packages()
    set_global_seeds(args.seed)

    run_dir = Path(args.run_dir or (Path("runs") / f"emerald_{now_stamp()}"))
    run_dir.mkdir(parents=True, exist_ok=True)

    # Vectorized env factory
    def make_thunk(rank: int):
        def _thunk():
            env = make_emerald_env(
                game=args.retro_game,
                state=args.state,
                scenario=args.scenario,
                seed=args.seed + rank,
                run_dir=run_dir,
                frame_skip=args.frame_skip,
                intrinsic_beta=args.intrinsic_beta,
                include_diagonals=args.include_diagonals,
                render_mode=None,
            )
            return env
        return _thunk

    n_envs = int(args.n_envs)
    vec_env = make_vec_env(env_id=lambda: None, n_envs=n_envs, env_kwargs=None, vec_env_cls=SubprocVecEnv,
                           env_fn=make_thunk)  # type: ignore[arg-type]

    # Frame-stack for temporal info
    vec_env = VecFrameStack(vec_env, n_stack=int(args.frame_stack))

    # Evaluation environment (single)
    eval_env = make_vec_env(env_id=lambda: None, n_envs=1, env_kwargs=None, vec_env_cls=DummyVecEnv,
                            env_fn=make_thunk)  # type: ignore[arg-type]
    eval_env = VecFrameStack(eval_env, n_stack=int(args.frame_stack))

    # Callbacks
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(eval_dir),
        eval_freq=max(10000 // n_envs, 1),
        deterministic=False,
        render=False,
        n_eval_episodes=3,
    )
    checkpoint_callback = CheckpointCallback(save_freq=max(50000 // n_envs, 1), save_path=str(run_dir), name_prefix="ckpt")

    policy = "CnnPolicy"
    model = PPO(
        policy,
        vec_env,
        verbose=1,
        batch_size=256,
        n_steps=128,
        n_epochs=4,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        tensorboard_log=str(run_dir / "tb"),
        device="auto",
    )

    print(f"[train] Starting training for {args.total_timesteps} timesteps, run_dir={run_dir}")
    model.learn(total_timesteps=int(args.total_timesteps), callback=[eval_callback, checkpoint_callback])

    final_path = run_dir / "final_model"
    model.save(str(final_path))
    print(f"[train] Saved model to: {final_path}.zip")

    # Attempt to save knowledge JSON and an initial guide
    kb_path = run_dir / "knowledge.json"
    if kb_path.exists():
        kb = KnowledgeBase.load(kb_path)
        kb.write_markdown_guide(run_dir / "guide.md")

    # Symlink/update latest
    latest = Path("runs/latest")
    try:
        if latest.exists() or latest.is_symlink():
            if latest.is_symlink() or latest.is_file():
                latest.unlink()
            else:
                shutil.rmtree(latest)
        latest.symlink_to(run_dir.resolve(), target_is_directory=True)
    except Exception:
        pass


def play(args: argparse.Namespace) -> None:
    ensure_packages()
    set_global_seeds(args.seed)

    run_dir = Path(args.run_dir or (Path("runs") / f"emerald_play_{now_stamp()}"))
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create single env with rendering if requested
    env = make_emerald_env(
        game=args.retro_game,
        state=args.state,
        scenario=args.scenario,
        seed=args.seed,
        run_dir=run_dir,
        frame_skip=args.frame_skip,
        intrinsic_beta=args.intrinsic_beta,
        include_diagonals=args.include_diagonals,
        render_mode="human" if args.render else None,
    )

    # Optionally load model
    model = None
    if args.model_path:
        if PPO is None:
            ensure_packages()
        print(f"[play] Loading model from {args.model_path}")
        model = PPO.load(args.model_path, device="auto")

    obs, info = env.reset()
    episode_reward = 0.0
    start_time = time.time()
    for step in range(int(args.max_steps)):
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += float(reward)
        if args.render and hasattr(env, "render"):
            try:
                env.render()
            except Exception:
                pass
        if terminated or truncated:
            print(f"[play] Episode finished at step {step}, reward={episode_reward:.2f}")
            episode_reward = 0.0
            obs, info = env.reset()
    elapsed = time.time() - start_time
    print(f"[play] Finished {args.max_steps} steps in {elapsed:.1f}s")


def guide(args: argparse.Namespace) -> None:
    kb_path = Path(args.knowledge_json)
    if not kb_path.exists():
        print(f"ERROR: knowledge JSON not found at {kb_path}")
        sys.exit(2)
    kb = KnowledgeBase.load(kb_path)
    out = Path(args.out)
    kb.write_markdown_guide(out)
    print(f"[guide] Wrote Markdown guide to {out}")


# ------------------------
# CLI
# ------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pokemon Emerald single-file RL agent")

    sub = p.add_subparsers(dest="cmd", required=True)

    # Shared args factory
    def add_env_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--retro-game", type=str, required=True, help="Gym Retro game id (from retro.list_games())")
        sp.add_argument("--state", type=str, default=None, help="Starting state (from retro.list_states(game))")
        sp.add_argument("--scenario", type=str, default=None, help="Optional scenario file name")
        sp.add_argument("--frame-skip", type=int, default=4, help="Action repeat / frame skip")
        sp.add_argument("--intrinsic-beta", type=float, default=0.2, help="Weight of intrinsic novelty reward")
        sp.add_argument("--include-diagonals", action="store_true", help="Include diagonal D-pad actions")
        sp.add_argument("--seed", type=int, default=1337, help="Random seed")
        sp.add_argument("--run-dir", type=str, default=None, help="Output run directory; defaults to runs/emerald_*")

    # Train
    sp_train = sub.add_parser("train", help="Train the agent with PPO")
    add_env_args(sp_train)
    sp_train.add_argument("--total-timesteps", type=int, default=1_000_000)
    sp_train.add_argument("--n-envs", type=int, default=4)
    sp_train.add_argument("--frame-stack", type=int, default=4)

    # Play
    sp_play = sub.add_parser("play", help="Run environment with (optional) trained model")
    add_env_args(sp_play)
    sp_play.add_argument("--model-path", type=str, default=None, help="Path to SB3 model .zip")
    sp_play.add_argument("--max-steps", type=int, default=10000)
    sp_play.add_argument("--render", action="store_true")

    # Guide
    sp_guide = sub.add_parser("guide", help="Generate a Markdown guide from knowledge.json")
    sp_guide.add_argument("--knowledge-json", type=str, required=True)
    sp_guide.add_argument("--out", type=str, default="guide.md")

    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.cmd == "train":
        train(args)
    elif args.cmd == "play":
        play(args)
    elif args.cmd == "guide":
        guide(args)
    else:
        raise ValueError(f"Unknown command {args.cmd}")


if __name__ == "__main__":
    main()
