"""Render a captioned GIF of a controller playing held-out layouts, headless.

Each listed seed is played once from reset and the frames are concatenated.
The caption on every frame names the controller and the layout seed, and the
final frame of each episode is held briefly so its outcome is visible.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import _bootstrap  # noqa: E402, F401  (adds the repository root to sys.path)
from firewater.firewater_env import FireWaterEnv  # noqa: E402
from firewater.generalized_planner import plan_level  # noqa: E402
from firewater.safety_shield import shield_action  # noqa: E402
from firewater.temple_env import TempleEnv  # noqa: E402
from firewater.temple_expert import TempleExpert  # noqa: E402

CONTROLLERS = (
    "generalist",
    "generalist-shield",
    "planner",
    "temple-policy",
    "temple-expert",
)


def load_model(path: Path | None):
    if path is None:
        raise SystemExit("this controller needs --model")
    from stable_baselines3 import PPO

    return PPO.load(path, device="cpu")


def play_generalist(seed: int, controller: str, model) -> tuple[list, str]:
    env = FireWaterEnv(
        render_mode="rgb_array",
        procedural=True,
        level_seed=seed,
        observation_mode="generalized",
    )
    observation, _ = env.reset(seed=0)
    actions = None
    if controller == "planner":
        actions = list(plan_level(env).actions)
        observation, _ = env.reset(options={"level_seed": seed})
    frames = [env.render()]
    info = {"reason": None}
    for step in range(env.max_steps):
        if actions is not None:
            action = actions[step]
        else:
            action, _ = model.predict(observation, deterministic=True)
            if controller == "generalist-shield":
                action = shield_action(env, action)
        observation, _, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            break
    env.close()
    return frames, str(info["reason"])


def play_temple(seed: int, controller: str, model) -> tuple[list, str]:
    mode = "egocentric"
    if model is not None:
        mode = TempleEnv.mode_for_size(model.observation_space.shape[0])
    env = TempleEnv(render_mode="rgb_array", level_seed=seed, observation_mode=mode)
    observation, _ = env.reset(seed=0)
    expert = TempleExpert()
    frames = [env.render()]
    info = {"reason": None}
    for _ in range(env.max_steps):
        if controller == "temple-expert":
            action = expert.action(env)
        else:
            action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            break
    env.close()
    return frames, str(info["reason"])


def caption_frame(frame: np.ndarray, text: str, width: int, font) -> Image.Image:
    image = Image.fromarray(frame)
    height = round(image.height * width / image.width)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    band = max(22, font.size + 10)
    canvas = Image.new("RGB", (width, height + band), (16, 16, 20))
    canvas.paste(image, (0, band))
    ImageDraw.Draw(canvas).text(
        (8, (band - font.size) // 2), text, font=font, fill=(235, 235, 235)
    )
    return canvas


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("controller", choices=CONTROLLERS)
    parser.add_argument("--label", required=True, help="caption controller name")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[100_000])
    parser.add_argument("--every", type=int, default=1, help="keep every Nth frame")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--colors", type=int, default=96)
    parser.add_argument("--hold", type=float, default=1.0, help="seconds on last frame")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.every < 1 or args.fps <= 0 or args.width < 64:
        raise SystemExit("--every must be >= 1, --fps > 0, --width >= 64")
    model = None
    if args.controller in ("generalist", "generalist-shield", "temple-policy"):
        model = load_model(args.model)
    play = play_temple if args.controller.startswith("temple") else play_generalist
    font = ImageFont.load_default(size=max(12, args.width // 34))

    images: list[Image.Image] = []
    for seed in args.seeds:
        frames, reason = play(seed, args.controller, model)
        kept = frames[:: args.every]
        if kept[-1] is not frames[-1]:
            kept.append(frames[-1])
        text = f"{args.label} | held-out seed {seed}"
        episode = [caption_frame(frame, text, args.width, font) for frame in kept]
        final = caption_frame(frames[-1], f"{text} | {reason}", args.width, font)
        episode.extend([final] * max(1, round(args.hold * args.fps)))
        images.extend(episode)
        print(f"seed={seed} reason={reason} frames={len(frames)} kept={len(episode)}")

    palette = images[len(images) // 2].quantize(
        colors=args.colors, method=Image.Quantize.MEDIANCUT
    )
    paletted = [
        image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paletted[0].save(
        args.output,
        save_all=True,
        append_images=paletted[1:],
        duration=round(1000 / args.fps),
        loop=0,
        optimize=True,
    )
    seconds = len(images) / args.fps
    size_mb = args.output.stat().st_size / 1e6
    print(f"wrote {args.output}: {len(images)} frames, {seconds:.1f} s, {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
