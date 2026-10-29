#!/usr/bin/env python
"""Convert the LPS real-world Franka (DROID-format) HDF5 trajectories into a LeRobot dataset.

The four released datasets live on the Hub as raw DROID HDF5 under ``success/`` and ``failure/``:

    jellyho/droid_pnp_carrot   jellyho/droid_eggplant_bin
    jellyho/droid_plug_in_bulb jellyho/droid_refill_tape

What this writes, per step, follows the paper's real-robot setup: a 7-D proprioceptive state
(end-effector pose + gripper), a 7-D action (delta end-effector velocity + gripper), and the two
ZED2 views at 224x224.

Reward. Offline RL needs a scalar, and the demonstrations only carry a per-trajectory success flag,
so the written reward is -1 on every step and 0 on the final step of a successful trajectory; a
failure never reaches 0. ``load_lerobot_dataset`` re-derives success from this reward and then
rebuilds the reward on the horizon-aware convention of ``assemble_dataset``.

Writing a LeRobot dataset needs the ``lerobot`` package, so run this in an environment that has it
(reading the result back into JAX does not -- see ``envs/robot_datasets.load_lerobot_dataset``):

    python tools/droid_to_lerobot.py --repo_id jellyho/droid_pnp_carrot \
        --out_repo_id jellyho/droid_pnp_carrot_lerobot --root /path/to/lerobot/droid_pnp_carrot \
        --task "pick up the carrot and place it in the bowl"
"""
import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np

# wrist cam shows the gripper fingers; exterior cam sees the whole arm. Same serials in all four
# released datasets, but they are flags so a re-record with different hardware still converts.
WRIST_CAM = "14013996_left"
EXTERIOR_CAM = "36776608_left"
FPS = 10  # control_start timestamps are ~103 ms apart


def features(image_shape):
    return {
        "observation.state": {"dtype": "float32", "shape": (7,),
                              "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"]},
        "observation.images.exterior": {"dtype": "video", "shape": image_shape,
                                        "names": ["height", "width", "channels"]},
        "observation.images.wrist": {"dtype": "video", "shape": image_shape,
                                     "names": ["height", "width", "channels"]},
        "action": {"dtype": "float32", "shape": (7,),
                   "names": ["vx", "vy", "vz", "wx", "wy", "wz", "gripper"]},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }


def read_trajectory(path, wrist, exterior):
    """One HDF5 file -> dict of per-step arrays. Returns None if a required field is missing."""
    with h5py.File(path, "r") as f:
        try:
            ee = np.asarray(f["observation/robot_state/cartesian_position"], np.float32)
            grip = np.asarray(f["observation/robot_state/gripper_position"], np.float32)
            vel = np.asarray(f["action/cartesian_velocity"], np.float32)
            agrip = np.asarray(f["action/gripper_position"], np.float32)
            # DROID records BGRA; drop the alpha and write RGB, so the dataset reads the way it
            # looks.
            # h5py cannot take a negative stride, so drop the alpha on read and reverse after.
            img_w = np.asarray(f[f"observation/image/{wrist}"][:, :, :, :3])[..., ::-1]
            img_e = np.asarray(f[f"observation/image/{exterior}"][:, :, :, :3])[..., ::-1]
        except KeyError as e:
            return None, f"missing {e}"
        success = bool(f.attrs.get("success", False))
        task = str(f.attrs.get("current_task", "") or "")

    n = min(len(ee), len(grip), len(vel), len(agrip), len(img_w), len(img_e))
    if n < 2:
        return None, f"too short ({n})"
    state = np.concatenate([ee[:n], grip[:n, None]], axis=1)
    action = np.concatenate([vel[:n], agrip[:n, None]], axis=1)
    reward = np.full((n, 1), -1.0, np.float32)
    if success:
        reward[-1, 0] = 0.0
    done = np.zeros((n, 1), bool)
    done[-1, 0] = True
    return dict(state=state, action=action, wrist=img_w[:n], exterior=img_e[:n],
                reward=reward, done=done, success=success, task=task), None


def source_files(args):
    """Local directory, or pull the Hub repo first. Returns (path, is_success) pairs."""
    if args.src:
        root = Path(args.src)
    else:
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(args.repo_id, repo_type="dataset",
                                      local_dir=args.cache or None))
    out = []
    for split, ok in (("success", True), ("failure", False)):
        d = root / split
        if d.is_dir():
            out += [(p, ok) for p in sorted(d.glob("*.h5"), key=lambda q: _num(q.name))]
    return out


def _num(name):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", help="source Hub dataset, e.g. jellyho/droid_pnp_carrot")
    ap.add_argument("--src", help="local directory holding success/ and failure/ instead")
    ap.add_argument("--out_repo_id", required=True, help="repo id to give the LeRobot dataset")
    ap.add_argument("--root", required=True, help="where to write the LeRobot dataset")
    ap.add_argument("--cache", help="where to download the raw HDF5 (defaults to the HF cache)")
    ap.add_argument("--task", default="", help="language instruction; defaults to the HDF5 attr")
    ap.add_argument("--wrist_cam", default=WRIST_CAM)
    ap.add_argument("--exterior_cam", default=EXTERIOR_CAM)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--include_failures", action="store_true",
                    help="also convert the failure/ folder; the offline datasets are successes only")
    ap.add_argument("--limit", type=int, help="convert at most this many trajectories (smoke test)")
    ap.add_argument("--dry_run", action="store_true", help="read and validate, write nothing")
    args = ap.parse_args()
    if not args.repo_id and not args.src:
        ap.error("one of --repo_id or --src is required")

    files = source_files(args)
    if not args.include_failures:
        files = [(p, ok) for p, ok in files if ok]
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} trajectories "
          f"({sum(1 for _, ok in files if ok)} success / {sum(1 for _, ok in files if not ok)} failure)")

    ds = None
    kept = skipped = steps = 0
    for path, _ in files:
        traj, why = read_trajectory(path, args.wrist_cam, args.exterior_cam)
        if traj is None:
            print(f"  skip {path.name}: {why}")
            skipped += 1
            continue
        if ds is None and not args.dry_run:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            ds = LeRobotDataset.create(
                repo_id=args.out_repo_id, fps=args.fps,
                features=features(traj["wrist"].shape[1:]),
                root=args.root, robot_type="franka",
            )
        task = args.task or traj["task"] or "manipulate"
        for i in range(len(traj["state"])):
            frame = {
                "observation.state": traj["state"][i],
                "observation.images.exterior": traj["exterior"][i],
                "observation.images.wrist": traj["wrist"][i],
                "action": traj["action"][i],
                "next.reward": traj["reward"][i],
                "next.done": traj["done"][i],
                "task": task,
            }
            if not args.dry_run:
                ds.add_frame(frame)
        if not args.dry_run:
            ds.save_episode()
        kept += 1
        steps += len(traj["state"])
        print(f"  {path.parent.name}/{path.name:26} T={len(traj['state']):4} "
              f"success={traj['success']} task={task!r}")

    print(f"\n{kept} episodes, {steps} steps, {skipped} skipped")
    if not args.dry_run:
        info_path = Path(args.root) / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        info["channel_order"] = "rgb"
        info_path.write_text(json.dumps(info, indent=4))
        print(f"written to {args.root}")


if __name__ == "__main__":
    main()
