#!/usr/bin/env python3
"""End-to-end WebSocket client for testing realtime video streaming.

Usage:
    # Interactive mode: type prompts while video generates continuously
    python tests/test_realtime_e2e_client.py -i

    # Batch mode with fixed block count:
    python tests/test_realtime_e2e_client.py --blocks 5

    # Batch mode with scheduled prompt change:
    python tests/test_realtime_e2e_client.py --blocks 8 --change-prompt-at 3 \
        --new-prompt "A rocket launching into space"

    # Save output frames to disk:
    python tests/test_realtime_e2e_client.py -i --save-dir /tmp/frames

    # V2V mode:
    python tests/test_realtime_e2e_client.py --mode v2v --blocks 5

Requires: pip install websockets msgpack Pillow
"""

import argparse
import asyncio
import io
import os
import sys
import time
import threading

try:
    import msgpack
except ImportError:
    msgpack = None

try:
    import websockets
except ImportError:
    websockets = None

try:
    from PIL import Image
except ImportError:
    Image = None


def pack_msg(msg: dict) -> bytes:
    if msgpack is not None:
        return msgpack.packb(msg, use_bin_type=True)
    import json
    return json.dumps(msg).encode()


def unpack_msg(data: bytes) -> dict:
    if msgpack is not None:
        return msgpack.unpackb(data, raw=False)
    import json
    return json.loads(data)


async def run_t2v_interactive(args):
    """Interactive T2V: generates continuously, user types prompts to steer."""
    uri = f"ws://{args.host}:{args.port}/v1/realtime_video/generate"
    print(f"Connecting to {uri} ...")

    async with websockets.connect(uri, max_size=50 * 1024 * 1024) as ws:
        config = {
            "type": "config",
            "mode": "t2v",
            "prompt": args.prompt,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.steps,
        }
        await ws.send(pack_msg(config))
        print(f"Config: prompt='{args.prompt}', "
              f"{args.width}x{args.height}, steps={args.steps}")

        data = await ws.recv()
        msg = unpack_msg(data)
        if not (msg.get("type") == "status"
                and msg.get("content") == "session_started"):
            print(f"Unexpected response: {msg}")
            return
        print("Session started!")
        print("-" * 60)
        print("Type a new prompt and press Enter to change the scene.")
        print("Type 'quit' or 'exit' to stop.")
        print("-" * 60)

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)

        # Use a thread to read stdin, push lines into a thread-safe queue
        input_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        quit_flag = False

        def _read_stdin():
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                loop.call_soon_threadsafe(input_queue.put_nowait, line.strip())

        stdin_thread = threading.Thread(target=_read_stdin, daemon=True)
        stdin_thread.start()

        frame_count = 0
        start_time = time.time()
        current_prompt = args.prompt

        try:
            while True:
                # Check for new prompts (non-blocking)
                while not input_queue.empty():
                    line = input_queue.get_nowait()
                    if not line:
                        continue
                    if line.lower() in ("quit", "exit", "q"):
                        quit_flag = True
                        break

                    prompt_msg = {"type": "prompt", "content": line}
                    await ws.send(pack_msg(prompt_msg))
                    current_prompt = line
                    print(f">>> Prompt updated: '{line}'")

                if quit_flag:
                    break

                # Receive next frame
                try:
                    data = await asyncio.wait_for(ws.recv(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                msg = unpack_msg(data)
                elapsed = time.time() - start_time

                if msg["type"] == "frame":
                    content = msg["content"]
                    frame_count += 1

                    parts = [f"Block {frame_count}",
                             f"{len(content)} bytes",
                             f"{elapsed:.1f}s"]

                    if Image is not None and isinstance(content, bytes):
                        try:
                            img = Image.open(io.BytesIO(content))
                            parts.append(f"{img.size[0]}x{img.size[1]}")

                            if args.save_dir:
                                path = os.path.join(
                                    args.save_dir,
                                    f"frame_{frame_count:04d}.png",
                                )
                                img.save(path)
                                parts.append(f"saved:{path}")
                        except Exception as e:
                            parts.append(f"decode err: {e}")

                    print(f"  {' | '.join(parts)}")

                elif msg["type"] == "error":
                    print(f"ERROR: {msg['content']}")
                    break

                elif msg["type"] == "status":
                    print(f"Status: {msg['content']}")

        except (KeyboardInterrupt,):
            print("\nInterrupted.")
        except Exception as e:
            if "close" not in str(e).lower():
                print(f"\nConnection error: {e}")

        total = time.time() - start_time
        if frame_count > 0:
            print(f"\nDone! {frame_count} blocks in {total:.1f}s "
                  f"({frame_count / total:.2f} blocks/sec)")


async def run_t2v_batch(args):
    """Batch T2V: generate a fixed number of blocks, optionally change prompt."""
    uri = f"ws://{args.host}:{args.port}/v1/realtime_video/generate"
    print(f"Connecting to {uri} ...")

    async with websockets.connect(uri, max_size=50 * 1024 * 1024) as ws:
        config = {
            "type": "config",
            "mode": "t2v",
            "prompt": args.prompt,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.steps,
        }
        await ws.send(pack_msg(config))
        print(f"Sent config: mode=t2v, prompt='{args.prompt}', "
              f"{args.width}x{args.height}, steps={args.steps}")

        data = await ws.recv()
        msg = unpack_msg(data)
        assert msg["type"] == "status" and msg["content"] == "session_started", \
            f"Expected session_started, got: {msg}"
        print("Session started!")

        frame_count = 0
        start_time = time.time()

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)

        prompt_changed = False

        while frame_count < args.blocks:
            if (args.change_prompt_at
                    and frame_count == args.change_prompt_at
                    and not prompt_changed):
                new_prompt = args.new_prompt or \
                    "A rocket launching into space with flames"
                prompt_msg = {"type": "prompt", "content": new_prompt}
                await ws.send(pack_msg(prompt_msg))
                print(f"\n>>> Prompt changed to: '{new_prompt}'")
                prompt_changed = True

            data = await ws.recv()
            msg = unpack_msg(data)
            elapsed = time.time() - start_time

            if msg["type"] == "frame":
                content = msg["content"]
                frame_count += 1

                info = f"Block {frame_count}/{args.blocks} | "
                info += f"{len(content)} bytes | "
                info += f"elapsed: {elapsed:.1f}s"

                if Image is not None and isinstance(content, bytes):
                    try:
                        img = Image.open(io.BytesIO(content))
                        info += f" | {img.size[0]}x{img.size[1]} {img.mode}"

                        if args.save_dir:
                            path = os.path.join(
                                args.save_dir, f"frame_{frame_count:04d}.png"
                            )
                            img.save(path)
                            info += f" | saved: {path}"
                    except Exception as e:
                        info += f" | decode err: {e}"

                print(info)

            elif msg["type"] == "error":
                print(f"ERROR from server: {msg['content']}")
                break

            elif msg["type"] == "status":
                print(f"Status: {msg['content']}")

        total = time.time() - start_time
        print(f"\nDone! {frame_count} blocks in {total:.1f}s "
              f"({frame_count / total:.2f} blocks/sec)")


async def run_v2v_test(args):
    """V2V mode: send video frames, receive transformed output."""
    uri = f"ws://{args.host}:{args.port}/v1/realtime_video/generate"
    print(f"Connecting to {uri} (V2V mode) ...")

    if Image is None:
        print("ERROR: Pillow is required for V2V test")
        return

    async with websockets.connect(uri, max_size=50 * 1024 * 1024) as ws:
        config = {
            "type": "config",
            "mode": "v2v",
            "prompt": args.prompt,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.steps,
        }
        await ws.send(pack_msg(config))
        print(f"Sent config: mode=v2v, prompt='{args.prompt}'")

        data = await ws.recv()
        msg = unpack_msg(data)
        assert msg["type"] == "status" and msg["content"] == "session_started"
        print("Session started!")

        import numpy as np
        num_input_frames = 12
        print(f"Sending {num_input_frames} dummy input frames...")

        frames_bytes = []
        for i in range(num_input_frames):
            arr = np.random.randint(
                0, 255, (args.height, args.width, 3), dtype=np.uint8
            )
            img = Image.fromarray(arr)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            frames_bytes.append(buf.getvalue())

        video_msg = {"type": "video", "frames": frames_bytes}
        await ws.send(pack_msg(video_msg))
        print(f"Sent {num_input_frames} frames")

        frame_count = 0
        start = time.time()

        while frame_count < args.blocks:
            if frame_count > 0:
                more_frames = []
                for _ in range(12):
                    arr = np.random.randint(
                        0, 255, (args.height, args.width, 3), dtype=np.uint8
                    )
                    img = Image.fromarray(arr)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=80)
                    more_frames.append(buf.getvalue())
                await ws.send(
                    pack_msg({"type": "video", "frames": more_frames})
                )

            data = await ws.recv()
            msg = unpack_msg(data)

            if msg["type"] == "frame":
                frame_count += 1
                content = msg["content"]
                elapsed = time.time() - start
                print(f"V2V Block {frame_count}/{args.blocks} | "
                      f"{len(content)} bytes | {elapsed:.1f}s")

                if args.save_dir and Image is not None:
                    try:
                        img = Image.open(io.BytesIO(content))
                        path = os.path.join(
                            args.save_dir, f"v2v_frame_{frame_count:04d}.png"
                        )
                        img.save(path)
                    except Exception:
                        pass

            elif msg["type"] == "error":
                print(f"ERROR: {msg['content']}")
                break

        total = time.time() - start
        print(f"\nV2V done! {frame_count} blocks in {total:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Realtime video E2E test client"
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode", choices=["t2v", "v2v"], default="t2v")
    parser.add_argument(
        "--prompt",
        default="A beautiful sunset over the ocean, cinematic",
    )
    parser.add_argument(
        "--blocks", type=int, default=None,
        help="Number of blocks (default: unlimited in interactive, 5 in batch)",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--steps", type=int, default=4,
                        help="Denoising steps per block")
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Interactive mode: type prompts while video generates continuously",
    )
    parser.add_argument("--change-prompt-at", type=int, default=None,
                        help="[batch] Change prompt at this block number")
    parser.add_argument("--new-prompt", default=None,
                        help="[batch] New prompt for --change-prompt-at")
    parser.add_argument("--save-dir", default=None,
                        help="Directory to save output frames as PNG")

    args = parser.parse_args()

    if websockets is None:
        print("ERROR: pip install websockets")
        return
    if msgpack is None:
        print("WARNING: msgpack not installed, using JSON fallback")

    if args.mode == "v2v":
        if args.blocks is None:
            args.blocks = 5
        asyncio.run(run_v2v_test(args))
    elif args.interactive:
        asyncio.run(run_t2v_interactive(args))
    else:
        if args.blocks is None:
            args.blocks = 5
        asyncio.run(run_t2v_batch(args))


if __name__ == "__main__":
    main()
