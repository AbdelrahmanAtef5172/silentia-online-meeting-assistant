# Just script to concatenate multiple videos into one, resizing them to a target size and fps.
import sys, os
import cv2
import numpy as np

def concatenate_videos(input_paths, output_path, target_size=(1280, 720), target_fps=30.0, gap_frames=0):
    clips = []
    for path in input_paths:
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if (w, h) != target_size:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
            frames.append(frame)
        cap.release()
        if not frames:
            print(f'Warning: {path} has no frames, skipping')
            continue
        clips.append(frames)
        print(f'{os.path.basename(path)}: {len(frames)} frames')

    black = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8) if gap_frames > 0 else None

    fourcc_map = {'.mp4': 'mp4v', '.avi': 'MJPG', '.mov': 'mp4v', '.mkv': 'X264'}
    ext = os.path.splitext(output_path)[1].lower()
    fourcc_code = fourcc_map.get(ext, 'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*fourcc_code.ljust(4))
    out = cv2.VideoWriter(output_path, fourcc, target_fps, target_size)
    if not out.isOpened():
        print('ERROR: Could not open VideoWriter')
        return

    for i, frames in enumerate(clips):
        for f in frames:
            out.write(f)
        if gap_frames > 0 and i < len(clips) - 1:
            for _ in range(gap_frames):
                out.write(black)
    out.release()

    total = sum(len(c) for c in clips) + gap_frames * max(0, len(clips) - 1)
    print(f'\nWritten {total} frames to {output_path}')
    return output_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Concatenate sign language test videos')
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--gap-frames', type=int, default=0)
    args = parser.parse_args()

    concatenate_videos(
        args.inputs, args.output,
        target_size=(args.width, args.height),
        target_fps=args.fps,
        gap_frames=args.gap_frames,
    )
