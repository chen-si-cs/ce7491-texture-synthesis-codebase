"""Render a recorded synthesis NPZ as a portable GIF, using NumPy and Pillow."""
from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def raster(values, known=None, size=320):
    gray = np.rint(np.clip(values, 0, 1) * 255).astype('uint8')
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    if known is not None:
        rgb[~known] = [222, 231, 239]
    image = Image.fromarray(rgb)
    scale = size / max(image.size)
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.NEAREST)


def make_gif(run_path, output, frame_count=120, seconds=20):
    if frame_count < 2 or seconds <= 0:
        raise ValueError('frames must be at least 2 and seconds must be positive')
    with np.load(run_path, allow_pickle=False) as run:
        source = run['source'].copy()
        state, known, log = run['initial'].copy(), run['known'].copy(), run['decisions'].copy()
    font = ImageFont.load_default(size=22)
    frames, applied = [], 0
    for target in np.linspace(0, len(log), frame_count).astype(int):
        for row in log[applied:target]:
            y, x, value = row[:3]
            state[int(y), int(x)], known[int(y), int(x)] = value, True
        applied = target
        image = Image.new('RGB', (760, 430), 'white')
        draw = ImageDraw.Draw(image)
        draw.text((20, 15), 'Exemplar', font=font, fill='#073F73')
        draw.text((405, 15), 'Recorded growth', font=font, fill='#073F73')
        image.paste(raster(source), (20, 55))
        image.paste(raster(state, known), (405, 55))
        draw.text((20, 388), f'{target:,} / {len(log):,} pixels. Normalized progress.', font=font, fill='#073F73')
        frames.append(image)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = max(10, round(seconds * 1000 / frame_count / 10) * 10)
    durations = [duration] * frame_count
    durations[-1] += 2000
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True, disposal=2)
    with Image.open(output) as check:
        elapsed = 0
        for index in range(check.n_frames):
            check.seek(index)
            elapsed += check.info.get('duration', 0)
        summary = {'file': output.name, 'frames': check.n_frames, 'seconds': elapsed / 1000,
                   'source': Path(run_path).name, 'timing': 'normalized commit progress plus final hold'}
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, default=Path('output/growth.gif'))
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--seconds', type=float, default=20)
    args = parser.parse_args()
    make_gif(args.run, args.output, args.frames, args.seconds)
