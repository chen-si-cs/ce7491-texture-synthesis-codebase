"""Run the teaching experiment suite without any presentation-folder dependency."""
from pathlib import Path
import argparse
import json
import platform
import sys
import numpy as np
from PIL import Image, __version__ as pillow_version
from efros_leung import load_grayscale, synthesize, save_run

ROOT = Path(__file__).resolve().parent


def synthetic_rings():
    y, x = np.mgrid[:64, :64]
    rings = np.zeros((64, 64), dtype=float)
    for cy in [8, 24, 40, 56]:
        for cx in [8, 24, 40, 56]:
            radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
            rings[(radius >= 3.2) & (radius <= 4.5)] = 1
    return rings


def run(args):
    if args.size < 3:
        raise ValueError('output size must be at least 3')
    if any(w < 3 or w % 2 == 0 or w > 64 for w in args.windows):
        raise ValueError('windows must be odd integers between 3 and 63')
    # Check optional data before beginning the expensive suite.
    if not args.synthetic_only:
        missing = [args.data / f'{n}.gif' for n in ['D1', 'bread']
                   if not (args.data / f'{n}.gif').is_file()]
        if missing:
            raise FileNotFoundError('Run python download_examples.py first, or use --synthetic-only')
    args.out.mkdir(parents=True, exist_ok=True)
    rings = synthetic_rings()
    Image.fromarray((rings * 255).astype('uint8')).save(args.out / 'synthetic_rings_input.png')
    jobs = [(f'rings_w{w}', rings, w, 7) for w in args.windows]
    if 11 in args.windows:
        jobs += [('rings_seed19', rings, 11, 19), ('rings_seed31', rings, 11, 31)]
    if not args.synthetic_only:
        for name in ['D1', 'bread']:
            jobs.append((name.lower() + '_w11', load_grayscale(args.data / f'{name}.gif')[:64, :64], 11, 7))
        jobs.append(('d1_w23', load_grayscale(args.data / 'D1.gif')[:64, :64], 23, 7))
    manifest = []
    for name, source, window, seed in jobs:
        print('START', name, flush=True)
        result = synthesize(source, (args.size, args.size), window, seed, seed_position=(20, 20))
        stats = save_run(args.out, name, source, result)
        stats['name'] = name
        manifest.append(stats)
        print(f'DONE {name}: {stats["runtime_seconds"]:.2f} seconds', flush=True)
        (args.out / 'experiment_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if not args.synthetic_only:
        bread = load_grayscale(args.data / 'bread.gif')
        if bread.shape[0] < 78 or bread.shape[1] < 93:
            raise ValueError('bread exemplar is too small for the documented hole experiment')
        known = np.ones(bread.shape, dtype=bool)
        known[50:78, 65:93] = False
        source = bread[:45, :60]
        result = synthesize(source, bread.shape, 11, 7, initial=bread, known=known)
        stats = save_run(args.out, 'bread_hole', source, result)
        stats.update(name='bread_hole', known_region_unchanged=bool(np.array_equal(result[0][known], bread[known])))
        manifest.append(stats)
        print(f'DONE bread_hole: {stats["runtime_seconds"]:.2f} seconds', flush=True)
        (args.out / 'experiment_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    environment = {'python': sys.version, 'numpy': np.__version__, 'pillow': pillow_version,
                   'platform': platform.platform(), 'processor': platform.processor(), 'gpu_used': False}
    (args.out / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic-only', action='store_true', help='run without downloading images')
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--out', type=Path, default=ROOT / 'output/experiments')
    parser.add_argument('--size', type=int, default=96)
    parser.add_argument('--windows', type=int, nargs='+', default=[5, 11, 15, 23])
    run(parser.parse_args())
