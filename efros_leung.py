"""Independent teaching reproduction of Efros & Leung (1999).

Based on the paper and authors' alg.html, not copied from third-party code.
Grayscale, valid source windows, explicit known mask, logged pixel decisions.
"""
from pathlib import Path
import argparse, json, time
import numpy as np
from PIL import Image


def gaussian(window):
    if window < 3 or window % 2 != 1:
        raise ValueError('window must be an odd integer >= 3')
    axis = np.arange(window) - window // 2
    return np.exp(-(axis[:, None] ** 2 + axis[None, :] ** 2) / (2 * (window / 6.4) ** 2))


class Matcher:
    def __init__(self, source, window):
        self.source = np.asarray(source, dtype=np.float64)
        self.window, self.half = window, window // 2
        if min(self.source.shape) < window:
            raise ValueError('source smaller than window')
        self.patches = np.lib.stride_tricks.sliding_window_view(self.source, (window, window)).reshape(-1, window * window).copy()
        self.weights = gaussian(window).ravel()
        self.grid_shape = (self.source.shape[0] - window + 1, self.source.shape[1] - window + 1)

    def distances(self, template, valid):
        valid = np.asarray(valid, dtype=bool).ravel().copy()
        valid[self.window * self.half + self.half] = False
        ids = np.flatnonzero(valid)
        if not len(ids):
            raise ValueError('at least one known neighbor required')
        weights = self.weights[ids]
        diff = self.patches[:, ids] - np.asarray(template).ravel()[ids]
        return np.einsum('ij,j,ij->i', diff, weights, diff, optimize=False) / weights.sum()

    def candidates(self, template, valid, epsilon=0.1):
        distances = self.distances(template, valid)
        return np.flatnonzero(distances <= distances.min() * (1 + epsilon)), distances

    def coordinate(self, index):
        y, x = np.unravel_index(index, self.grid_shape)
        return int(y + self.half), int(x + self.half)


def neighborhood_counts(mask, window):
    half = window // 2
    padded = np.pad(mask.astype(np.int64), half)
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return integral[window:, window:] - integral[:-window, window:] - integral[window:, :-window] + integral[:-window, :-window]


def boundary(mask):
    padded = np.pad(mask, 1)
    adjacent = np.zeros(mask.shape, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            if dy != 1 or dx != 1:
                adjacent |= padded[dy:dy+mask.shape[0], dx:dx+mask.shape[1]]
    return adjacent & ~mask


def synthesize(source, shape=(96, 96), window=11, seed=7, initial=None, known=None, seed_position=None):
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    matcher = Matcher(source, window)
    half = window // 2
    if initial is None:
        output = np.zeros(shape, dtype=np.float64)
        mask = np.zeros(shape, dtype=bool)
        sy, sx = seed_position if seed_position is not None else (int(rng.integers(0, source.shape[0]-2)), int(rng.integers(0, source.shape[1]-2)))
        oy, ox = shape[0] // 2-1, shape[1] // 2-1
        output[oy:oy+3, ox:ox+3] = source[sy:sy+3, sx:sx+3]
        mask[oy:oy+3, ox:ox+3] = True
    else:
        output = np.asarray(initial, dtype=np.float64).copy()
        mask = np.asarray(known, dtype=bool).copy()
        if output.shape != mask.shape or output.shape != shape:
            raise ValueError('initial, known, and shape must agree')
        if not mask.any():
            raise ValueError('known region cannot be empty')
        output[~mask] = 0
        sy = sx = None
    initial_output, initial_mask = output.copy(), mask.copy()
    values = np.pad(output, half)
    valid = np.pad(mask, half)
    threshold = 0.3
    decisions, passes, relaxations = [], 0, 0
    while not mask.all():
        passes += 1
        coords = np.argwhere(boundary(mask))
        if not len(coords):
            raise RuntimeError('no boundary remains')
        rng.shuffle(coords)
        counts = neighborhood_counts(mask, window)
        coords = coords[np.argsort(-counts[coords[:, 0], coords[:, 1]], kind='stable')]
        progress = 0
        for y, x in coords:
            template = values[y:y+window, x:x+window]
            known_window = valid[y:y+window, x:x+window]
            candidates, distances = matcher.candidates(template, known_window)
            picked = int(rng.choice(candidates))
            if distances[picked] < threshold:
                py, px = matcher.coordinate(picked)
                value = float(source[py, px])
                values[y+half, x+half] = value
                valid[y+half, x+half] = True
                output[y, x], mask[y, x] = value, True
                decisions.append((int(y), int(x), value, py, px, float(distances.min()), float(distances[picked]), len(candidates), threshold))
                progress += 1
        if progress == 0:
            threshold *= 1.1
            relaxations += 1
        if passes > 10000:
            raise RuntimeError('pass safety limit exceeded')
    stats = {'window':window,'rng_seed':seed,'source_shape':list(source.shape),'output_shape':list(shape),'source_seed_top_left':[sy,sx],
             'epsilon':0.1,'sigma':window/6.4,'initial_max_error':0.3,'final_max_error':threshold,'source_boundary':'valid windows only',
             'boundary':'8-connected dilation; decreasing known count in full context; randomized stable ties',
             'passes':passes,'relaxations':relaxations,'committed_pixels':len(decisions),'runtime_seconds':time.perf_counter()-start}
    return output, np.asarray(decisions), initial_output, initial_mask, stats


def load_grayscale(path):
    return np.asarray(Image.open(path).convert('L'), dtype=np.float64) / 255


def save_run(directory, name, source, result):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    output, log, initial, known, stats = result
    np.savez_compressed(directory/f'{name}.npz',source=source,output=output,decisions=log,initial=initial,known=known)
    for suffix, array in [('output',output),('source',source),('initial',initial)]:
        Image.fromarray(np.rint(array*255).astype(np.uint8)).save(directory/f'{name}_{suffix}.png')
    (directory/f'{name}.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    return stats


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source');parser.add_argument('out_dir');parser.add_argument('--window',type=int,default=11)
    parser.add_argument('--size',type=int,default=96);parser.add_argument('--seed',type=int,default=7)
    args=parser.parse_args();src=load_grayscale(args.source)
    print(json.dumps(save_run(args.out_dir,'run',src,synthesize(src,(args.size,args.size),args.window,args.seed)),indent=2))
