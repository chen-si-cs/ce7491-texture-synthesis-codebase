# Recorded teaching experiments

These measurements came from the presentation preparation on 7 October 2026. They are single runs, not a performance benchmark. The repository preserves the same core algorithm and default setup. Wrapper paths and GIF rendering were adapted so the code works without the original presentation workspace.

All synthesis inputs are 64×64 grayscale crops, outputs are 96×96, and the 3×3 seed patch starts at source row/column (20,20). RNG seed is 7 unless noted. The ring input is generated in `run_experiments.py`, and is not the paper's exact ring image.

| Run | Window | CPU seconds |
|---|---:|---:|
| Synthetic rings | 5 | 2.41 |
| Synthetic rings | 11 | 25.63 |
| Synthetic rings | 15 | 31.40 |
| Synthetic rings | 23 | 77.16 |
| Rings, seed 19 | 11 | 20.05 |
| Rings, seed 31 | 11 | 15.22 |
| D1 | 11 | 15.09 |
| D1 | 23 | 40.31 |
| Bread | 11 | 14.59 |
| Bread hole filling | 11 | 0.97 |

The hole experiment uses a 99×108 target with missing rows 50:78 and columns 65:93, totaling 784 pixels. Its separate 45×60 source is `bread[:45,:60]`, so no hidden target pixels enter the exemplar. Known target pixels were unchanged. None of the ten runs needed acceptance-threshold relaxation.

The recorded environment was Python 3.12.14, NumPy 2.3.5, Pillow 12.3.0, and Windows 11 on an Intel CPU (family 6, model 186). No GPU was used. Actual runtimes vary with hardware, software, and competing workloads.

Observed results: w=5 rings fragment, intermediate windows recover rings with some broken motifs, and w=23 gives a more regular layout on this input. Different seeds at w=11 change the layout. D1 retains local edges but global alignment is imperfect, including at w=23. Bread hole filling looks locally plausible without recovering a unique hidden truth.

Checks verify tested algorithm behavior, not perceptual quality or exact equivalence to the authors' original code. The original seeds and full protocol needed for exact figure replication are unavailable. The explicit valid-source-window policy is also an implementation choice.

The README's images show only our synthetic ring runs. Author exemplars and their derived outputs are generated locally when you run the optional suite.
