# `rsliding` package

This python package contains utilities to compute a sliding sigma clipping, where the kernel can contain weights and the data can contain NaN values (has to be float64).

The actual core code is in Rust.
This package was created to have a less memory hungry sigma clipping code compared to the similar
`sliding` python package (cf. https://github.com/avoyeux/sliding.git). It is also a few times faster than the `sliding` package equivalent (except in some cases when using the Convolution or SlidingMean class).
Check the **Functions** markdown section to know about the different available classes.

For high numerical stability (c.f. https://dl.acm.org/doi/epdf/10.1145/359146.359152), the standard deviation is computed using the two-pass algorithm (i.e. mean computation then variance). Furthermore, the user can decide to use 'Neumaier's summation' for the standard deviation and mean computation (highest numerical stability that I know of). While the Welford algorithm is faster and quite stable, I did find (from literature and tests) that the two-pass algorithm is more stable, even more so when using Neumaier's summation.

**IMPORTANT:** the code only works if the kernel dimensions are odd and has the same dimensionality than the input data.

**NOTES**: for the standard deviation computation, compared to using numpy.std on each window, the ``rsliding`` implementation should be a little less numerically stable if 'neumaier=False'. This is because, while numpy.std seem to also use the two-pass algorithm (c.f. https://numpy.org/doc/stable/reference/generated/numpy.std.html), numpy functions also implement pairwise summation (not done in ``rsliding``). That being said, Neumaier's summation is more stable than pairwise summation and, as such, when 'neumaier=True`, the 'SlidingMean' and 'SlidingStandardDeviation' implementation should be more numerically stable than implementing sliding operations using np.mean and np.std (or nanmean/nanstd).

## Install package

Given that pre-compiled binaries are needed if the user doesn't have the Rust compiler installed, the user should install the package through *PyPi*.

#### (**OPTIONAL**) Create and activate a python virtual environnement:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

or on Windows OS:

```bash
python -m venv .venv

# Command Prompt
.venv\Scripts\activate

# PowerShell
.venv\Scripts\Activate.ps1

# Git Bash or WSL
source .venv/Scripts/activate
```

#### Install package in virtual environnement (or on bare-metal - wouldn't recommend):

```bash
pip install --upgrade pip
pip install rsliding
```

## Functions

The `rsliding` package has 6 different classes:
- **Padding** which returns the padded data. Not really useful given np.pad is way more efficient. A Python binding exist just so that the user can check the results if wanted be.
- **Convolution** which lets you perform a convolution (NaN handling done).
- **SlidingMean** which performs a sliding mean (NaN handling done).
- **SlidingMedian** which performs a sliding median (NaN handling done).
- **SlidingStandardDeviation** which performs a sliding standard deviation (NaN handling done).
- **SlidingSigmaClipping** which performs a sliding sigma clipping (NaN handling done).

#### Example
```python
# IMPORTs
import numpy as np
from rsliding import SlidingSigmaClipping

# CREATE fake data
fake_data = np.random.rand(36, 1024, 128).astype(np.float64)
fake_data[10:15, 100:200, 50:75] = 1.3
fake_data[7:, 40:60, 70:] = 1.7

# KERNEL
kernel = np.ones((5, 3, 7), dtype=np.float64)
kernel[2, 1, 3] = 0.

# NaN ~5%
is_nan = np.random.rand(*fake_data.shape) < 0.05
fake_data[is_nan] = np.nan

# CLIPPING no lower value
clipped = SlidingSigmaClipping(
    data=fake_data,
    kernel=kernel,
    center_choice='median',
    sigma=3.,
    sigma_lower=None,
    max_iters=3,
    borders='reflect',
    threads=1,
    masked_array=False,
    neumaier=True,
).clipped
```

## IMPORTANT

Before using this package some information is needed:

- **float64** values for the data: the data needs to be of float64 type. Given the default class argument values, the data will be cast to float64 before calling Rust.
- **threads**: there is a threads argument for most of the classes. It is not used during the padding operations but used for in all other intensive operations. Done like so as using threads for the padding would most likely slow down the computation speed in most cases.
