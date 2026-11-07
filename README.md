# Learning

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C)](https://matplotlib.org/)
[![ipywidgets](https://img.shields.io/badge/ipywidgets-interactive-4D9DE0)](https://ipywidgets.readthedocs.io/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sfarrens/learning/main)

> Author: [Samuel Farrens](https://sfarrens.github.io)  
> Email: [samuel.farrens@cea.fr](mailto:samuel.farrens@cea.fr)

Interactive Jupyter notebooks for building an intuition for core concepts in data analysis. Each notebook combines short explanations, the maths written out in plain NumPy, and interactive widgets to explore the ideas hands-on.

## Contents

### Signal Processing

| Notebook | Description | Launch |
|:--|:--|:--|
| [The Fourier Transform and Fourier Space](fourier_transform.ipynb) | The continuous Fourier transform, the DFT and the FFT, with 1D (time-frequency) and 2D (space-spatial frequency) examples, filtering, sparsity and incomplete Fourier sampling. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sfarrens/learning/blob/main/fourier_transform.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sfarrens/learning/main?labpath=fourier_transform.ipynb) |

## Getting Started

Clone the repository and install the requirements:

```bash
git clone <repository-url>
cd learning
pip install -r requirements.txt
```

Then launch JupyterLab:

```bash
jupyter lab
```

The notebooks also run in VS Code and Google Colab. Keep each notebook's helper module (*e.g.* [`ft_plot.py`](ft_plot.py)) in the same directory as the notebook.

### Interactive figures

The interactive cells use [ipywidgets](https://ipywidgets.readthedocs.io/) with the [ipympl](https://matplotlib.org/ipympl/) backend (`%matplotlib widget`), which updates figures in place. If `ipympl` is unavailable, replace `%matplotlib widget` with `%matplotlib inline`. The widgets still work, but figures are redrawn after each change.
