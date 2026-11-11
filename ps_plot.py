"""Plotting, widget and model helpers for ``power_spectrum.ipynb``.

The power-spectrum maths (periodograms, autocorrelations, random fields and
their estimators) is written out in the notebook itself. This module contains:

* plotting code and the widget plumbing for the interactive cells
* a fitting formula for the matter transfer function (``transfer_function``),
  which is long and not very instructive to write out.

The interactive cells work best with the ipympl backend (``%matplotlib widget``),
which updates figures in place. With the default inline backend the figures are
simply redrawn as static images after each change.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import Audio, display

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#6250d6"
RED = "#e34948"
GREY = "#8a8985"
LIGHT = "#c9c8c3"
INK = "#52514e"

plt.rcParams.update(
    {
        "figure.dpi": 90,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.6,
        "lines.linewidth": 2,
        "legend.frameon": False,
    }
)


def _widget_backend():
    backend = matplotlib.get_backend().lower()
    return "ipympl" in backend or backend == "widget"


def _figure(figsize):
    """Create a figure without displaying it."""
    with plt.ioff():
        fig = plt.figure(figsize=figsize, layout="constrained")
    if _widget_backend():
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.toolbar_position = "right"
    return fig


def _app(fig, update, controls, rows=None, below=None):
    """Wire ``controls`` to ``update`` and display controls above the figure.

    ``below`` is an optional list of extra widgets shown under the figure.
    """
    live = _widget_backend()
    if not live:
        plt.close(fig)
    out = widgets.Output()

    def _run(change=None):
        kwargs = {name: control.value for name, control in controls.items()}
        # Clear immediately (not clear_output(wait=True), which some front ends,
        # e.g. VS Code, do not handle reliably and which can duplicate output).
        out.clear_output()
        with out:
            update(**kwargs)
            if live:
                fig.canvas.draw_idle()
            else:
                display(fig)

    for control in controls.values():
        control.observe(_run, names="value")

    rows = rows or [list(controls.values())]
    labelled = {id(c): _labelled(c) for c in controls.values()}
    children = [widgets.HBox([labelled.get(id(c), c) for c in row]) for row in rows]
    children += [fig.canvas, out] if live else [out]
    children += list(below or [])
    display(widgets.VBox(children))
    _run()


def _labelled(control):
    """Move a control's description into an HTML label (renders in every Jupyter front end)."""
    text = control.description
    if not text or isinstance(control, widgets.Checkbox):
        return control
    control.description = ""
    label = widgets.HTML(f"<span style='white-space:nowrap'>{text}</span>",
                         layout=widgets.Layout(margin="0 4px 0 8px"))
    return widgets.HBox([label, control], layout=widgets.Layout(align_items="center"))


def _slider_style(width="420px"):
    return dict(style={"description_width": "initial"}, layout=widgets.Layout(width=width),
                continuous_update=False)


def _image(ax, data, title=None, cmap="gray", **kwargs):
    im = ax.imshow(data, cmap=cmap, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if title:
        ax.set_title(title)
    return im


def _positive(f, P):
    """Positive frequencies only, in increasing order."""
    keep = f > 0
    return f[keep], P[keep]


# ---------------------------------------------------------------------------
# Model: matter transfer function
# ---------------------------------------------------------------------------


def sound_horizon(Omega_m, f_b, h=0.674):
    """Sound horizon at the drag epoch in Mpc/h (Eisenstein & Hu 1998, eq. 26)."""
    omega_m = Omega_m * h**2
    omega_b = f_b * omega_m
    return 44.5 * np.log(9.83 / omega_m) / np.sqrt(1 + 10 * omega_b**0.75) * h


def transfer_function(k, Omega_m=0.315, f_b=0.157, h=0.674, wiggles=True):
    """Matter transfer function T(k), with k in h/Mpc.

    The smooth part is the "no-wiggle" fitting formula of Eisenstein & Hu (1998),
    which includes the turnover at matter-radiation equality and the suppression
    of small-scale power by baryons. The baryon acoustic oscillations are added
    as a simple damped sinusoid in k s (a toy approximation that captures their
    position, amplitude and damping, but not their exact shape).
    """
    k = np.asarray(k, dtype=float)
    omega_m = Omega_m * h**2
    theta = 2.7255 / 2.7                                   # CMB temperature / 2.7 K
    s = sound_horizon(Omega_m, f_b, h)                     # Mpc/h
    alpha = 1 - 0.328 * np.log(431 * omega_m) * f_b + 0.38 * np.log(22.3 * omega_m) * f_b**2
    gamma = Omega_m * h * (alpha + (1 - alpha) / (1 + (0.43 * k * s) ** 4))
    q = k * theta**2 / gamma
    L0 = np.log(2 * np.e + 1.8 * q)
    C0 = 14.2 + 731 / (1 + 62.5 * q)
    T = L0 / (L0 + C0 * q**2)
    if wiggles:
        x = k * s
        T = T * (1 + 0.7 * f_b * np.sin(x) * x / (x + 8) * np.exp(-(k / 0.15) ** 1.4))
    return T


# ---------------------------------------------------------------------------
# Static figures
# ---------------------------------------------------------------------------


def show_images(images, titles=None, cmaps=None, figsize=None, **kwargs):
    """Display a row of images."""
    n = len(images)
    titles = titles or [None] * n
    cmaps = cmaps or ["gray"] * n
    fig, axes = plt.subplots(1, n, figsize=figsize or (3.2 * n, 3.4), layout="constrained")
    for ax, img, title, cmap in zip(np.atleast_1d(axes), images, titles, cmaps):
        _image(ax, img, title, cmap, **kwargs)
    plt.show()


def signal_and_power(t, x, f, P, title="Signal"):
    """A signal and its (two-sided) power spectrum."""
    fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(10, 5.4), layout="constrained")
    ax_t.plot(t, x, color=INK, lw=1.5)
    ax_t.set_xlabel("$t$ [s]")
    ax_t.set_ylabel("$x_n$")
    ax_t.set_title(title)
    order = np.argsort(f)
    ax_f.vlines(f[order], 0, P[order], color=BLUE, lw=1.2)
    ax_f.plot(f[order], P[order], "o", color=BLUE, ms=4)
    ax_f.set_xlim(-12, 12)
    ax_f.set_xlabel("$f_k$ [Hz]")
    ax_f.set_ylabel("$P_k = |X_k|^2/N^2$")
    ax_f.set_title(f"Power spectrum: total power $\\sum_k P_k = {P.sum():.4f}$")
    plt.show()


def phase_swap(rows):
    """Rows of (title, original, phase-randomised) pairs plus their power spectra."""
    fig, axes = plt.subplots(len(rows), 3, figsize=(11, 3.4 * len(rows)), layout="constrained")
    axes = np.atleast_2d(axes)
    for (name, a, b, k, Pa, Pb), ax in zip(rows, axes):
        if a.ndim == 1:
            for axis, data, title in ((ax[0], a, name), (ax[1], b, "same power, random phases")):
                axis.plot(data, color=INK, lw=1)
                axis.set_title(title)
                axis.set_xlabel("$n$")
        else:
            _image(ax[0], a, name)
            _image(ax[1], b, "same power, random phases")
        ax[2].loglog(k, Pa, color=BLUE, lw=3, label="original")
        ax[2].loglog(k, Pb, color=ORANGE, lw=1.2, ls="--", label="random phases")
        ax[2].set_xlabel("frequency")
        ax[2].set_ylabel("power")
        ax[2].set_title("Power spectra")
        ax[2].legend()
    plt.show()


def autocorrelation_plot(t, x, lag, R, f, P):
    """Signal, autocorrelation and power spectrum side by side."""
    fig = plt.figure(figsize=(11, 6), layout="constrained")
    gs = fig.add_gridspec(2, 2)
    ax_t = fig.add_subplot(gs[0, :])
    ax_r = fig.add_subplot(gs[1, 0])
    ax_f = fig.add_subplot(gs[1, 1])
    ax_t.plot(t, x, color=INK, lw=1)
    ax_t.set_xlabel("$t$ [s]")
    ax_t.set_title("Signal $x_n$: a 5 Hz sine buried in noise")
    order = np.argsort(lag)
    ax_r.plot(lag[order], R[order], color=AQUA, lw=1.5)
    ax_r.set_xlim(-1, 1)
    ax_r.set_xlabel(r"lag $\tau_m$ [s]")
    ax_r.set_title("Autocorrelation $R_m$")
    fp, Pp = _positive(f, P)
    ax_f.plot(fp, Pp, color=BLUE, lw=1)
    ax_f.set_xlim(0, 30)
    ax_f.set_xlabel("$f_k$ [Hz]")
    ax_f.set_title("Power spectrum $P_k$")
    plt.show()


def field_and_spectrum(field, k, P_measured, P_true, n_modes, title, k_label="$k$", extent=None,
                       cmap="RdBu_r", log=True, ratio_to=None):
    """A 2D field (or slice) next to its measured, radially averaged power spectrum.

    Error bars are the Gaussian sample variance, sigma_P / P = sqrt(2 / N_modes).
    If ``ratio_to`` is given, a third panel shows P / ratio_to (e.g. to reveal BAO).
    """
    ncols = 3 if ratio_to is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols - 1, 4.4), layout="constrained")
    lim = np.percentile(np.abs(field), 99.5)
    im = axes[0].imshow(field, cmap=cmap, vmin=-lim, vmax=lim, extent=extent, origin="lower")
    axes[0].grid(False)
    axes[0].set_title(title)
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    err = P_true * np.sqrt(2 / n_modes)
    ax = axes[1]
    ax.fill_between(k, P_true - err, P_true + err, color=LIGHT, lw=0, label=r"input $\pm$ sample variance")
    ax.plot(k, P_true, color=INK, lw=1.5, label="input $P(k)$")
    ax.plot(k, P_measured, "o", color=BLUE, ms=3.5, label="measured")
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(k_label)
    ax.set_ylabel("$P(k)$")
    ax.set_title("Radially averaged power spectrum")
    ax.legend(fontsize=9)
    if ratio_to is not None:
        ax = axes[2]
        ax.fill_between(k, (P_true - err) / ratio_to, (P_true + err) / ratio_to, color=LIGHT, lw=0)
        ax.plot(k, P_true / ratio_to, color=INK, lw=1.5)
        ax.plot(k, P_measured / ratio_to, "o", color=BLUE, ms=3.5)
        ax.axhline(1, color=GREY, lw=0.8)
        ax.set_xlim(0.01, k.max())
        ax.set_ylim(0.8, 1.2)
        ax.set_xlabel(k_label)
        ax.set_ylabel(r"$P(k) / P_{\rm no\ wiggle}(k)$")
        ax.set_title("Divided by the smooth spectrum: BAO wiggles")
    plt.show()


def pdf_comparison(fields, titles):
    """Fields with identical power spectra but different one-point distributions."""
    fig, axes = plt.subplots(1, len(fields) + 1, figsize=(3.4 * (len(fields) + 1) + 0.5, 3.5),
                             layout="constrained")
    colors = [BLUE, ORANGE, AQUA]
    for ax, field, title in zip(axes, fields, titles):
        _image(ax, field, title, cmap="gray")
    bins = np.linspace(min(f.min() for f in fields), max(f.max() for f in fields), 60)
    for field, title, c in zip(fields, titles, colors):
        axes[-1].hist(field.ravel(), bins=bins, histtype="step", color=c, lw=1.8, density=True, label=title)
    axes[-1].set_yscale("log")
    axes[-1].set_xlabel("pixel value")
    axes[-1].set_title("Histogram of pixel values")
    axes[-1].legend(fontsize=8)
    plt.show()


# ---------------------------------------------------------------------------
# Interactive figures
# ---------------------------------------------------------------------------


def coloured_noise_demo(coloured_noise, periodogram, fs=22050, N=2**15):
    """Listen to and look at noise with P(f) proportional to f^-beta."""
    t = np.arange(N) / fs
    fig = _figure((10, 6.2))
    ax_t, ax_f = fig.subplots(2, 1)
    names = {-1: "blue", 0: "white", 1: "pink", 2: "brown (red)"}

    def update(beta):
        x = coloured_noise(N, beta, seed=1)
        f, P = _positive(*periodogram(x, fs))
        ax_t.cla()
        ax_t.plot(t[:2000] * 1e3, x[:2000], color=INK, lw=0.8)
        ax_t.set_xlabel("$t$ [ms]")
        ax_t.set_ylabel("$x_n$")
        ax_t.set_ylim(-4.5, 4.5)
        name = names.get(round(beta, 2), "")
        ax_t.set_title(f"Noise with $P(f) \\propto f^{{-\\beta}}$, $\\beta = {beta:g}$" +
                       (f" ({name} noise)" if name else ""))
        ax_f.cla()
        ax_f.loglog(f, P, color=LIGHT, lw=0.5, label="periodogram $P_k$")
        model = f ** -beta
        model *= P.sum() / model.sum()
        ax_f.loglog(f, model, color=ORANGE, lw=2, label=f"$\\propto f^{{-{beta:g}}}$")
        ax_f.set_xlim(f[0], f[-1])
        ax_f.set_ylim(P.max() * 1e-9, P.max() * 10)
        ax_f.set_xlabel("$f$ [Hz]")
        ax_f.set_ylabel("$P_k$")
        ax_f.legend(loc="upper right")
        player.value = Audio(x, rate=fs).data

    player = widgets.Audio(value=Audio(np.zeros(16), rate=fs, normalize=False).data, format="wav", autoplay=False, loop=False)
    controls = dict(beta=widgets.FloatSlider(1.0, min=-1, max=3, step=0.25,
                                             description="spectral index <i>β</i>", **_slider_style()))
    _app(fig, update, controls, below=[widgets.VBox([widgets.HTML("<b>Listen</b>"), player])])


def averaging_demo(coloured_noise, bartlett, fs=1024, N=2**16, beta=1.0):
    """Average the periodograms of M segments and watch the scatter shrink."""
    x = coloured_noise(N, beta, seed=3)
    fig = _figure((10, 4.6))
    ax = fig.subplots()

    def update(M):
        f, P = _positive(*bartlett(x, M, fs))
        model = f ** -beta
        model *= P.sum() / model.sum()
        ax.cla()
        ax.loglog(f, P, "o" if f.size < 200 else "-", color=BLUE, ms=3, lw=0.6, label=f"average of $M = {M}$ periodograms")
        ax.loglog(f, model, color=ORANGE, lw=2, label=r"true $P(f) \propto f^{-1}$")
        ax.fill_between(f, model * (1 - 1 / np.sqrt(M)), model * (1 + 1 / np.sqrt(M)),
                        color=ORANGE, alpha=0.15, lw=0, label=r"$\pm 1/\sqrt{M}$")
        ax.set_xlim(0.01, fs / 2)
        ax.set_ylim(1e-9, 1e-1)
        ax.set_xlabel("$f$ [Hz]")
        ax.set_ylabel("$P_k$")
        ax.set_title(f"{M} segment{'s' if M > 1 else ''} of {N // M} samples: $\\Delta f = M/T = {M * fs / N:.3g}$ Hz, "
                     f"scatter $\\sigma_P/P = 1/\\sqrt{{M}} = {1 / np.sqrt(M):.2f}$")
        ax.legend(loc="lower left")

    controls = dict(M=widgets.SelectionSlider(options=[2**i for i in range(0, 11)], value=1,
                                              description="segments <i>M</i>", **_slider_style()))
    _app(fig, update, controls)


def grf_demo(gaussian_random_field, radial_power_spectrum, N=256):
    """Gaussian random fields with power-law spectra P(k) proportional to k^n."""
    L = N
    fig = _figure((10.5, 4.6))
    ax_m, ax_p = fig.subplots(1, 2)
    cbar = []

    def update(n, seed):
        power = lambda k: k**n
        field = gaussian_random_field(N, L, power, seed=seed)
        k, P, n_modes = radial_power_spectrum(field, L)
        ax_m.cla()
        lim = 3 * field.std()
        im = ax_m.imshow(field, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
        ax_m.grid(False)
        ax_m.set_xticks([])
        ax_m.set_yticks([])
        ax_m.set_title(f"Gaussian random field, $P(k) \\propto k^{{{n:g}}}$")
        if not cbar:
            cbar.append(fig.colorbar(im, ax=ax_m, shrink=0.8))
        else:
            cbar[0].update_normal(im)
        ax_p.cla()
        ax_p.loglog(k, power(k), color=INK, lw=1.5, label=f"input $k^{{{n:g}}}$")
        ax_p.loglog(k, P, "o", color=BLUE, ms=3, label="measured")
        ax_p.set_ylim(power(k).min() / 10, power(k).max() * 10)
        ax_p.set_xlabel("$k$ [rad / pixel]")
        ax_p.set_ylabel("$P(k)$")
        ax_p.set_title("Radially averaged power spectrum")
        ax_p.legend(loc="lower left" if n > 0 else "upper right")

    s = _slider_style("330px")
    controls = dict(
        n=widgets.FloatSlider(-2.0, min=-4, max=1, step=0.25, description="spectral index <i>n</i>", **s),
        seed=widgets.IntSlider(0, min=0, max=20, description="random seed", **s),
    )
    _app(fig, update, controls)


def matter_demo(matter_power, correlation_function):
    """Matter power spectrum and correlation function for different cosmologies."""
    k = np.logspace(-3.5, 0.5, 800)
    r = np.linspace(20, 180, 161)
    defaults = dict(Omega_m=0.315, f_b=0.157, n_s=0.965)
    P_ref = matter_power(k, **defaults)
    xi_ref = correlation_function(r, k, P_ref)
    fig = _figure((11, 7.2))
    gs = fig.add_gridspec(2, 2)
    ax_p = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_x = fig.add_subplot(gs[1, 1])

    def update(Omega_m, f_b, n_s):
        P = matter_power(k, Omega_m, f_b, n_s)
        P_nw = matter_power(k, Omega_m, f_b, n_s, wiggles=False)
        xi = correlation_function(r, k, P)
        for ax in (ax_p, ax_b, ax_x):
            ax.cla()
        ax_p.loglog(k, P_ref, color=LIGHT, lw=3, label="Planck 2018 values")
        ax_p.loglog(k, P, color=BLUE, label="$P(k)$")
        k_eq = 0.073 * Omega_m * 0.674  # 0.073 Omega_m h^2 Mpc^-1, converted to h/Mpc
        ax_p.axvline(k_eq, color=ORANGE, lw=1, ls="--", label=r"$k_{\rm eq}$ (equality)")
        ax_p.set_xlim(k[0], k[-1])
        ax_p.set_ylim(10, 1e5)
        ax_p.set_xlabel(r"$k$ [$h\,$Mpc$^{-1}$]")
        ax_p.set_ylabel(r"$P(k)$ [$h^{-3}\,$Mpc$^3$]")
        ax_p.set_title("Linear matter power spectrum today, normalised to $\\sigma_8 = 0.81$")
        ax_p.legend(loc="lower left")

        ax_b.semilogx(k, P_ref / matter_power(k, **defaults, wiggles=False), color=LIGHT, lw=3)
        ax_b.semilogx(k, P / P_nw, color=BLUE)
        ax_b.axhline(1, color=GREY, lw=0.8)
        ax_b.set_xlim(0.005, 0.5)
        ax_b.set_ylim(0.88, 1.12)
        ax_b.set_xlabel(r"$k$ [$h\,$Mpc$^{-1}$]")
        ax_b.set_ylabel(r"$P(k) / P_{\rm no\ wiggle}(k)$")
        ax_b.set_title("Baryon acoustic oscillations")

        ax_x.plot(r, r**2 * xi_ref, color=LIGHT, lw=3)
        ax_x.plot(r, r**2 * xi, color=BLUE)
        ax_x.axhline(0, color=GREY, lw=0.8)
        ax_x.set_ylim(-30, 60)
        ax_x.set_xlabel(r"$r$ [$h^{-1}\,$Mpc]")
        ax_x.set_ylabel(r"$r^2 \xi(r)$ [$h^{-2}\,$Mpc$^2$]")
        ax_x.set_title("Correlation function: the BAO peak")

    s = _slider_style("300px")
    controls = dict(
        Omega_m=widgets.FloatSlider(defaults["Omega_m"], min=0.1, max=0.6, step=0.005,
                                    description="Ω<sub>m</sub>", readout_format=".3f", **s),
        f_b=widgets.FloatSlider(defaults["f_b"], min=0.01, max=0.3, step=0.001,
                                description="<i>f</i><sub>b</sub> = Ω<sub>b</sub>/Ω<sub>m</sub>",
                                readout_format=".3f", **s),
        n_s=widgets.FloatSlider(defaults["n_s"], min=0.8, max=1.1, step=0.005,
                                description="<i>n</i><sub>s</sub>", readout_format=".3f", **s),
    )
    _app(fig, update, controls)


def _flat_sky_ell(N, L):
    ell1 = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    return np.hypot(*np.meshgrid(ell1, ell1, indexing="ij"))


def cmb_demo(toy_cmb_spectrum, gaussian_random_field, size_deg=20, N=256):
    """The toy CMB spectrum and a matching flat-sky map."""
    L = np.radians(size_deg)
    ell = np.arange(2, 2500)
    defaults = dict(ell_A=300, R=0.06, ell_D=1100)
    D_ref = toy_cmb_spectrum(ell, **defaults)
    fig = _figure((11, 4.6))
    ax_d, ax_m = fig.subplots(1, 2, width_ratios=[1.35, 1])
    cbar = []

    def update(ell_A, R, ell_D):
        D = toy_cmb_spectrum(ell, ell_A, R, ell_D)
        power = lambda l: 2 * np.pi * np.interp(l, ell, D) / (l * (l + 1))
        cmb = gaussian_random_field(N, L, power, seed=42)
        ax_d.cla()
        ax_d.plot(ell, D_ref, color=LIGHT, lw=3, label="default")
        ax_d.plot(ell, D, color=BLUE, label=r"$\mathcal{D}_\ell$")
        ax_d.set_xlim(2, 2500)
        ax_d.set_ylim(0, 8000)
        ax_d.set_xlabel(r"multipole $\ell$")
        ax_d.set_ylabel(r"$\mathcal{D}_\ell = \ell(\ell+1)C_\ell/2\pi$ [$\mu$K$^2$]")
        ax_d.set_title("Toy CMB temperature power spectrum")
        top = ax_d.secondary_xaxis("top", functions=(lambda l: 180 / np.maximum(l, 1e-3),
                                                     lambda t: 180 / np.maximum(t, 1e-3)))
        top.set_xticks([1, 0.5, 0.25, 0.1])
        top.set_xticklabels(["1°", "0.5°", "0.25°", "0.1°"])
        ax_d.legend(loc="upper right")
        ax_m.cla()
        im = ax_m.imshow(cmb, cmap="RdBu_r", vmin=-300, vmax=300, origin="lower",
                         extent=[0, size_deg, 0, size_deg])
        ax_m.grid(False)
        ax_m.set_xlabel("degrees")
        ax_m.set_title(f"Simulated {size_deg}° × {size_deg}° patch")
        if not cbar:
            cbar.append(fig.colorbar(im, ax=ax_m, shrink=0.8, label=r"$\Delta T$ [$\mu$K]"))

    s = _slider_style("320px")
    controls = dict(
        ell_A=widgets.IntSlider(defaults["ell_A"], min=150, max=450, step=5,
                                description="acoustic scale <i>ℓ</i><sub>A</sub>", **s),
        R=widgets.FloatSlider(defaults["R"], min=0, max=0.4, step=0.02,
                              description="baryon loading <i>R</i>", **s),
        ell_D=widgets.IntSlider(defaults["ell_D"], min=500, max=2500, step=50,
                                description="damping scale <i>ℓ</i><sub>D</sub>", **s),
    )
    _app(fig, update, controls, rows=[[controls["ell_A"], controls["R"]], [controls["ell_D"]]])


def cmb_instrument_demo(cmb, L, ell_theory, D_theory, radial_power_spectrum, n_bins=64):
    """Observe a CMB patch with a telescope of given beam and noise, then measure C_ell."""
    N = cmb.shape[0]
    arcmin = np.radians(1 / 60)
    pixel = L / N
    ell_grid = _flat_sky_ell(N, L)
    white = np.random.default_rng(7).standard_normal(cmb.shape)
    cmb_ft = np.fft.fft2(cmb)
    size_deg = np.degrees(L)
    fig = _figure((11, 4.8))
    ax_m, ax_d = fig.subplots(1, 2, width_ratios=[1, 1.35])
    cbar = []

    def update(fwhm, noise):
        sigma_b = fwhm * arcmin / np.sqrt(8 * np.log(2))
        beam = lambda l: np.exp(-l * (l + 1) * sigma_b**2 / 2)
        N_ell = (noise * arcmin) ** 2                         # white noise C_ell [uK^2 sr]
        observed = np.fft.ifft2(cmb_ft * beam(ell_grid)).real + noise * arcmin / pixel * white
        ell, C_obs, n_modes = radial_power_spectrum(observed, L, n_bins)
        B2 = beam(ell) ** 2
        C_hat = (C_obs - N_ell) / B2                          # remove noise bias and beam
        C_th = 2 * np.pi * np.interp(ell, ell_theory, D_theory) / (ell * (ell + 1))
        err = np.sqrt(2 / n_modes) * (C_th + N_ell / B2)      # cosmic variance + noise
        to_D = ell * (ell + 1) / (2 * np.pi)

        ax_m.cla()
        im = ax_m.imshow(observed, cmap="RdBu_r", vmin=-300, vmax=300, origin="lower",
                         extent=[0, size_deg, 0, size_deg])
        ax_m.grid(False)
        ax_m.set_xlabel("degrees")
        ax_m.set_title(f"Observed: {fwhm:g}′ beam, {noise:g} μK-arcmin noise")
        if not cbar:
            cbar.append(fig.colorbar(im, ax=ax_m, shrink=0.8, label=r"$\Delta T$ [$\mu$K]"))

        ax_d.cla()
        ax_d.plot(ell_theory, D_theory, color=INK, lw=1.5, label=r"true $\mathcal{D}_\ell$")
        lt = ell_theory
        ax_d.plot(lt, D_theory * beam(lt) ** 2 + lt * (lt + 1) * N_ell / (2 * np.pi),
                  color=ORANGE, lw=1.2, ls="--", label=r"expected: $C_\ell B_\ell^2 + N_\ell$")
        ax_d.plot(ell, to_D * C_obs, "o", color=GREY, ms=3, label="measured (raw)")
        good = (B2 > 0.02) & (err < 2 * C_th)
        ax_d.errorbar(ell[good], (to_D * C_hat)[good], (to_D * err)[good], fmt="o", color=BLUE, ms=3.5,
                      elinewidth=1, label=r"corrected: $(\hat{C}_\ell - N_\ell)/B_\ell^2$")
        ax_d.set_xlim(0, 2300)
        ax_d.set_ylim(0, 8000)
        ax_d.set_xlabel(r"multipole $\ell$")
        ax_d.set_ylabel(r"$\mathcal{D}_\ell$ [$\mu$K$^2$]")
        ax_d.set_title("Measured power spectrum")
        ax_d.legend(loc="upper right", fontsize=8.5)

    s = _slider_style("330px")
    controls = dict(
        fwhm=widgets.FloatSlider(5.0, min=0, max=30, step=0.5, description="beam FWHM [arcmin]", **s),
        noise=widgets.FloatSlider(30.0, min=0, max=200, step=5, description="noise [μK-arcmin]", **s),
    )
    _app(fig, update, controls)


# ---------------------------------------------------------------------------
# Points: correlation function and power spectrum
# ---------------------------------------------------------------------------


def points_demo(distributions, L, pair_correlation, point_power_spectrum, theory=None):
    """Point distributions with their two-point correlation function and power spectrum.

    ``distributions`` maps a name to an (n, 2) array of positions in a periodic box of
    side L. ``theory`` optionally maps a name to a pair of functions (xi(r), P(k)).
    """
    theory = theory or {}
    r_edges = np.logspace(np.log10(L / 400), np.log10(0.4 * L), 31)
    cache = {}

    def stats(name):
        if name not in cache:
            pts = distributions[name]
            r, xi = pair_correlation(pts, L, r_edges)
            k, P, _ = point_power_spectrum(pts, L)
            cache[name] = (r, xi, k, P)
        return cache[name]

    fig = _figure((11.5, 4.2))
    ax_p, ax_x, ax_k = fig.subplots(1, 3)

    def update(name):
        pts = distributions[name]
        r, xi, k, P = stats(name)
        shot = L**2 / len(pts)
        for ax in (ax_p, ax_x, ax_k):
            ax.cla()
        ax_p.plot(pts[:, 0], pts[:, 1], ".", color=INK, ms=1.5)
        ax_p.set_xlim(0, L)
        ax_p.set_ylim(0, L)
        ax_p.set_aspect("equal")
        ax_p.set_xticks([])
        ax_p.set_yticks([])
        ax_p.grid(False)
        ax_p.set_title(f"{name}: {len(pts)} points")

        ax_x.axhline(0, color=GREY, lw=0.8)
        ax_x.plot(r, xi, "o-", color=BLUE, ms=3.5, lw=1.2, label=r"measured $\hat{\xi}(r)$")
        ax_k.axhline(shot, color=GREY, lw=1.2, ls="--", label=r"shot noise $1/\bar{n}$")
        ax_k.loglog(k, P, "o", color=BLUE, ms=2.5, label=r"measured $\hat{P}(k)$")
        if name in theory:
            xi_th, P_th = theory[name]
            rr = np.logspace(np.log10(r[0]), np.log10(r[-1]), 200)
            ax_x.plot(rr, xi_th(rr), color=ORANGE, lw=1.5, label="theory")
            ax_k.loglog(k, P_th(k) + shot, color=ORANGE, lw=1.5, label=r"theory + $1/\bar{n}$")
        ax_x.set_xscale("log")
        ax_x.set_yscale("symlog", linthresh=0.1)
        ax_x.set_ylim(-1.5, 300)
        ax_x.set_xlabel("separation $r$")
        ax_x.set_title(r"Two-point correlation function $\xi(r)$")
        ax_x.legend(loc="upper right", fontsize=9)
        ax_k.set_ylim(shot / 30, shot * 300)
        ax_k.set_xlabel("wavenumber $k$ [rad per unit length]")
        ax_k.set_title("Power spectrum $P(k)$")
        ax_k.legend(loc="upper right", fontsize=9)

    controls = dict(name=widgets.Dropdown(options=list(distributions), value=list(distributions)[0],
                                          description="distribution", layout=widgets.Layout(width="300px")))
    _app(fig, update, controls)


# ---------------------------------------------------------------------------
# Masks and pseudo-C_ell
# ---------------------------------------------------------------------------


def _smooth(field, L, sigma):
    """Gaussian smoothing (periodic), with sigma in the same units as L."""
    k = _flat_sky_ell(field.shape[0], L)
    return np.fft.ifft2(np.fft.fft2(field) * np.exp(-(k * sigma) ** 2 / 2)).real


def make_mask(kind, N, L, apodise=False, seed=0):
    """A flat-sky mask: 1 where the sky is observed, 0 where it is masked.

    ``kind`` is one of "Point-source holes", "Galactic strip" or "Holes + strip". With
    ``apodise`` the edges are tapered smoothly to zero over about half a degree.
    """
    deg = np.radians(1)
    x, y = np.meshgrid(np.linspace(0, L, N, endpoint=False), np.linspace(0, L, N, endpoint=False))
    mask = np.ones((N, N))
    if "holes" in kind.lower():
        rng = np.random.default_rng(seed)
        for cx, cy, radius in zip(*rng.uniform(0, L, (2, 50)), rng.uniform(0.15, 0.5, 50) * deg):
            dx = (x - cx + L / 2) % L - L / 2
            dy = (y - cy + L / 2) % L - L / 2
            mask[np.hypot(dx, dy) < radius] = 0
    if "strip" in kind.lower():
        mask[np.abs(y - 0.5 * L - 0.25 * (x - 0.5 * L)) < 2 * deg] = 0
    if apodise:
        smooth = _smooth(mask, L, 0.4 * deg)
        mask = np.clip((smooth - 0.5) / 0.5, 0, 1)
        mask = np.sin(np.pi / 2 * mask) ** 2
    return mask


def pseudo_cl_demo(cmb, L, ell_theory, D_theory, radial_power_spectrum, coupling_matrix, n_bins=32):
    """Measure the power spectrum of a masked CMB patch: pseudo-C_ell, f_sky and MASTER."""
    N = cmb.shape[0]
    size_deg = np.degrees(L)
    ell_true, C_true, _ = radial_power_spectrum(cmb, L, n_bins)
    to_D = lambda ell, C: ell * (ell + 1) * C / (2 * np.pi)
    fig = _figure((12, 4.4))
    ax_m, ax_d, ax_c = fig.subplots(1, 3, width_ratios=[1, 1.3, 1])
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(LIGHT)
    images = []

    def update(kind, apodise):
        mask = make_mask(kind, N, L, apodise)
        f_sky = np.mean(mask**2)
        ell, C_pseudo, _ = radial_power_spectrum(cmb * mask, L, n_bins)
        M = coupling_matrix(mask, L, n_bins)
        C_master = np.linalg.solve(M, C_pseudo)

        ax_m.cla()
        shown = np.ma.masked_where(mask == 0, cmb * mask)
        ax_m.imshow(shown, cmap=cmap, vmin=-300, vmax=300, origin="lower", extent=[0, size_deg, 0, size_deg])
        ax_m.grid(False)
        ax_m.set_xlabel("degrees")
        ax_m.set_title(f"Masked map: $f_{{\\rm sky}} = {f_sky:.2f}$")

        ax_d.cla()
        ax_d.plot(ell_theory, D_theory, color=INK, lw=1.2, label="true spectrum")
        ax_d.plot(ell_true, to_D(ell_true, C_true), "o", color=LIGHT, ms=7, label="unmasked map")
        ax_d.plot(ell, to_D(ell, C_pseudo), "x", color=GREY, ms=5, label=r"pseudo-$C_\ell$ $\tilde{C}_\ell$")
        ax_d.plot(ell, to_D(ell, C_pseudo / f_sky), "s", color=ORANGE, ms=3.5, label=r"$\tilde{C}_\ell / f_{\rm sky}$")
        ax_d.plot(ell, to_D(ell, C_master), "o", color=BLUE, ms=3.5, label=r"MASTER: $M^{-1}\tilde{C}$")
        ax_d.set_xlim(0, 2300)
        ax_d.set_ylim(0, 7000)
        ax_d.set_xlabel(r"multipole $\ell$")
        ax_d.set_ylabel(r"$\mathcal{D}_\ell$ [$\mu$K$^2$]")
        ax_d.set_title("Power spectrum of the masked map")
        ax_d.legend(loc="upper right", fontsize=8.5)

        ax_c.cla()
        edges = np.linspace(np.pi / L, np.pi * N / L, n_bins + 1)
        ax_c.imshow(np.log10(np.maximum(M / f_sky, 1e-4)), cmap="magma", vmin=-4, vmax=0, origin="lower",
                    extent=[edges[0], edges[-1], edges[0], edges[-1]])
        ax_c.grid(False)
        ax_c.set_xlabel(r"true multipole $\ell'$")
        ax_c.set_ylabel(r"measured multipole $\ell$")
        ax_c.set_title(r"Coupling matrix $\log_{10}(M_{\ell\ell'}/f_{\rm sky})$")

    controls = dict(
        kind=widgets.Dropdown(options=["Point-source holes", "Galactic strip", "Holes + strip"],
                              value="Holes + strip", description="mask", layout=widgets.Layout(width="280px")),
        apodise=widgets.Checkbox(False, description="apodise the edges"),
    )
    _app(fig, update, controls)


# ---------------------------------------------------------------------------
# Weak lensing: 3x2pt
# ---------------------------------------------------------------------------

SOURCE_COLORS = [BLUE, ORANGE, AQUA]
LENS_COLORS = [VIOLET, RED]


def three_by_two_demo(three_by_two_point, ell, z, n_source, n_lens, shape_noise, shot_noise):
    """The three sets of power spectra of a 3x2pt analysis, for adjustable Omega_m, sigma_8 and bias."""
    defaults = dict(Omega_m=0.315, sigma_8=0.811, b=1.5)
    reference = three_by_two_point(ell, **defaults)
    to_D = ell * (ell + 1) / (2 * np.pi)
    fig = _figure((11.5, 8))
    axes = fig.subplots(2, 2)
    ax_n, ax_s, ax_g, ax_c = axes.ravel()
    styles = ["-", "--", ":"]

    # Redshift distributions (static).
    for i, n in enumerate(n_source):
        ax_n.fill_between(z, n, color=SOURCE_COLORS[i], alpha=0.25, lw=0)
        ax_n.plot(z, n, color=SOURCE_COLORS[i], lw=1.5, label=f"source bin {i + 1}")
    for i, n in enumerate(n_lens):
        ax_n.plot(z, n, color=LENS_COLORS[i], lw=2, ls="--", label=f"lens bin {i + 1}")
    ax_n.set_xlim(0, 2.5)
    ax_n.set_ylim(0)
    ax_n.set_xlabel("redshift $z$")
    ax_n.set_ylabel("$n_i(z)$")
    ax_n.set_title("Tomographic redshift distributions")
    ax_n.legend(fontsize=8.5)

    def update(Omega_m, sigma_8, b):
        shear, ggl, clustering = three_by_two_point(ell, Omega_m, sigma_8, b)
        for ax in (ax_s, ax_g, ax_c):
            ax.cla()
        for (i, j), C in shear.items():
            ax_s.loglog(ell, to_D * reference[0][i, j], color=LIGHT, lw=2.5)
            ax_s.loglog(ell, to_D * C, color=SOURCE_COLORS[j], ls="-" if i == j else styles[1 + (j - i - 1) % 2],
                        lw=1.6 if i == j else 1.1, label=f"({i + 1}, {j + 1})")
        ax_s.loglog(ell, to_D * shape_noise, color=GREY, lw=1, ls=(0, (1, 1)), label=r"shape noise $\sigma_\epsilon^2/\bar{n}_i$")
        ax_s.set_title(r"Cosmic shear $C_\ell^{\gamma\gamma}$ (source $\times$ source)")
        for (l, s), C in ggl.items():
            ax_g.loglog(ell, to_D * np.maximum(reference[1][l, s], 1e-30), color=LIGHT, lw=2.5)
            ax_g.loglog(ell, to_D * np.maximum(C, 1e-30), color=LENS_COLORS[l], ls=styles[s], lw=1.4,
                        label=f"lens {l + 1} $\\times$ source {s + 1}")
        ax_g.set_title(r"Galaxy-galaxy lensing $C_\ell^{g\gamma}$ (lens $\times$ source)")
        for (l, m), C in clustering.items():
            ax_c.loglog(ell, to_D * np.maximum(reference[2][l, m], 1e-30), color=LIGHT, lw=2.5)
            ax_c.loglog(ell, to_D * np.maximum(C, 1e-30), color=LENS_COLORS[m] if l == m else INK,
                        ls="-" if l == m else "--", lw=1.6 if l == m else 1.1, label=f"({l + 1}, {m + 1})")
        ax_c.loglog(ell, to_D * shot_noise, color=GREY, lw=1, ls=(0, (1, 1)), label=r"shot noise $1/\bar{n}_i$")
        ax_c.set_title(r"Galaxy clustering $C_\ell^{gg}$ (lens $\times$ lens)")
        for ax, lo, hi in ((ax_s, 1e-7, 1e-3), (ax_g, 1e-7, 1e-3), (ax_c, 1e-7, 1e-1)):
            ax.set_xlim(ell[0], ell[-1])
            ax.set_ylim(lo, hi)
            ax.set_xlabel(r"multipole $\ell$")
            ax.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$")
            ax.legend(fontsize=7.5, ncols=2, loc="upper left")

    s = _slider_style("300px")
    controls = dict(
        Omega_m=widgets.FloatSlider(defaults["Omega_m"], min=0.2, max=0.45, step=0.005,
                                    description="Ω<sub>m</sub>", readout_format=".3f", **s),
        sigma_8=widgets.FloatSlider(defaults["sigma_8"], min=0.5, max=1.1, step=0.01,
                                    description="<i>σ</i><sub>8</sub>", readout_format=".2f", **s),
        b=widgets.FloatSlider(defaults["b"], min=0.5, max=3.0, step=0.05, description="galaxy bias <i>b</i>", **s),
    )
    _app(fig, update, controls)
