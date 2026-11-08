"""Plotting and interactive-widget helpers for ``fourier_transform.ipynb``.

All of the Fourier maths is written out in the notebook itself. This module only
contains plotting code and the widget plumbing needed to make the interactive
cells responsive.

The interactive cells work best with the ipympl backend (``%matplotlib widget``),
which updates figures in place. With the default inline backend the figures are
simply redrawn as static images after each change.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import Audio, display
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#6250d6"
RED = "#e34948"
GREY = "#8a8985"
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
    return dict(style={"description_width": "initial"}, layout=widgets.Layout(width=width))


def _image(ax, data, title=None, cmap="gray", **kwargs):
    im = ax.imshow(data, cmap=cmap, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if title:
        ax.set_title(title)
    return im


def _log_amp(F):
    """Log-scaled amplitude of a centred spectrum, for display."""
    return np.log10(1 + np.abs(F))


# ---------------------------------------------------------------------------
# Static helpers
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


def show_spectrum(img, title="Image"):
    """Show an image with its centred log-amplitude and phase spectra."""
    F = np.fft.fftshift(np.fft.fft2(img))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), layout="constrained")
    _image(axes[0], img, title)
    ny, nx = img.shape
    ext = [-0.5, 0.5, 0.5, -0.5]
    for ax, data, cmap, label in (
        (axes[1], _log_amp(F), "magma", r"$\log_{10}(1+|F|)$"),
        (axes[2], np.angle(F), "twilight", "phase [rad]"),
    ):
        im = ax.imshow(data, cmap=cmap, extent=ext)
        ax.grid(False)
        ax.set_xlabel(r"$u$ [cycles / pixel]")
        ax.set_ylabel(r"$v$ [cycles / pixel]")
        fig.colorbar(im, ax=ax, shrink=0.8, label=label)
    axes[1].set_title("Amplitude")
    axes[2].set_title("Phase")
    plt.show()


def stem(ax, x, y, color=BLUE, label=None, markersize=4):
    """Light-weight stem plot."""
    ax.vlines(x, 0, y, color=color, lw=1.2)
    ax.plot(x, y, "o", color=color, ms=markersize, label=label)


def timing_plot(sizes, timings):
    """Log-log plot of run time against signal size."""
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    colors = [BLUE, ORANGE, AQUA]
    for (name, times), color in zip(timings.items(), colors):
        n = np.asarray(sizes[: len(times)])
        ax.loglog(n, times, "o-", color=color, label=name, ms=5)
    n = np.asarray(sizes, dtype=float)
    ref = timings[list(timings)[0]][0]
    ax.loglog(n, ref * (n / n[0]) ** 2, ":", color=GREY, lw=1.5, label=r"$\propto N^2$")
    ref = timings[list(timings)[1]][0]
    ax.loglog(
        n, ref * (n * np.log2(n)) / (n[0] * np.log2(n[0])), "--", color=GREY, lw=1.5,
        label=r"$\propto N\log_2 N$",
    )
    ax.set_xlabel("$N$")
    ax.set_ylabel("time [s]")
    ax.set_title("Run time of the DFT and FFT")
    ax.legend()
    plt.show()


def butterfly_diagram(N=8):
    """Draw the signal-flow (butterfly) diagram of a radix-2 FFT."""
    n_stages = int(np.log2(N))
    bits = n_stages
    order = [int(f"{i:0{bits}b}"[::-1], 2) for i in range(N)]

    fig, ax = plt.subplots(figsize=(9, 0.55 * N + 1), layout="constrained")
    ax.axis("off")
    y = lambda i: N - 1 - i
    dx = 2.2

    for s in range(n_stages):
        span = 2**s
        x0, x1 = s * dx, (s + 1) * dx
        for start in range(0, N, 2 * span):
            for j in range(span):
                top, bot = start + j, start + j + span
                for a, b in ((top, top), (bot, top), (top, bot), (bot, bot)):
                    color = ORANGE if a == bot else BLUE
                    ax.plot([x0, x1], [y(a), y(b)], color=color, lw=1.2, alpha=0.85)
                k = j * N // (2 * span)
                ax.text(x0 + 0.15, y(bot) - 0.32, rf"$W_{{{N}}}^{{{k}}}$", fontsize=8, color=ORANGE)
                ax.text(x1 - 0.35, y(bot) - 0.32, r"$-$", fontsize=10, color=ORANGE)
        ax.text((x0 + x1) / 2, N - 0.2, f"stage {s + 1}\n(size {2 * span})", ha="center", color=INK)

    for col in range(n_stages + 1):
        ax.plot([col * dx] * N, [y(i) for i in range(N)], "o", color=INK, ms=5, zorder=3)
    for i in range(N):
        ax.text(-0.25, y(i), f"$x_{order[i]}$", ha="right", va="center", fontsize=11)
        ax.text(n_stages * dx + 0.25, y(i), f"$X_{i}$", ha="left", va="center", fontsize=11)
    ax.set_xlim(-1, n_stages * dx + 1)
    ax.set_ylim(-0.8, N + 0.6)
    ax.set_title(
        f"Radix-2 FFT butterflies for $N={N}$ (inputs in bit-reversed order; "
        "orange paths are multiplied by the twiddle factor)",
        fontsize=10,
    )
    plt.show()


# ---------------------------------------------------------------------------
# Interactive demos
# ---------------------------------------------------------------------------


def complex_exponential(z_func, t_max=2.0):
    """A point A e^{2 pi i f t} rotating in the complex plane, and its real and imaginary parts."""
    tc = np.linspace(0, t_max, 1000)
    theta = np.linspace(0, 2 * np.pi, 200)
    fig = _figure((10, 4.4))
    ax_c, ax_t = fig.subplots(1, 2, width_ratios=[1, 1.6])

    def update(A, f, t):
        z = z_func(A, f, t)
        path = z_func(A, f, tc[tc <= t])

        ax_c.cla()
        ax_c.plot(A * np.cos(theta), A * np.sin(theta), color=GREY, lw=0.8)
        ax_c.plot(path.real, path.imag, color=INK, lw=2, alpha=0.35)
        ax_c.plot([z.real, z.real], [0, z.imag], ":", color=BLUE, lw=1.5)
        ax_c.plot([0, z.real], [z.imag, z.imag], ":", color=ORANGE, lw=1.5)
        ax_c.annotate("", xy=(z.real, z.imag), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
        ax_c.plot([z.real], [0], "o", color=BLUE, ms=8, mec="white", mew=1.5)
        ax_c.plot([0], [z.imag], "o", color=ORANGE, ms=8, mec="white", mew=1.5)
        ax_c.set_aspect("equal")
        ax_c.set_xlim(-2.2, 2.2)
        ax_c.set_ylim(-2.2, 2.2)
        ax_c.axhline(0, color=GREY, lw=0.8)
        ax_c.axvline(0, color=GREY, lw=0.8)
        ax_c.set_xlabel("real")
        ax_c.set_ylabel("imaginary")
        sign = "+" if z.imag >= 0 else "-"
        ax_c.set_title(f"$A e^{{2\\pi i f t}} = {z.real:.2f} {sign} {abs(z.imag):.2f}i$")

        ax_t.cla()
        ax_t.plot(tc, z_func(A, f, tc).real, color=BLUE, label=r"Re: $A\cos(2\pi f t)$")
        ax_t.plot(tc, z_func(A, f, tc).imag, color=ORANGE, label=r"Im: $A\sin(2\pi f t)$")
        ax_t.axvline(t, color=INK, lw=1)
        ax_t.plot([t], [z.real], "o", color=BLUE, ms=8, mec="white", mew=1.5)
        ax_t.plot([t], [z.imag], "o", color=ORANGE, ms=8, mec="white", mew=1.5)
        ax_t.set_xlim(0, t_max)
        ax_t.set_ylim(-2.2, 2.9)
        ax_t.set_xlabel("$t$ [s]")
        ax_t.legend(loc="upper center", ncols=2)
        turns = f * t
        ax_t.set_title(f"angle $2\\pi f t$ = {turns:.2f} turns ({np.degrees(2 * np.pi * turns):.0f}°)")

    s = _slider_style("300px")
    controls = dict(
        A=widgets.FloatSlider(1, min=0.1, max=2, step=0.05, description="amplitude <i>A</i>", **s),
        f=widgets.FloatSlider(1, min=-3, max=3, step=0.1, description="frequency <i>f</i> [Hz]", **s),
        t=widgets.FloatSlider(0.2, min=0, max=t_max, step=0.01, description="time <i>t</i> [s]", **s),
    )
    play = widgets.Play(min=0, max=200, step=1, interval=50, description="animate")

    def _play(change):
        controls["t"].value = change["new"] * t_max / 200

    play.observe(_play, names="value")
    _app(fig, update, controls, rows=[[controls["A"], controls["f"]], [play, controls["t"]]])


def fourier_series(t, shapes, coefficients, synthesis):
    """Build periodic signals from a growing number of sinusoids."""
    fig = _figure((10, 3.8))
    ax_t, ax_c = fig.subplots(1, 2, width_ratios=[1.6, 1])

    def update(shape, n_max):
        g = shapes[shape]
        n, c = coefficients(g, t, n_max)
        ax_t.cla()
        ax_t.plot(t, g, color=GREY, lw=1.5, label="$g(t)$")
        ax_t.plot(
            t, synthesis(n, c, t), color=BLUE,
            label=rf"$\sum_{{n=-n_{{\max}}}}^{{n_{{\max}}}} c_n\, e^{{2\pi i n t/T}}$  ($n_{{\max}}={n_max}$)",
        )
        ax_t.set_xlabel("$t$ [s]")
        ax_t.set_ylabel("$g(t)$")
        ax_t.set_ylim(-1.5, 2.4)
        ax_t.legend(loc="upper center", ncols=2)
        ax_t.set_title(f"Signal $g(t)$: {shape.lower()} with period $T = {t[-1] - t[0] + t[1] - t[0]:g}$ s")
        ax_c.cla()
        stem(ax_c, n, np.abs(c))
        ax_c.set_xlim(-51, 51)
        ax_c.set_ylim(0, 0.75)
        ax_c.set_xlabel("harmonic $n$ (frequency $n/T$)")
        ax_c.set_ylabel(r"$|c_n|$")
        ax_c.set_title(r"Fourier coefficients $|c_n|$, $-n_{\max} \leq n \leq n_{\max}$")

    controls = dict(
        shape=widgets.Dropdown(options=list(shapes), description="<i>g</i>(<i>t</i>)"),
        n_max=widgets.IntSlider(1, 1, 50, description="<i>n</i><sub>max</sub>", **_slider_style()),
    )
    _app(fig, update, controls)


def winding_machine(t, g, wind, f_max=5.0):
    """Wrap a signal around the origin at frequency f and track its centre of mass."""
    freqs = np.linspace(0, f_max, 501)
    com = np.array([wind(g, t, f).mean() for f in freqs])
    r = 1.1 * np.abs(g).max()

    fig = _figure((10, 6.6))
    mosaic = fig.subplot_mosaic([["sig", "sig"], ["wind", "com"]], height_ratios=[1, 1.7])
    ax_s, ax_w, ax_c = mosaic["sig"], mosaic["wind"], mosaic["com"]

    ax_s.plot(t, g, color=BLUE)
    ax_s.set_xlabel("$t$ [s]")
    ax_s.set_ylabel("$g(t)$")
    ax_s.set_xlim(t[0], t[-1])
    bands = []

    (curve,) = ax_w.plot([], [], color=BLUE, lw=1.2, alpha=0.9)
    (cm_dot,) = ax_w.plot([], [], "o", color=ORANGE, ms=10, mec="white", mew=2, label="centre of mass")
    ax_w.set_aspect("equal")
    ax_w.set_xlim(-r, r)
    ax_w.set_ylim(-r, r)
    ax_w.axhline(0, color=GREY, lw=0.8)
    ax_w.axvline(0, color=GREY, lw=0.8)
    ax_w.set_xlabel("real")
    ax_w.set_ylabel("imaginary")
    ax_w.legend(loc="upper right", fontsize=9)

    ax_c.plot(freqs, com.real, color=BLUE, label="real part")
    ax_c.plot(freqs, np.abs(com), color=AQUA, lw=1.5, ls="--", label="magnitude")
    marker = ax_c.axvline(0, color=ORANGE, lw=1.2)
    (c_dot,) = ax_c.plot([], [], "o", color=ORANGE, ms=8, mec="white", mew=1.5)
    ax_c.set_xlim(0, f_max)
    ax_c.set_xlabel("winding frequency $f$ [Hz]")
    ax_c.set_ylabel("centre of mass")
    ax_c.legend(loc="upper right", fontsize=9)

    def update(f):
        z = wind(g, t, f)
        m = z.mean()
        curve.set_data(z.real, z.imag)
        cm_dot.set_data([m.real], [m.imag])
        marker.set_xdata([f, f])
        c_dot.set_data([f], [m.real])
        ax_w.set_title(f"$g(t)\\,e^{{-2\\pi i f t}}$ with $f = {f:.2f}$ Hz")
        while bands:
            bands.pop().remove()
        if f > 0.15:
            for start in np.arange(t[0], t[-1], 2 / f):
                bands.append(ax_s.axvspan(start, start + 1 / f, color=ORANGE, alpha=0.08, lw=0))
        ax_s.set_title("Signal (shaded bands = one full turn around the circle)")

    slider = widgets.FloatSlider(
        0.5, min=0, max=f_max, step=0.01, description="winding frequency <i>f</i> [Hz]",
        readout_format=".2f", **_slider_style("520px"),
    )
    play = widgets.Play(min=0, max=500, step=1, interval=60, description="animate")

    def _play(change):
        slider.value = change["new"] * f_max / 500

    play.observe(_play, names="value")
    _app(fig, update, dict(f=slider), rows=[[play, slider]])


def ft_pairs(t, f, pairs, cft):
    """Compare analytic and numerical continuous Fourier transforms on time grid t and frequency grid f."""
    fig = _figure((10, 6))
    ax_t, ax_f = fig.subplots(2, 1)

    def rms_width(x, p):
        p = p / np.trapezoid(p, x) if hasattr(np, "trapezoid") else p / np.trapz(p, x)
        mean = np.sum(x * p) * (x[1] - x[0])
        return np.sqrt(np.sum((x - mean) ** 2 * p) * (x[1] - x[0]))

    def update(pair, width, shift):
        g_func, G_func = pairs[pair]
        g = g_func(t - shift, width)
        G = G_func(f, width) * np.exp(-2j * np.pi * f * shift)
        G_num = cft(g, t, f[::8])

        ax_t.cla()
        ax_t.plot(t, g.real, color=BLUE)
        ax_t.set_xlim(t[0], t[-1])
        ax_t.set_ylim(-1.1, 1.2)
        ax_t.set_xlabel("$t$")
        ax_t.set_ylabel("$g(t)$")
        ax_t.set_title(f"{pair}: width = {width:.2f}, shift $t_0$ = {shift:.2f}")

        ax_f.cla()
        ax_f.plot(f, np.abs(G), color=GREY, ls="--", lw=1.5, label=r"$|\hat{g}(f)|$")
        ax_f.plot(f, G.real, color=BLUE, label=r"Re $\hat{g}(f)$")
        ax_f.plot(f, G.imag, color=ORANGE, label=r"Im $\hat{g}(f)$")
        ax_f.plot(f[::8], G_num.real, "o", mfc="none", mec=INK, ms=5, mew=1, label="numerical (Re)")
        lim = 1.15 * np.abs(G).max()
        ax_f.set_ylim(-lim, lim)
        ax_f.set_xlim(f[0], f[-1])
        ax_f.set_xlabel("$f$")
        ax_f.set_ylabel(r"$\hat{g}(f)$")
        prod = rms_width(t, np.abs(g) ** 2) * rms_width(f, np.abs(G) ** 2)
        ax_f.set_title(
            rf"$\sigma_t\,\sigma_f = {prod:.3f}$   (uncertainty limit $1/4\pi \approx 0.080$)"
        )
        ax_f.legend(loc="upper right", fontsize=9, ncols=2)

    controls = dict(
        pair=widgets.Dropdown(options=list(pairs), description="Function"),
        width=widgets.FloatSlider(1, min=0.2, max=2.5, step=0.05, description="width", **_slider_style("300px")),
        shift=widgets.FloatSlider(0, min=-3, max=3, step=0.1, description="shift <i>t</i><sub>0</sub>", **_slider_style("300px")),
    )
    _app(fig, update, controls)


def dft_basis(W):
    """Explore the rows (basis vectors) of a DFT matrix."""
    N = W.shape[0]
    n = np.arange(N)
    tc = np.linspace(0, N - 1, 500)

    fig = _figure((11, 4.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.6])
    ax_m = fig.add_subplot(gs[:, 0])
    ax_re = fig.add_subplot(gs[0, 1])
    ax_im = fig.add_subplot(gs[1, 1], sharex=ax_re)

    ax_m.imshow(W.real, cmap="RdBu_r", vmin=-1, vmax=1, origin="upper")
    ax_m.grid(False)
    ax_m.set_xlabel("$n$ (time sample)")
    ax_m.set_ylabel("$k$ (frequency index)")
    ax_m.set_title(r"Re$(W_{kn}) = \cos(2\pi kn/N)$")
    box = Rectangle((-0.5, -0.5), N, 1, fill=False, ec=ORANGE, lw=2.5)
    ax_m.add_patch(box)

    def update(k):
        box.set_y(k - 0.5)
        alias = k - N if k > N // 2 else k
        for ax, part, func, name in (
            (ax_re, W[k].real, np.cos, "Re"),
            (ax_im, W[k].imag, lambda x: -np.sin(x), "Im"),
        ):
            ax.cla()
            ax.plot(tc, func(2 * np.pi * k * tc / N), color=GREY, lw=1, label=f"frequency $k={k}$")
            if alias != k:
                ax.plot(tc, func(2 * np.pi * alias * tc / N), color=AQUA, lw=1.5, ls="--",
                        label=f"frequency $k-N={alias}$")
            stem(ax, n, part, color=BLUE, markersize=6)
            ax.set_ylim(-1.3, 1.3)
            ax.set_ylabel(name)
            ax.legend(loc="upper right", fontsize=8, ncols=2)
        ax_im.set_xlabel("$n$")
        ax_re.set_title(f"Row $k = {k}$ of the DFT matrix ($N={N}$)")

    controls = dict(k=widgets.IntSlider(1, 0, N - 1, description="row <i>k</i>", **_slider_style()))
    _app(fig, update, controls)


def signal_and_spectrum(t, x, freqs, X, title="Signal"):
    """Static stem plots of a sampled signal and its (centred) DFT amplitude."""
    fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(10, 5.4), layout="constrained")
    stem(ax_t, t, x)
    ax_t.set_xlabel("$t_n$ [s]")
    ax_t.set_ylabel("$x_n$")
    ax_t.set_title(f"{title}: {x.size} samples")
    order = np.argsort(freqs)
    stem(ax_f, freqs[order], np.abs(X[order]) / x.size)
    ax_f.set_xlabel("$f_k$ [Hz]")
    ax_f.set_ylabel("$|X_k|/N$")
    ax_f.set_title("DFT amplitude")
    plt.show()


def sine_mixer(dft, fs=64, T=1.0):
    """Mix up to three sampled sine waves and look at the DFT of the sum."""
    N = int(fs * T)
    tn = np.arange(N) / fs
    tc = np.linspace(0, T, 1000, endpoint=False)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1 / fs))
    noise = np.random.default_rng(0).normal(size=N)
    colors = [BLUE, ORANGE, AQUA]
    fig = _figure((10, 6.4))
    ax_t, ax_f = fig.subplots(2, 1)

    def update(A1, f1, A2, f2, A3, f3, sigma):
        comps = [(A, f, c) for A, f, c in zip((A1, A2, A3), (f1, f2, f3), colors)]
        x = sum(A * np.sin(2 * np.pi * f * tn) for A, f, _ in comps) + sigma * noise
        X = np.fft.fftshift(dft(x)) / N

        ax_t.cla()
        for j, (A, f, c) in enumerate(comps, 1):
            if A > 0:
                ax_t.plot(tc, A * np.sin(2 * np.pi * f * tc), color=c, lw=1, alpha=0.6,
                          label=f"$A_{j}\\sin(2\\pi f_{j} t)$")
        stem(ax_t, tn, x, color=INK, markersize=4)
        ax_t.plot([], [], "o", color=INK, ms=4, label="samples $x_n$ (sum + noise)")
        ax_t.set_xlim(0, T)
        ax_t.set_ylim(-4, 5)
        ax_t.set_xlabel("$t$ [s]")
        ax_t.legend(loc="upper right", ncols=4, fontsize=8)
        ax_t.set_title(f"$N = {N}$ samples at $f_s = {fs}$ Hz over $T = {T:g}$ s")

        ax_f.cla()
        for A, f, c in comps:
            if A > 0 and f > 0:
                ax_f.vlines([-f, f], 0, A / 2, color=c, lw=6, alpha=0.25)
        stem(ax_f, freqs, np.abs(X), color=INK, markersize=4)
        ax_f.set_xlim(-fs / 2, fs / 2)
        ax_f.set_ylim(0, 1.1)
        ax_f.set_xlabel(f"$f_k$ [Hz]  (bin spacing $\\Delta f = 1/T = {1 / T:g}$ Hz, Nyquist $= {fs / 2:g}$ Hz)")
        ax_f.set_ylabel("$|X_k|/N$")
        ax_f.set_title("DFT amplitude: each sine gives a pair of spikes of height $A_j/2$ at $\\pm f_j$")

    s = _slider_style("260px")
    lab = lambda txt, c: f"<span style='color:{c}'><b>{txt}</b></span>"
    controls = {}
    for j, (A0, f0), c in zip((1, 2, 3), ((1.0, 2), (0.0, 5), (0.0, 12)), colors):
        controls[f"A{j}"] = widgets.FloatSlider(A0, min=0, max=2, step=0.05,
                                                description=lab(f"<i>A</i><sub>{j}</sub>", c), **s)
        controls[f"f{j}"] = widgets.IntSlider(f0, 0, int(fs / 2) - 1,
                                              description=lab(f"<i>f</i><sub>{j}</sub> [Hz]", c), **s)
    controls["sigma"] = widgets.FloatSlider(0, min=0, max=1, step=0.05, description="noise <i>σ</i>", **s)
    c = list(controls.values())
    _app(fig, update, controls, rows=[c[0:2], c[2:4], c[4:6], c[6:]])


def aliasing(T=1.0):
    """Sample a cosine at different rates and see its DFT."""
    tc = np.linspace(0, T, 2000)
    fig = _figure((10, 6))
    ax_t, ax_f = fig.subplots(2, 1)

    def update(f0, fs):
        N = int(fs * T)
        tn = np.arange(N) / fs
        x = np.cos(2 * np.pi * f0 * tn)
        f_alias = abs((f0 + fs / 2) % fs - fs / 2)

        ax_t.cla()
        ax_t.plot(tc, np.cos(2 * np.pi * f0 * tc), color=GREY, lw=1.2, label=f"true signal ({f0} Hz)")
        if not np.isclose(f_alias, f0):
            ax_t.plot(tc, np.cos(2 * np.pi * f_alias * tc), color=ORANGE, ls="--", lw=1.5,
                      label=f"alias ({f_alias:g} Hz)")
        stem(ax_t, tn, x, color=BLUE, markersize=5)
        ax_t.set_ylim(-1.3, 1.6)
        ax_t.set_xlim(0, T)
        ax_t.set_xlabel("$t$ [s]")
        ax_t.legend(loc="upper right", ncols=2, fontsize=9)
        ax_t.set_title(f"{N} samples at $f_s = {fs}$ Hz  (Nyquist frequency $f_s/2 = {fs / 2:g}$ Hz)")

        X = np.fft.fftshift(np.fft.fft(x)) / N
        freqs = np.fft.fftshift(np.fft.fftfreq(N, 1 / fs))
        ax_f.cla()
        ax_f.axvspan(-fs / 2, fs / 2, color=BLUE, alpha=0.07, lw=0)
        for m in (-3, -2, -1, 1, 2, 3):
            ax_f.vlines(freqs + m * fs, 0, np.abs(X), color=GREY, lw=1, alpha=0.6)
        stem(ax_f, freqs, np.abs(X), color=BLUE)
        for s in (-1, 1):
            ax_f.axvline(s * f0, color=ORANGE, ls=":", lw=1.5)
        ax_f.set_xlim(-45, 45)
        ax_f.set_ylim(0, 1.1)
        ax_f.set_xlabel("$f$ [Hz]")
        ax_f.set_ylabel(r"$|X_k|/N$")
        ax_f.set_title("DFT (blue band) and its periodic copies every $f_s$ (grey); dotted = true $\\pm f_0$")

    controls = dict(
        f0=widgets.IntSlider(3, 0, 30, description="signal frequency <i>f</i><sub>0</sub> [Hz]", **_slider_style("380px")),
        fs=widgets.IntSlider(20, 4, 40, description="sampling rate <i>f</i><sub>s</sub> [Hz]", **_slider_style("380px")),
    )
    _app(fig, update, controls)


def leakage(N=64, pad=16, weak_db=-40, weak_offset=8):
    """Spectral leakage for non-integer numbers of cycles, and the effect of windows."""
    n = np.arange(N)
    windows = {"Rectangular (none)": np.ones(N), "Hann": np.hanning(N), "Blackman": np.blackman(N)}
    fig = _figure((10, 6.4))
    ax_t, ax_f = fig.subplots(2, 1)

    def update(cycles, window, weak):
        w = windows[window]
        x = np.cos(2 * np.pi * cycles * n / N)
        if weak:
            x = x + 10 ** (weak_db / 20) * np.cos(2 * np.pi * (cycles + weak_offset) * n / N)
        x = x * w
        ax_t.cla()
        ext = np.arange(-N // 2, N + N // 2)
        ax_t.plot(ext, np.tile(x, 3)[N // 2: N // 2 + 2 * N], ".", color=GREY, ms=4,
                  label="periodic copies the DFT implicitly assumes")
        ax_t.plot(n, x, "o-", color=BLUE, ms=4, lw=1, label="windowed samples $x_n$")
        ax_t.axvspan(-0.5, N - 0.5, color=BLUE, alpha=0.05, lw=0)
        ax_t.set_ylim(-1.3, 1.6)
        ax_t.set_xlabel("sample $n$")
        ax_t.legend(loc="upper right", ncols=2, fontsize=9)
        ax_t.set_title(f"{cycles:.2f} cycles in the window of $N = {N}$ samples, {window} window")

        dense = np.abs(np.fft.rfft(x, pad * N))
        coarse = np.abs(np.fft.rfft(x))
        ref = dense.max()
        db = lambda a: 20 * np.log10(np.maximum(a / ref, 1e-12))
        ax_f.cla()
        ax_f.plot(np.arange(dense.size) / pad, db(dense), color=GREY, lw=1.2,
                  label="underlying continuous spectrum (zero-padded)")
        ax_f.vlines(np.arange(coarse.size), -120, db(coarse), color=BLUE, lw=1.2)
        ax_f.plot(np.arange(coarse.size), db(coarse), "o", color=BLUE, ms=4, label="DFT bins $|X_k|$")
        ax_f.axvline(cycles, color=ORANGE, ls=":", lw=1.5, label="true frequency")
        if weak:
            ax_f.plot([cycles + weak_offset], [weak_db], "v", color=ORANGE, ms=9,
                      label=f"weak tone ({weak_db} dB)")
        ax_f.set_ylim(-100, 8)
        ax_f.set_xlim(0, N // 2)
        ax_f.set_xlabel("frequency bin $k$  (bin $k$ = exactly $k$ cycles per window, $f_k = k/T$)")
        ax_f.set_ylabel("dB relative to peak")
        ax_f.legend(loc="upper right", fontsize=9, ncols=2)

    controls = dict(
        cycles=widgets.FloatSlider(8, min=2, max=24, step=0.05, description="cycles in window",
                                   **_slider_style()),
        window=widgets.Dropdown(options=list(windows), description="window"),
        weak=widgets.Checkbox(False, description=f"add a weak tone ({weak_db} dB, {weak_offset} bins higher)",
                              indent=False),
    )
    keys = list(controls.values())
    _app(fig, update, controls, rows=[keys[:2], keys[2:]])


def audio_filter(t, y, fs, bandpass, clean=None, zoom=0.03):
    """Band-pass filter a noisy recording in Fourier space and listen to the result."""
    freqs = np.fft.rfftfreq(y.size, 1 / fs)
    Y = np.abs(np.fft.rfft(y)) / y.size
    fig = _figure((10, 6))
    ax_f, ax_t = fig.subplots(2, 1)
    mask = t < zoom

    ax_f.semilogy(freqs, Y, color=GREY, lw=0.8)
    ax_f.set_xlim(0, 1500)
    ax_f.set_ylim(Y.max() * 1e-5, Y.max() * 3)
    ax_f.set_xlabel("$f$ [Hz]")
    ax_f.set_ylabel("$|Y(f)|$")
    band = [ax_f.axvspan(0, 1, color=AQUA, alpha=0.2, lw=0)]

    def update(band_hz):
        lo, hi = band_hz
        filtered = bandpass(y, fs, lo, hi)
        band.pop().remove()
        band.append(ax_f.axvspan(lo, hi, color=AQUA, alpha=0.2, lw=0))
        ax_f.set_title(f"Amplitude spectrum; shaded = pass band {lo:.0f} to {hi:.0f} Hz")
        ax_t.cla()
        ax_t.plot(t[mask] * 1e3, y[mask], color=GREY, lw=1, label="noisy recording")
        if clean is not None:
            ax_t.plot(t[mask] * 1e3, clean[mask], color=ORANGE, lw=1.5, ls="--", label="clean chord")
        ax_t.plot(t[mask] * 1e3, filtered[mask], color=BLUE, lw=1.8, label="filtered")
        ax_t.set_xlabel("$t$ [ms]")
        ax_t.legend(loc="upper right", ncols=3, fontsize=9)
        player_filtered.value = _wav(filtered)

    _wav = lambda signal: Audio(signal, rate=fs).data
    player_noisy = widgets.Audio(value=_wav(y), format="wav", autoplay=False, loop=False)
    player_filtered = widgets.Audio(value=_wav(y), format="wav", autoplay=False, loop=False)
    players = widgets.HBox([
        widgets.VBox([widgets.HTML("<b>Noisy recording</b>"), player_noisy]),
        widgets.VBox([widgets.HTML("<b>Filtered</b>"), player_filtered],
                     layout=widgets.Layout(margin="0 0 0 24px")),
    ])

    controls = dict(
        band_hz=widgets.FloatRangeSlider(
            value=(0, 1500), min=0, max=1500, step=10, description="pass band [Hz]",
            continuous_update=False, **_slider_style("520px"),
        )
    )
    _app(fig, update, controls, below=[players])


def plane_wave(N=64):
    """A single 2D Fourier mode in real and Fourier space."""
    y, x = np.mgrid[:N, :N]
    lim = 16
    fig = _figure((10, 4.6))
    ax_r, ax_f = fig.subplots(1, 2)
    im_r = _image(ax_r, np.zeros((N, N)), cmap="gray", vmin=-1, vmax=1, origin="lower")
    im_f = ax_f.imshow(np.zeros((N, N)), cmap="magma", vmin=0, vmax=1, origin="lower",
                       extent=[-N / 2 - 0.5, N / 2 - 0.5, -N / 2 - 0.5, N / 2 - 0.5])
    ax_f.set_xlim(-lim, lim)
    ax_f.set_ylim(-lim, lim)
    ax_f.set_xlabel("$k_x$ [cycles per image]")
    ax_f.set_ylabel("$k_y$ [cycles per image]")
    ax_r.set_xlabel("$x$ [px]")
    ax_r.set_ylabel("$y$ [px]")
    ax_f.grid(False)
    ax_f.axhline(0, color="white", lw=0.5, alpha=0.4)
    ax_f.axvline(0, color="white", lw=0.5, alpha=0.4)
    arrows = []

    def update(kx, ky, phase):
        img = np.cos(2 * np.pi * (kx * x + ky * y) / N + phase)
        F = np.abs(np.fft.fftshift(np.fft.fft2(img))) / (N * N)
        im_r.set_data(img)
        im_f.set_data(F / max(F.max(), 1e-12))
        while arrows:
            arrows.pop().remove()
        if kx or ky:
            arrows.append(ax_f.annotate("", xy=(kx, ky), xytext=(0, 0),
                                        arrowprops=dict(arrowstyle="->", color=AQUA, lw=2)))
            k = np.hypot(kx, ky)
            arrows.append(ax_r.annotate("", xy=(N / 2 + 12 * kx / k, N / 2 + 12 * ky / k),
                                        xytext=(N / 2, N / 2),
                                        arrowprops=dict(arrowstyle="->", color=AQUA, lw=2.5)))
            ax_r.set_title(f"wavelength = {N / k:.1f} px, direction = {np.degrees(np.arctan2(ky, kx)):.0f}°")
        else:
            ax_r.set_title("constant image (zero frequency)")
        ax_f.set_title(f"$|F|$: spikes at $\\pm(k_x, k_y) = \\pm({kx}, {ky})$\\n"
                       f"i.e. $(u, v) = \\pm({kx / N:.3f}, {ky / N:.3f})$ cycles/px", fontsize=10)

    s = _slider_style("300px")
    controls = dict(
        kx=widgets.IntSlider(4, -lim, lim, description="<i>k</i><sub>x</sub>", **s),
        ky=widgets.IntSlider(2, -lim, lim, description="<i>k</i><sub>y</sub>", **s),
        phase=widgets.FloatSlider(0, min=0, max=2 * np.pi, step=0.1, description="phase <i>φ</i>", **s),
    )
    _app(fig, update, controls)


def _shape(name, N, size, angle, shift):
    y, x = np.mgrid[:N, :N] - N / 2
    x = x - shift
    a = np.radians(angle)
    xr = x * np.cos(a) + y * np.sin(a)
    yr = -x * np.sin(a) + y * np.cos(a)
    if name == "Rectangle":
        return ((np.abs(xr) < size) & (np.abs(yr) < size / 3)).astype(float)
    if name == "Disc":
        return (np.hypot(xr, yr) < size).astype(float)
    if name == "Gaussian":
        return np.exp(-(xr**2 + (yr * 2) ** 2) / (2 * (size / 2) ** 2))
    if name == "Slit (line)":
        return ((np.abs(xr) < 1.5) & (np.abs(yr) < 3 * size)).astype(float)
    if name == "Lattice of dots":
        p = max(size / 2, 3)
        xm = (xr + p / 2) % p - p / 2
        ym = (yr + p / 2) % p - p / 2
        return np.exp(-(xm**2 + ym**2) / 2) * np.exp(-(x**2 + y**2) / (2 * (N / 5) ** 2))
    raise ValueError(name)


def shapes_2d(N=256):
    """Simple shapes and their 2D Fourier transforms."""
    zoom = N // 4
    fig = _figure((10, 4.6))
    ax_r, ax_f = fig.subplots(1, 2)
    im_r = _image(ax_r, np.zeros((N, N)), vmin=0, vmax=1, origin="lower")
    ext = [-N / 2 - 0.5, N / 2 - 0.5, -N / 2 - 0.5, N / 2 - 0.5]
    im_f = ax_f.imshow(np.zeros((N, N)), cmap="magma", extent=ext, vmin=0, vmax=1, origin="lower")
    ax_f.set_xlim(-zoom, zoom)
    ax_f.set_ylim(-zoom, zoom)
    ax_f.grid(False)
    ax_f.set_xlabel("$k_x$")
    ax_f.set_ylabel("$k_y$")
    ax_f.set_title(r"$\log(1+|F|)$, normalised (central region)")

    def update(shape, size, angle, shift):
        img = _shape(shape, N, size, angle, shift)
        F = _log_amp(np.fft.fftshift(np.fft.fft2(img)))
        im_r.set_data(img / img.max())
        im_f.set_data(F / F.max())
        ax_r.set_title(f"{shape}: size {size}, angle {angle}°, shift {shift} px")

    s = _slider_style("300px")
    controls = dict(
        shape=widgets.Dropdown(options=["Rectangle", "Disc", "Gaussian", "Slit (line)", "Lattice of dots"],
                               description="Shape"),
        size=widgets.IntSlider(20, 4, 60, description="size", **s),
        angle=widgets.IntSlider(0, 0, 180, step=5, description="rotation [°]", **s),
        shift=widgets.IntSlider(0, -60, 60, step=2, description="shift x [px]", **s),
    )
    keys = list(controls.values())
    _app(fig, update, controls, rows=[keys[:2], keys[2:]])


def image_filter(img, radial_mask, fourier_filter):
    """Low-, high- and band-pass filtering of an image in Fourier space."""
    F = _log_amp(np.fft.fftshift(np.fft.fft2(img)))
    F /= F.max()
    fig = _figure((10, 4.8))
    ax_f, ax_r = fig.subplots(1, 2)
    im_f = ax_f.imshow(F, cmap="magma", extent=[-0.5, 0.5, 0.5, -0.5], vmin=0, vmax=1)
    ax_f.grid(False)
    ax_f.set_xlabel("$u$ [cycles / pixel]")
    ax_f.set_ylabel("$v$ [cycles / pixel]")
    im_r = _image(ax_r, img)

    def update(kind, radii, smooth):
        r_lo, r_hi = radii
        if kind == "Low-pass":
            r_lo = 0.0
        elif kind == "High-pass":
            r_hi = 1.0
        mask = radial_mask(img.shape, r_lo, r_hi, smooth)
        out = fourier_filter(img, mask)
        im_f.set_data(F * np.fft.fftshift(mask))
        im_r.set_data(out)
        if kind == "High-pass":
            v = np.percentile(np.abs(out), 99.5)
            im_r.set_clim(-v, v)
        else:
            im_r.set_clim(img.min(), img.max())
        ax_f.set_title(f"Kept frequencies: {r_lo:.3f} < r < {min(r_hi, 0.707):.3f}")
        ax_r.set_title(f"{kind} filtered image")

    controls = dict(
        kind=widgets.ToggleButtons(options=["Low-pass", "High-pass", "Band-pass"], description="Filter"),
        radii=widgets.FloatRangeSlider(value=(0.02, 0.08), min=0, max=0.5, step=0.005,
                                       readout_format=".3f", description="radii [cycles/px]",
                                       **_slider_style("460px")),
        smooth=widgets.Checkbox(False, description="smooth (Gaussian) edges"),
    )
    keys = list(controls.values())
    _app(fig, update, controls, rows=[keys[:1], keys[1:]])


def compression(img, keep_largest):
    """Reconstruct an image from only its largest Fourier coefficients."""
    F = np.fft.fft2(img)
    fig = _figure((11, 4))
    ax_m, ax_r, ax_e = fig.subplots(1, 3)
    im_m = _image(ax_m, np.zeros(img.shape), cmap="magma", vmin=0, vmax=1)
    im_r = _image(ax_r, img, vmin=img.min(), vmax=img.max())
    im_e = _image(ax_e, img, cmap="RdBu_r", vmin=-0.2, vmax=0.2)
    ax_e.set_title("residual (original - reconstruction)")

    def update(fraction):
        F_kept = keep_largest(F, fraction)
        rec = np.fft.ifft2(F_kept).real
        im_m.set_data(np.fft.fftshift(F_kept != 0).astype(float))
        im_r.set_data(rec)
        im_e.set_data(img - rec)
        rmse = np.sqrt(np.mean((img - rec) ** 2))
        ax_m.set_title(f"kept coefficients ({100 * fraction:.2g}%)")
        ax_r.set_title(f"reconstruction, RMSE = {rmse:.3f}")

    controls = dict(
        fraction=widgets.FloatLogSlider(0.02, base=10, min=-3.5, max=0, step=0.05,
                                        description="fraction kept", readout_format=".4f",
                                        **_slider_style("460px"))
    )
    _app(fig, update, controls)


def _kspace_mask(pattern, shape, fraction, seed=0):
    rng = np.random.default_rng(seed)
    ny, nx = shape
    v, u = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="ij")
    r = np.hypot(u, v)
    if pattern == "Low frequencies only":
        return r <= np.sqrt(fraction / np.pi)
    if pattern == "Random points":
        return rng.random(shape) < fraction
    if pattern == "Random + dense centre":
        centre = r <= np.sqrt(0.25 * fraction / np.pi)
        return centre | (rng.random(shape) < 0.75 * fraction)
    if pattern == "Radial spokes":
        n_spokes = max(1, int(fraction * nx * 0.8))
        theta = np.arctan2(v, u) % np.pi
        d = np.abs((theta[..., None] - np.linspace(0, np.pi, n_spokes, endpoint=False) + np.pi / 2) % np.pi - np.pi / 2)
        return np.any(d * np.maximum(r, 1e-9)[..., None] * nx < 0.6, axis=-1) | (r == 0)
    if pattern == "Every n-th row":
        step = max(1, int(round(1 / fraction)))
        return np.broadcast_to((np.arange(ny) % step == 0)[:, None], shape)
    raise ValueError(pattern)


def kspace_sampling(img):
    """Reconstruct an image from an incomplete set of Fourier samples."""
    F = np.fft.fft2(img)
    fig = _figure((10, 4.8))
    ax_m, ax_r = fig.subplots(1, 2)
    im_m = _image(ax_m, np.zeros(img.shape), cmap="gray", vmin=0, vmax=1)
    im_r = _image(ax_r, img, vmin=0, vmax=img.max())

    def update(pattern, fraction):
        mask = _kspace_mask(pattern, img.shape, fraction)
        rec = np.abs(np.fft.ifft2(F * mask))
        im_m.set_data(np.fft.fftshift(mask).astype(float))
        im_r.set_data(rec)
        im_r.set_clim(0, np.percentile(rec, 99.8))
        ax_m.set_title(f"sampled Fourier coefficients ({100 * mask.mean():.1f}%)")
        ax_r.set_title("zero-filled reconstruction $|\\mathcal{F}^{-1}(M \\cdot F)|$")

    controls = dict(
        pattern=widgets.Dropdown(
            options=["Low frequencies only", "Random points", "Random + dense centre", "Radial spokes",
                     "Every n-th row"],
            description="Sampling",
        ),
        fraction=widgets.FloatSlider(0.15, min=0.02, max=0.6, step=0.01, description="fraction",
                                     **_slider_style("380px")),
    )
    _app(fig, update, controls)
