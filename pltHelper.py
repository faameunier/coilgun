import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.available
plt.style.use("switch")


class CenteredFormatter(mpl.ticker.ScalarFormatter):
    """Acts exactly like the default Scalar Formatter, but yields an empty
    label for ticks at "center"."""
    center = 0

    def __call__(self, value, pos=None):
        if value == self.center:
            return ''
        else:
            return mpl.ticker.ScalarFormatter.__call__(self, value, pos)


def centerPlt(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.xaxis.set_label_coords(1, 0.45)
    ax.yaxis.set_label_coords(0.6, 0.95)
    ax.yaxis.set_major_formatter(CenteredFormatter())
