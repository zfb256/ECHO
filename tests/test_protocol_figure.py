import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from make_fig_protocol import build, plt


def test_protocol_labels_and_canvas():
    fig = build()
    try:
        fig.canvas.draw()
        ax = fig.axes[0]
        labels = {text.get_text() for text in ax.texts}
        assert {'directed discordance', 'Hunyuan-only placebo gap',
                'Saved outputs from (a)'} <= labels
        assert 'exposure effect' not in labels
        for text in ax.texts:
            bounds = text.get_window_extent(fig.canvas.get_renderer())
            assert ax.bbox.contains(bounds.x0, bounds.y0)
            assert ax.bbox.contains(bounds.x1, bounds.y1)
    finally:
        plt.close(fig)
