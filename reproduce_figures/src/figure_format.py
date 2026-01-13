"""Format the font and axis labels.
"""
import matplotlib.pyplot as plt

def get_font_parameters():
    tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "Computer Modern",
    "mathtext.fontset" : "custom",
    "mathtext.rm" : "Bitstream Vera Sans",
    "mathtext.it": "Bitstream Vera Sans:italic",
    "mathtext.bf" : "Bitstream Vera Sans:bold",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 9,
    "font.size": 9,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    # Make title and subplot labels bold
    "axes.titleweight": "bold",
    "axes.labelweight": "bold"
    }
    return tex_fonts

def set_matplotlib_defaults():
    plt.rcParams.update({
        "savefig.dpi": 1000,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })