import matplotlib
from collections import OrderedDict
import numpy as np
import scipy.constants as cnt
# Figure size A6
figsize=(297/25.4,210/25.4)
# Figure size A5
figsize5=(210/25.4,148.5/25.4)
# Constants
C0=cnt.c
e0=cnt.e
Pi=np.pi
h0=cnt.h
nm=1e-9

# DPI
dpi=300

#Nice fonts
fontsize=16

# Plot frame thickness
framethick=1.5


fontparams = {
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Arial",
    "font.size": fontsize,
    'mathtext.fontset': 'custom',
    'mathtext.it': 'Arial:italic',
    'mathtext.rm': 'Arial',
    "text.latex.preamble": r"""
        \usepackage{amsmath}
        \usepackage{sansmath}   % Enables sans-serif math
        \sansmath               % Apply sans-serif math to the whole document
        \renewcommand{\familydefault}{\sfdefault}  % Set default family to sans-serif
        \usepackage{helvet}     % Set font to Helvetica (Arial is very similar to Helvetica)
        \renewcommand{\rmdefault}{phv} % Override default text font
    """
}

matplotlib.rcParams.update(fontparams)



cells = [
     ['SP-E2', 'Flat', '#727872', 18.5],
     ['P2-H7', 'Periodic', '#1E7E42', 27.5],
     ['HO-C5', 'Holes', '#00429d', 27.8],
     ['HC-B2', 'Honeycomb', '#fd9291', 28.2],
   # ['HC-D5', 'Honeycomb', '#fd9291'],
    # ['SN-C2', 'SiN','#c52a52', 21.4],
    ]

cell_colors={
'Flat': '#727872',
'Holes': '#00429d',
'HUD Holes': '#00429d',
'Periodic': '#1E7E42', 
'Honeycomb': '#fd9291',
'HUD Walls': '#fd9291',
'SiN': '#c52a52',
'Simulation': '#6E2B1D',
}

shade_colors=['#F1B4B4', '#7FA0CE','#B29696']
shade_colors=['#F55A5A', '#208484','#8B6F6F']
shade_colors=['#F1B4B4', '#7FA0CE','#EBEBEB']
heatmap_shade='#CDCDCD'
modeax_color='#669999'
lcol=[ '#A41717', 'black', '#3C3319']

def update_fontsize(fontsize):
	fontparams = {
		"text.usetex": True,
		"font.family": "sans-serif",
		"font.sans-serif": "Arial",
		"font.size": fontsize,
		'mathtext.fontset': 'custom',
		'mathtext.it': 'Arial:italic',
		'mathtext.rm': 'Arial',
		"text.latex.preamble": r"""
			\usepackage{amsmath}
			\usepackage{sansmath}   % Enables sans-serif math
			\sansmath               % Apply sans-serif math to the whole document
			\renewcommand{\familydefault}{\sfdefault}  % Set default family to sans-serif
			\usepackage{helvet}     % Set font to Helvetica (Arial is very similar to Helvetica)
			\renewcommand{\rmdefault}{phv} % Override default text font
		"""
	}

	matplotlib.rcParams.update(fontparams)


# Define custom formatter function
def custom_formatter(x, pos):
    if x == 0.00 or x == 0.0:
        return '0'
    elif x == 1.00 or x == 1.0:
        return '1'
    elif x == 2.00 or x == 2.0:
        return '2'
    elif x == 3.00 or x == 3.0:
        return '3'
    elif x == 4.00 or x == 4.0:
        return '4'
    elif x == 20.00 or x == 20.0:
        return '20'
    elif x == 40.00 or x == 40.0:
        return '40'
    elif x == 60.00 or x == 60.0:
        return '60'
    elif int(x) == x:
        return '{}'.format(int(x))
    else:
        return f'{x:.1f}'
