import matplotlib as mpl


PALETTE_HEATMAP = ('#5f5f47', '#8d8d7c', '#d3d3c5', '#f1f1f1', '#f1f1f1', '#d3d3c5', '#8d8d7c', '#5f5f47')
PALETTE = ('#7eb0d5','#fd7f6f', '#b2e061','#bd7ebe','#ffb55a','#ffee65','#beb9db','#fdcce5','#8bd3c7')

DEFAULT_FONT_COLOR = '#909090'



def set_styles():
    # remove spines
    for s in ['top', 'bottom', 'left', 'right']:
        mpl.rcParams[f'axes.spines.{s}'] = False

    # ax 
    mpl.rcParams['axes.grid'] = False
    mpl.rcParams['axes.labelcolor'] = DEFAULT_FONT_COLOR
    mpl.rcParams['axes.labelsize'] = 'small'
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', PALETTE)
    mpl.rcParams['axes.titlelocation'] = 'left'
    mpl.rcParams['axes.titlesize'] = 18
    mpl.rcParams['axes.titleweight'] = 'bold'
    mpl.rcParams['axes.titlecolor'] = DEFAULT_FONT_COLOR

    mpl.rcParams['font.size'] = 7

    mpl.rcParams['grid.alpha'] = 0.3
    mpl.rcParams['grid.color'] = DEFAULT_FONT_COLOR
    mpl.rcParams['grid.linestyle'] = '--'

    mpl.rcParams['text.color'] = DEFAULT_FONT_COLOR
    mpl.rcParams['xtick.color'] = DEFAULT_FONT_COLOR
    mpl.rcParams['ytick.color'] = DEFAULT_FONT_COLOR
    mpl.rcParams['figure.facecolor'] = 'white'

    mpl.rcParams['xtick.major.size'] = 0             # major tick size in points
    mpl.rcParams['xtick.minor.size'] = 0             # minor tick size in points
    mpl.rcParams['ytick.major.size'] = 0
    mpl.rcParams['ytick.minor.size'] = 0

    mpl.rcParams['legend.frameon'] = False
    
    return



# ---- TEXT ACCENTS COLORS --------------------------------------------

# uncomment to print all 256 possible colors
# for i in range(0, 16):
#     for j in range(0, 16):
#         code = str(i * 16 + j)
#         print(u"\u001b[48;5;" + code + "m " + code.ljust(4), end=' ')
#     print (u"\u001b[0m")
    
TXT_BOLD = '\u001b[1m'
TXT_FOREGROUND = '\u001b[38;5;{}m'.format(254)
TXT_BACKGROUND = '\u001b[48;5;{}m'.format(240)

TXT_RESET = '\u001b[0m'

TXT_ACC = TXT_BOLD + TXT_FOREGROUND + TXT_BACKGROUND

# print(f'\n\n{TXT_ACC} Accented text {TXT_RESET} Normal text\n')





