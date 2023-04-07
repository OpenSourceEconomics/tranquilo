from pathlib import Path

import plotly.express as px

DOCS_DIR = Path(__file__).parent.parent / "docs"

EXAMPLE_DIR = Path(__file__).parent / "examples"

TEST_FIXTURES_DIR = Path(__file__).parent.parent.parent / "tests" / "fixtures"


PLOTLY_TEMPLATE = "simple_white"
PLOTLY_PALETTE = px.colors.qualitative.Set2

DEFAULT_N_CORES = 1

CRITERION_PENALTY_SLOPE = 0.1
CRITERION_PENALTY_CONSTANT = 100


# =================================================================================
# Dashboard Defaults
# =================================================================================

Y_RANGE_PADDING = 0.05
Y_RANGE_PADDING_UNITS = "absolute"
PLOT_WIDTH = 750
PLOT_HEIGHT = 300
MIN_BORDER_LEFT = 50
MIN_BORDER_RIGHT = 50
MIN_BORDER_TOP = 20
MIN_BORDER_BOTTOM = 50
TOOLBAR_LOCATION = None
GRID_VISIBLE = False
MINOR_TICK_LINE_COLOR = None
MAJOR_TICK_OUT = 0
MINOR_TICK_OUT = 0
MAJOR_TICK_IN = 0
OUTLINE_LINE_WIDTH = 0
LEGEND_LABEL_TEXT_FONT_SIZE = "11px"
LEGEND_SPACING = -2
