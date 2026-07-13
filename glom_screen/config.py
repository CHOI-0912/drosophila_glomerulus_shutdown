"""Paths and constants for the glomerulus lesion screen."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EDGELIST = ROOT / 'banc_888_edgelist_simple_v2.feather'
META = ROOT / 'banc_888_meta.feather'
SEEDS_TXT = ROOT / 'orn.txt'
GROUPS_CSV = ROOT / 'neuron.csv'

RESULTS = ROOT / 'glom_screen_results'
CACHE = RESULTS / 'cache'

# InfluenceCalculator construction (matches the BANC defaults used so far)
COUNT_THRESH = 5
SIGNED = False
LAMBDA_MAX = 0.99

# Top-N sweep.  N=500 is the headline; the rest establish how much the
# ranking depends on that choice.
N_SWEEP = (50, 100, 250, 500, 1000)
N_HEADLINE = 500

# Silence-target variants.  'all' is neuron.csv as given (minus seeds);
# 'pn' keeps only the glomerulus' projection neurons.  See the plan: LNs
# sit in ~9 glomeruli each, so 'all' lesions are not glomerulus-specific.
PN_CELL_CLASS = 'antennal_lobe_projection_neuron'
VARIANTS = ('pn', 'all')

# Metrics may drift marginally outside [0, 1] because alpha is recomputed
# on the lesioned matrix (measured at +0.68% for a 9% lesion; these lesions
# are <=154 of 150,299 neurons, so the drift is far smaller).  Clip, but
# shout if anything lands beyond this.
CLIP_TOL = 1e-6
