import numpy as np

# Simulater Setting
# ------------------------------

MINUTES = 60000000000
TIMESTEP = np.timedelta64(10 * MINUTES)
PICKUPTIMEWINDOW = np.timedelta64(1000 * MINUTES)  # neglect reposition node

# It can enable the neighbor car search system to determine the search range according to the set search distance and
# the size of the grid. It use dfs to find the nearest idle vehicles in the area.
NeighborCanServer = False

# You can adjust the size of the experimental area by entering latitude and longitude.
# The order, road network and grid division will be adaptive. Adjust to fit selected area

FocusOnLocalRegion = False
LocalRegionBound = (104.0178, 104.1128, 30.6218, 30.6974)
if FocusOnLocalRegion == False:
    LocalRegionBound = (104.0178, 104.1128, 30.6218, 30.6974)

# Input parameters
VehiclesNumber = 500
SideLengthMeter = 1200
VehiclesServiceMeter = 800

DispatchMode = "Simulation"
DemandPredictionMode = "None"
ClusterMode = "Grid"

