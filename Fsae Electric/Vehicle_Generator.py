import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass


# --- Storage container for Vehicle Parameters defined by doc ---
@dataclass
class VehicleParams:
    mass: float = 300.0           # kg
    tire_radius: float = 0.254    # defined in meters
    drive_ratio: float = 4.0      # Gear ratio
    efficiency: float = 0.90      # Powertrain efficiency
    mu: float = 1.0               # Friction coefficient for tires
    g: float = 9.81               # Gravity -> for converting to Gs -> standard unit for racing performance?
    cd: float = 0.8               # Drag coefficient
    frontal_area: float = 1.0     # m^2
    rho: float = 1.225            # Air density at sea level assumption for drag calculations


