import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class Track:
    """
    define track parameters and geometry.
    initialized with default J-turn parameters or loaded from a CSV.
    """
    # Default J-Turn Parameters define from the doc
    straight_length: float = 500.0
    radius: float = 50.0
    ds: float = 0.5  # Distance step resolution to sample track geo
    
    # Internal storage for track points to display later
    x_coords: np.ndarray = field(default_factory=lambda: np.array([]))
    y_coords: np.ndarray = field(default_factory=lambda: np.array([]))
    s_points: np.ndarray = field(default_factory=lambda: np.array([]))
    radii: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def __post_init__(self):
        # Automatically generates J-turn if a costom track isn't loaded
        if len(self.x_coords) == 0:
            self.generate_j_turn()
    

    @property
    def total_length(self) -> float:
        return self.s_points[-1] if (len(self.s_points) > 0) else 0.0
    

    def generate_j_turn(self):
        """Generates the standard J-shape track based on class attributes."""
        # Derived lengths
        # matches napkin math for  100 pi / 2 + 500 = 657.07m for J turn
        curve_len = np.pi * self.radius
        total_len = self.straight_length + curve_len
        
        # Generate distance steps
        self.s_points = np.arange(0, total_len, self.ds)
        
        x_list = []
        y_list = []
        r_list = []

        for s in self.s_points:
            # Radius Logic for J turn 
            if s <= self.straight_length: 
                current_r = np.inf # y does not change, x does
                x_list.append(s)
                y_list.append(0)
            else:
                current_r = self.radius
                # Curve coordinates
                dist_into_curve = s - self.straight_length  # get rid of straight length to calculate remainding for curve

                # Angle calculation (Left turn: 90 deg start to -90 deg end)
                angle = (np.pi/2) - (dist_into_curve / self.radius) # get angle starting at pi/2 then move down to 0
                
                # Using Center (straight_len, radius)
                x = self.straight_length + self.radius * np.cos(angle - np.pi/2) # Standard circle param adjustments
                
                x = self.straight_length + self.radius * np.cos(angle)  
                y = -self.radius + self.radius * np.sin(angle)       

                x_list.append(x)
                y_list.append(y)
            
            r_list.append(current_r)
            
        #  put into array to plot on mathplotlib
        self.x_coords = np.array(x_list)
        self.y_coords = np.array(y_list)
        self.radii = np.array(r_list)

        """
            Loads track points from a CSV file.
            Should expect columns named 'x' and 'y'
            Does not work. It reads for x and y values which is not what the other path files containted, it contains a section length and radius. Didn't think through math for this one yet
        """
    def load_from_csv(self, file_path: str):
        try:
            df = pd.read_csv(file_path)
            
            # Strip Upppercase column names to lowercase to find x/y easily
            df.columns = [c.lower().strip() for c in df.columns]
            
            if 'x' not in df.columns or 'y' not in df.columns:
                raise ValueError("CSV must contain 'x' and 'y' columns.") # throw error if x and y don't exist
                
            self.x_coords = df['x'].values # put into array to plot on mathplotlib
            self.y_coords = df['y'].values
            
            # Recalc s_points and radii based on the new x/y coordinates for better plot
            self._recalculate_path_geometry()
            
            print(f"[Track] Loaded custom track from {file_path}. Points: {len(self.x_coords)}")
            
        except Exception as e:
            print(f"[Error] Could not load track: {e}")
            print("Reverting to default J-turn.")
            self.generate_j_turn()


    def _recalculate_path_geometry(self):
        # Calculate distance between points
        dx = np.diff(self.x_coords)
        dy = np.diff(self.y_coords)
        distances = np.sqrt(dx**2 + dy**2) # put all dx and dy into array for every ds
        
        # Cumulative distance (s)
        self.s_points = np.zeros(len(self.x_coords))
        self.s_points[1:] = np.cumsum(distances)
        
        # Calculate Curvature (k = 1/R) using 3-point circumcircle or gradient method
        # Simple method: d(heading)/ds
        headings = np.arctan2(dy, dx)
        # Unwrap angles to prevent jumps from pi to -pi
        headings = np.unwrap(headings)
        
        d_heading = np.diff(headings)
        # Pad d_heading to match length
        d_heading = np.append(d_heading, d_heading[-1])
        
        # Pad distances for division
        dist_padded = np.append(distances, distances[-1])
        
        # Curvature k = d_heading / ds
        curvature = d_heading / dist_padded
        
        # Radius r = 1 / k
        # Handle straight lines (k close to 0)
        with np.errstate(divide='ignore'):
            self.radii = 1 / curvature
            self.radii[np.abs(curvature) < 1e-4] = np.inf


    def plot_track(self, time: float, energy: float):
        # Track Map
        plt.subplot(2, 2, 1)
        plt.plot(self.x_coords, self.y_coords, 'k-')
        plt.title("Track Map")
        plt.axis('equal')
        plt.grid(True)

        # Display track length, time, and energy on the plot
        plt.text(0.02,
            0.98, 
            f"Track Length: {self.total_length:.2f} m\nTime: {time:.2f} s\nEnergy: {energy:.2f} J",
            transform = plt.gca().transAxes,  # define coordinate system 
            va = 'top',  # Keep within plot area
            ha = 'left',
            bbox = dict(boxstyle="round", facecolor="white", alpha=0.8) # Color and define box type
        )

"""
    Testing for ensuring track generation is working as expected 
"""

# track = Track(straight_length=500, radius=50) 
# track.load_from_csv("FSAE Autocross Nebraska 2013 - Sheet1.csv") # Try loading custom track, if fails will generate J turn
# track.plot_track()

