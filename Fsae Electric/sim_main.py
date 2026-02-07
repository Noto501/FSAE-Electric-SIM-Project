import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Track_Generator import Track
from Vehicle_Generator import VehicleParams
import pandas as pd

# --- Initialize Vehicle and Track ---
# Instantiate the vehicle object using values from the data class
car = VehicleParams()
track = Track(straight_length=500, radius=50) 

# --- Define Functions and Track ---
# Initialize arrays and their sizes with the variables based on the TRACK object from Track_Generator
ds = 0.5  # distance step
s_points = track.s_points

# 3. Forward Pass (Acceleration Profile) & Final Merge
v_final = np.zeros_like(s_points)
accel_long = np.zeros_like(s_points)
accel_lat = np.zeros_like(s_points)
times = np.zeros_like(s_points)
torques = np.zeros_like(s_points)
powers = np.zeros_like(s_points)
energy_total = 0.0

# Start at 0 speed
v_final[0] = 0

def get_radius(s): # TODO For straight plus curve radius only. Does not work for more complex tracks
    return np.inf if s <= track.straight_length else track.radius

# --- Helper Functions for Powertrain Limits ---

# Load motor curve data from CSV
filename = 'Recruitment E228 Modified Motor Curves - 228.csv'
try:
    df_motor = pd.read_csv(filename)
    
    # Extract the specific columns indicated by the file structure and convert to numpy arrays
    motor_rpm_data = df_motor['Motor Speed (RPM)'].values # x
    motor_torque_data = df_motor['Torque (Capped at 80kW) (Nm)'].values # y
    
    # Sort by RPM to prevent mismatching in lining up data. For every rpm value, there is a corresponding torque value.
    sort_idx = np.argsort(motor_rpm_data)
    motor_rpm_data = motor_rpm_data[sort_idx]
    motor_torque_data = motor_torque_data[sort_idx]

    # Smooth the torque curve using moving average to make graph less noisy
    window_size = 3
    motor_torque_data = np.convolve(motor_torque_data, np.ones(window_size)/window_size, mode='same')
    
    print(f"Successfully loaded motor curve with {len(motor_rpm_data)} points.")
    print(f"Max RPM in data: {motor_rpm_data[-1]:.1f}, Max Torque: {motor_torque_data.max():.1f} Nm")

except FileNotFoundError:
    print(f"Error: Could not find '{filename}'. Using default fallback curve.")
    # Fallback dummy data if file is missing
    motor_rpm_data = np.array([0, 5000, 10000])
    motor_torque_data = np.array([25, 25, 0])

"""
    Returns the available torque at a specific RPM from interpolating loaded CSV data.
            - current_rpm: The RPM we want to know the torque for (the anchor column for sorting the x and y into the right steps)
            - motor_rpm_data: The x-axis data points from CSV
            - motor_torque_data: The y-axis data points from CSV
            - right=0: If RPM exceeds our data size limit (e.g., >5433), assume 0 torque (rev limit)
"""
def get_motor_torque(current_rpm):    
    return np.interp(current_rpm, motor_rpm_data, motor_torque_data, right = None)


"""
    Paragraph explaination so I know what I am talking about ;-;
     Calculates the maximum longitudinal acceleration (in G) 
    the powertrain can produce at a specific velocity v. This is done in terms of the motor, not in terms of the wheel
    It uses the motor torque curve from the csv to find the available torque at the current RPM, then calculates the propulsion force and subtracts drag to find net acceleration. Finally, it converts to Gs.

    f-drag uses the formula (0.5 * rho * A * cd * v^2) 
    Propulsion force is torque / tire radius. 
    Net force is propulsion - drag, and acceleration is net force / mass.
    v should be given in m/s and the car should be provided based on the VehicleParams class.

"""
def get_long_accel_limit(v, car: VehicleParams):
    # Calc Aerodynamic Drag Force (opposing motion) <----
    f_drag = 0.5 * car.rho * car.frontal_area * car.cd * (v**2)
    
    # Calc Motor Force (propelling motion) ---->
    wheel_rpm = (v / car.tire_radius) * (60 / (2 * np.pi))
    motor_rpm = wheel_rpm * car.drive_ratio
    
    motor_torque = get_motor_torque(motor_rpm) # Torque formula with t = L * F emperically measured
    wheel_torque = motor_torque * car.drive_ratio * car.efficiency 
    f_propulsion = wheel_torque / car.tire_radius
    
    # 3. Net Acceleration
    f_net = f_propulsion - f_drag # F = ma equation
    a_long_ms2 = f_net / car.mass
    
    return a_long_ms2 / car.g # Get G by dividing by gravity from acceleration G = a/g

# --- Generate Data for GGV Plot ---

"""
    Paragraph explaination so I know what I am talking about ;-;
    Wanna answer this question that I was interpreting from Office hours ;-;
    "At this specific speed, what is max combo of cornering (Lateral G) and acceleration (Longitudinal G) the car can achieve?"
    aka what speed can we go before we exit the circle

    For every velocity, calc max longitudinal G at that speed using the powertrain limits calc by method above
    combine it with the constant lateral G limit from the tires
    find the max G combo in every direction. 
    This will give us a contour of achievable Gs at that speed that we can plot

    Note:
    Lat_g = side to side
    Long_g = forward and backward
    
"""
#  Define resolution
velocities = np.linspace(0.1, 35, 50)  # 0 to ~35 m/s arbitrary max speed for FSAE car with 50 steps of resolution
angles = np.radians(np.linspace(0, 360, 100)) # 100 point resolution for 0-360 degrees

# Lists to store 3D coordinates
V_3d, Lat_3d, Long_3d = [], [], []

# We will also store 2D slices for specific speeds TODO can add more as the user specifie
contour_speeds = [5, 15, 25, 30] # m/s 
contours = {}

for v in velocities:
    # Physical limit at this speed due to motor/drag
    max_powertrain_g = get_long_accel_limit(v, car)
    
    # Tire limit is constant (defined by mu)
    tire_limit_g = car.mu 
    
    # For every angle, find the radius that satisfies BOTH limits
    current_lat = []
    current_long = []
    
    for theta in angles:
        # Pure tire circle coordinates
        lat_g = tire_limit_g * np.cos(theta) # side to side F
        long_g = tire_limit_g * np.sin(theta) # forward and backward F
        
        # Apply Powertrain Limit (Only affects positive Longitudinal G)
        if long_g > 0:
            # Clamp longitudinal G if motor/aero limits it since there are cases where motor can't go past the acceleration the tires can handle
            long_g = min(long_g, max_powertrain_g)

        current_lat.append(lat_g) # add to list
        current_long.append(long_g)
        
        # Store for 3D plot
        V_3d.append(v)
        Lat_3d.append(lat_g)
        Long_3d.append(long_g)

    # Store contours if this velocity is a target speed
    for cs in contour_speeds:
        if abs(v - cs) < 0.5 and cs not in contours:
            contours[cs] = (current_lat, current_long)


"""
    Paragraph explaination so I know what I am talking about ;-;

    Takes into account the powertrain limits and tire limits to find the max G combo at every angle for a specific speed. which we can do for mapping out to track

    Makes a comparision between the forwards and backwards passs and calculates if we should slow down to make a corner or if we can accelerate more.

    Calculates a final energy cost if our long accel is positive
"""
# --- Simulation Logic ---
# Calculate the Cornering Limit (V_max for every point)
# v = sqrt(mu * g * R) -? From centripital acceleration eq
v_limit = np.sqrt(car.mu * car.g * track.radii)
# Account for when we are on straight, just make radius very large (infinite speed allowed on straights)
v_limit[track.radii == np.inf] = 100000000000.0  # Just a huge number if we are on a straight cuz the calculation doesn't matter

# 2. Backward Pass (Braking Profile)
# We calculate max velocity BACKWARDS assuming 100% braking capability, hence why the next for loop goes backwards. We start at the end of the track and move backwards
v_braking = np.zeros_like(s_points)
v_braking[-1] = v_limit[-1] # End condition for when we loop through everything

for i in range(len(s_points) - 2, -1, -1): 
    ds = track.s_points[i+1] - track.s_points[i]
    
    # use tire friction for the limit and assume air doesn't help brake
    # Max braking deceleration TODO possible future addon to account for drag too?
    a_brake = car.mu * car.g 
    
    # v_i = sqrt(v_{i+1}^2 + 2 * a * ds)
    # We must also respect the static limit at this point
    next_v_sq = v_braking[i+1]**2 + 2 * a_brake * ds
    v_braking[i] = min(v_limit[i], np.sqrt(next_v_sq))

# Forward accel + determining if it's ok with the braking curve
for i in range(1, len(s_points)):
    ds = track.s_points[i] - track.s_points[i-1]
    
    # Part A. Determine Target Speed from Acceleration (Forward Physics)
    current_v = v_final[i-1]
    
    # Drag using drag formula(0.5 * rho * A * cd * v^2)
    f_drag = 0.5 * car.rho * car.frontal_area * car.cd * (current_v**2)
    
    # Motor
    wheel_rpm = (current_v / car.tire_radius) * (60 / (2 * np.pi))
    motor_rpm = wheel_rpm * car.drive_ratio
    motor_torque = get_motor_torque(motor_rpm)
    wheel_torque = motor_torque * car.drive_ratio * car.efficiency # TODO these are repeted calculations. can try making methods to store these better
    f_motor_propulsion = wheel_torque / car.tire_radius
    
    # Grip Limit (Longitudinal)
    f_lat_current = 0 # On straight, lat is 0. In corner, we are speed limited.
    
    f_propulsion = min(f_motor_propulsion, car.mu * car.mass * car.g)
    f_net = f_propulsion - f_drag
    a_accel = f_net / car.mass
    
    # Potential velocity from acceleration
    v_accel_step = np.sqrt(max(0, current_v**2 + 2 * a_accel * ds))
    
    # Part B. The car speed is the Min of Accel, Braking
    # Compare our potential acceleration speed against the braking curve calculated above
    v_next = min(v_accel_step, v_braking[i])
    
    v_final[i] = v_next
    
    # Part C. Back-calculate accelerations for plotting
    # Actual longitudinal acceleration (or deceleration)
    actual_a_long = (v_final[i]**2 - v_final[i-1]**2) / (2*ds)
    accel_long[i] = actual_a_long / car.g
    
    # Lateral Acceleration
    r = track.radii[i]
    accel_lat[i] = (v_final[i]**2 / r) / car.g if r != np.inf else 0
    
    # Part D. Stats
    v_avg = (v_final[i] + v_final[i-1]) / 2
    dt = ds / v_avg if v_avg > 0 else 0
    times[i] = times[i-1] + dt
    
    # Energy calculation (Only count positive longitudinal power usage) 
    if actual_a_long > 0:
        torques[i] = motor_torque
        # Power = Force * Velocity (Mechanical) / Efficiency
        # Approximate using propulsion force required for this acceleration
        f_req = (actual_a_long * car.mass) + f_drag
        mech_power = f_req * v_avg
        elec_power = mech_power / car.efficiency
        powers[i] = elec_power / 1000.0
        energy_total += elec_power * dt
    else:
        # Braking/Coasting
        torques[i] = 0
        powers[i] = 0

# --- Results & Visualization ---

# Time, Speed, Acceleration, Torque, Power Plots
fig = plt.figure(figsize=(12, 10))
# Track Map/lap time/ and energy display
track.plot_track(times[-1], energy_total/1000)

# 2. Speed
plt.subplot(5, 2, 2)
plt.plot(times, v_final * 2.237, color='blue')
plt.ylabel('Speed (mph)')
plt.title('Speed over Time')
plt.grid(True)

# 3. Acceleration
theoretical_motor_accel = ((torques * car.drive_ratio * car.efficiency / car.tire_radius) - 
                          (0.5 * car.rho * car.frontal_area * car.cd * (v_final**2))) / car.mass / car.g

plt.subplot(5, 2, 4)
# Plot the theoretical UNLIMITED acceleration in dashed line
# plt.plot(times, theoretical_motor_accel, 'g--', label='Motor Potential', alpha=0.5)
# Plot the actual LIMITED acceleration in solid line
plt.plot(times, accel_long, 'g-', label='Actual Long. Accel')
plt.plot(times, accel_lat, 'r-', label='Lat. Accel')

plt.ylabel('Acceleration (G)')
plt.legend()
plt.grid(True)
plt.title('Acceleration (G-G Diagram)')
plt.legend()
plt.grid(True)

# 4. Torque
plt.subplot(5, 2, 6)
plt.plot(times, torques, color='orange')
plt.ylabel('Motor Torque (Nm)')
plt.title('Motor Torque vs Time')
plt.grid(True)

# 5. Power
plt.subplot(5, 2, 8)
plt.plot(times, powers, color='purple')
plt.ylabel('Power Draw (kW)')
plt.title('Power Draw vs Time')
plt.xlabel('Time (s)')
plt.grid(True)


# --- GGV Visualization ---
fig2 = plt.figure(figsize=(14, 6))

# Plot 1: 2D Iso-Velocity Contours
ax1 = fig2.add_subplot(121)
circle = plt.Circle((0, 0), car.mu, color='k', fill=False, linestyle='--', label='Tire Friction Limit (1G)')
ax1.add_patch(circle)

colors = plt.cm.viridis(np.linspace(0, 1, len(contour_speeds)))
for i, speed in enumerate(contour_speeds):
    lat_data, long_data = contours[speed]
    ax1.plot(lat_data, long_data, color=colors[i], linewidth=2, label=f'{speed} m/s')

ax1.set_aspect('equal')
ax1.set_xlabel('Lateral Acceleration (G)')
ax1.set_ylabel('Longitudinal Acceleration (G)')
ax1.set_title('2D G-G Diagram (Speed Contours)')
ax1.grid(True)
ax1.legend(loc='lower right')
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)

# Plot 2: 3D Surface
ax2 = fig2.add_subplot(122, projection='3d')
sc = ax2.scatter(Lat_3d, Long_3d, V_3d, c=V_3d, cmap='viridis', s=2, alpha=0.5)

ax2.set_xlabel('Lat Accel (G)')
ax2.set_ylabel('Long Accel (G)')
ax2.set_zlabel('Velocity (m/s)')
ax2.set_title('3D GGV Surface')
ax2.set_xlim(-1.2, 1.2)
ax2.set_ylim(-1.2, 1.2)


## show all final plots from matplotlib
plt.tight_layout()
plt.show()