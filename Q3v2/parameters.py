# Car Physical Parameters
CAR_MASS = 250.0  # in kg
C_RR = 0.006     # Coefficient of rolling resistance
C_D = 0.12       # Drag coefficient
FRONTAL_AREA = 1.2 # in m^2
MOTOR_EFFICIENCY = 0.95 

# Battery Parameters
BATTERY_CAPACITY_KWH = 5.0
BATTERY_CAPACITY_J = BATTERY_CAPACITY_KWH * 3.6e6
INITIAL_SOC = 1.0 # Start with a full battery
MIN_SOC = 0.20    # Minimum allowed State of Charge

# Solar Array Parameters
PANEL_AREA = 5.0 # in m^2
PANEL_EFFICIENCY = 0.24

# Race & Simulation Parameters
RACE_DISTANCE_M = 350 * 1000 # 350 km race
RACE_DURATION_HOURS = 8.0
ROUTE_CSV_PATH = 'chennai_to_bangalore_route.csv'

# Optimization Bounds
MIN_VELOCITY_KPH = 30.0
MAX_VELOCITY_KPH = 100.0

# Physical Constants
AIR_DENSITY = 1.2 # in kg/m^3
GRAVITY = 9.81    # in m/s^2