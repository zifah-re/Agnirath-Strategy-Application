import math
# Original Failing Function
def reached_checkpoint_original(speed, time, checkpoint_distance):
  """Returns True if the car's traveled distance exactly matches the checkpoint"""
  return speed * time == checkpoint_distance
# Fixed and Robust Function
def reached_checkpoint_fixed(speed, time, checkpoint_distance):
  """
  Returns True if the car's traveled distance is close enough,
  accounting for floating-point imprecision
  """
  traveled_distance = speed * time
  return math.isclose(traveled_distance, checkpoint_distance)
# Simple Test Case
s, t, d = 0.1, 0.2, 0.02
print(f"Inputs: speed={s}, time={t}, distance={d}")
print(f"Actual product in Python: {s * t}")
print("-" * 20)
print(f"Original function: {reached_checkpoint_original(s, t, d)}") # False
print(f"Fixed function: {reached_checkpoint_fixed(s, t, d)}")    # True