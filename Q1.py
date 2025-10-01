def calculate_steady_state_temp(T_a, tau):
    T_w = 323.0
    tolerance = 1
    max_iterations = 100 
    for _ in range(max_iterations):
        T_w_old = T_w
        T_m = (T_a + T_w) / 2
        B = 1.32 - 0.0012 * (T_m - 293)
        i = 0.561 * B * tau
        R = 0.0575 * (1 + 0.0039 * (T_w - 293))
        P_c = 3 * (i**2) * R
        if R == 0:
            P_e = float('inf')
        else:
            P_e = (9.602e-6 * (B * tau)**2) / R
        T_w = 0.455 * (P_c + P_e) + T_a
        if abs(T_w - T_w_old) < tolerance:
            return round(T_w, 2)
            break

# Example Usage
ambient_temp = 293   # Mentioned in the paper
motor_torque = 16.2   # Mentioned in the paper
steady_temp = calculate_steady_state_temp(ambient_temp, motor_torque)
print(f"For an ambient temp of {ambient_temp} K and a torque of {motor_torque} Nm:")
print(f"The steady-state winding temperature is: {steady_temp} K")