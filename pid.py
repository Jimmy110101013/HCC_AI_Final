from simple_pid import PID
from djitellopy import Tello

class Controller:
    def __init__(self, kp, ki, kd):
        self.pid_x = PID(kp, ki, kd, setpoint=0)
        self.pid_y = PID(kp, ki, kd, setpoint=0)
        self.pid_z = PID(kp, ki, kd, setpoint=0)
        self.pid_yaw = PID(kp, ki, kd, setpoint=0)

        # Set output limits to prevent extreme control values
        self.pid_x.output_limits = (-100, 100)
        self.pid_y.output_limits = (-100, 100)
        self.pid_z.output_limits = (-100, 100)
        self.pid_yaw.output_limits = (-100, 100)

    def update(self, drone: Tello, x_error, y_error, z_error, yaw_error, dt):
        control_x = self.pid_x(x_error, dt)
        control_y = self.pid_y(y_error, dt)
        control_z = self.pid_z(z_error, dt)
        control_yaw = self.pid_yaw(yaw_error, dt)

        # Send control commands to the drone
        print(f"PID try to  send ({int(control_x), int(control_z), int(control_y), int(control_yaw)}) to drone")
        drone.send_rc_control(int(control_x), int(control_z), int(control_y), int(control_yaw))
        print(f"PID already send ({int(control_x), int(control_z), int(control_y), int(control_yaw)}) to drone")
