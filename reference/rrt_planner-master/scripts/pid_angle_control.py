#!/usr/bin/env python3
import numpy as np
import time
import sys
import os
import argparse
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2

def warp_to_range(angle, min_value=-180, max_value=180):
    """Normalize angle to specified range"""
    # 首先将角度映射到[0, 360)范围
    angle = angle % 360
    # 然后将角度映射到[-180, 180)范围
    if angle > 180:
        angle -= 360
    return angle

def calculate_angle_error(target, current, joint_index=0):
    """
    Calculate the shortest angular distance between target and current angles,
    considering joint limits and choosing the legal path
    """
    # Joint limits as (max_angle, min_angle) in degrees
    limits = [(205, -150), (205, -150), (210, -150), 
             (210, -145), (215, -140), (210, -150)]

    # Get limits for current joint
    max_angle, min_angle = limits[joint_index]

    # Calculate direct path error
    direct_error = target - current

    # Check if direct path would exceed limits
    next_pos = current + direct_error
    if next_pos > max_angle or next_pos < min_angle:
        # Calculate alternative path through opposite direction
        if direct_error > 0:
            # Try negative direction
            alt_error = (target - 360) - current
            next_pos = current + alt_error
            if min_angle <= next_pos <= max_angle:
                return alt_error
        else:
            # Try positive direction
            alt_error = (target + 360) - current
            next_pos = current + alt_error
            if min_angle <= next_pos <= max_angle:
                return alt_error

    # Return original error if no valid alternative found
    return direct_error

class PIDController:
    def __init__(self, Kp, Ki=0.0, Kd=0.0, max_i=10.0, max_output=40.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.error_sum = 0
        self.error_last = 0
        self.max_i = max_i          # 积分限幅
        self.max_output = max_output # 输出限幅

    def control(self, ref, fdb, joint_index):
        error = calculate_angle_error(ref, fdb, joint_index)
        self.error_sum = np.clip(self.error_sum + error, -self.max_i, self.max_i)
        error_diff = error - self.error_last
        self.error_last = error
        output = self.Kp * error + self.Ki * self.error_sum + self.Kd * error_diff
        return np.clip(output, -self.max_output, self.max_output)

def send_gripper_command(base, value):
    """
    Control the gripper
    base: robot control interface
    value: gripper position (0.0 to 1.0, where 0.0 is closed and 1.0 is open)
    """
    print("Performing gripper test in position...")
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    finger.finger_identifier = 1
    finger.value = max(0.0, min(1.0, value))  # Ensure value is between 0 and 1
    base.SendGripperCommand(gripper_command)
    time.sleep(3)  # Wait for gripper to move

def check_joint_limits(angles):
    """Check if joint angles are within limits"""
    # Joint limits as (max_angle, min_angle) in degrees
    limits = [(205, -150), (205, -150), (210, -150), 
             (210, -145), (215, -140), (210, -150)]
    
    for i, angle in enumerate(angles):
        max_angle, min_angle = limits[i]
        # if angle > max_angle or angle < min_angle:
        #     print(f"Joint {i+1} angle {angle}° exceeds limits [{min_angle}°, {max_angle}°]")
        #     return False
    return True

def move_to_angles(base, target_angles, gripper_value=0.0, is_endpoint=False):
    """
    Control all joints to target angles
    base: robot control interface
    target_angles: list of 6 target angles (degrees)
    gripper_value: gripper position (0.0 to 1.0)
    is_endpoint: whether this is a final position (True) or intermediate waypoint (False)
    """
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    if len(target_angles) != 6:
        print("Error: Must provide exactly 6 joint angles")
        return False

    if not check_joint_limits(target_angles):
        return False

    if is_endpoint:
        pids = [
            PIDController(Kp=1.5, Ki=0.01, Kd=0.2),  # Joint 1
            PIDController(Kp=0.8, Ki=0.01, Kd=0.15), # Joint 2
            PIDController(Kp=0.8, Ki=0.01, Kd=0.15), # Joint 3
            PIDController(Kp=1.5, Ki=0.02, Kd=0.15), # Joint 4
            PIDController(Kp=0.8, Ki=0.01, Kd=0.15), # Joint 5
            PIDController(Kp=3.0, Ki=0.01, Kd=0.2)   # Joint 6
        ]
        error_threshold = 0.2
        min_speed_threshold = 0.5 
    else:
        pids = [
            PIDController(Kp=1.5, Ki=0.0, Kd=0.2),  # Joint 1
            PIDController(Kp=1.5, Ki=0.0, Kd=0.1), # Joint 2
            PIDController(Kp=1.5, Ki=0.0, Kd=0.1), # Joint 3
            PIDController(Kp=1.5, Ki=0.0, Kd=0.2), # Joint 4
            PIDController(Kp=1.5, Ki=0.0, Kd=0.2), # Joint 5
            PIDController(Kp=1.5, Ki=0.0, Kd=0.2)   # Joint 6
        ]
        error_threshold = 2.0 
        min_speed_threshold = 4.0  
    
    
    # Normalize target angles
    target_angles = [warp_to_range(angle) for angle in target_angles]
    start_time = time.time()
    
    try:
        # Control gripper
        # send_gripper_command(base, gripper_value)

        prev_speeds = [0.0] * 6
        alpha = 0.2

        # Control loop
        # Joint control loop
        while True:
            # Get current joint angles
            feedback = base.GetMeasuredJointAngles()
            current_angles = [warp_to_range(joint.value) for joint in feedback.joint_angles]
            
            # Calculate speeds for all joints
            speeds = []
            errors = []
            for i in range(6):
                # Calculate base speed from PID
                base_speed = pids[i].control(target_angles[i], current_angles[i], i) * 2
                # Apply minimum speed threshold while maintaining direction
                if abs(base_speed) > 0:
                    base_speed = np.sign(base_speed) * max(abs(base_speed), min_speed_threshold)
                smoothed_speed = alpha * base_speed + (1 - alpha) * prev_speeds[i]
                prev_speeds[i] = smoothed_speed
                speeds.append(smoothed_speed)
                errors.append(abs(calculate_angle_error(target_angles[i], current_angles[i], i)))
            
            # Send velocity commands
            joint_speeds = Base_pb2.JointSpeeds()
            for i in range(6):
                joint_speed = joint_speeds.joint_speeds.add()
                joint_speed.joint_identifier = i
                joint_speed.value = speeds[i]
                joint_speed.duration = 0
            
            base.SendJointSpeedsCommand(joint_speeds)
            
            # Print status
            status = "Joints: " + ", ".join([f"{i+1}:{angle:.1f}" for i, angle in enumerate(current_angles)])
            # print(f"Errors: {max(errors):.2f}", flush=True)
            # print(f"Target: {[f'{a:.1f}' for a in target_angles]}", flush=True)
            # print(f"Current: {[f'{a:.1f}' for a in current_angles]}", flush=True)
            # Check if all joints reached target using appropriate error threshold
            if all(error < error_threshold for error in errors):
                point_type = "endpoint" if is_endpoint else "waypoint"
                if is_endpoint:
                    print(f"Target {point_type} reached")
                return True
                
            # Timeout protection (10 seconds)
            if time.time() - start_time > 10:
                print("Timeout reached")
                return False
                
    except KeyboardInterrupt:
        print("Control interrupted by user")
        return False
    finally:
        # Stop all joint motion
        joint_speeds = Base_pb2.JointSpeeds()
        for i in range(6):
            joint_speed = joint_speeds.joint_speeds.add()
            joint_speed.joint_identifier = i
            joint_speed.value = 0
            joint_speed.duration = 0
        base.SendJointSpeedsCommand(joint_speeds)

def execute_path(base, path, execution_event=None, gripper_value=0.0, is_first_motion=True):
    """
    Execute a sequence of joint configurations
    base: robot control interface
    path: list of joint angle configurations (list of 6-element arrays)
    gripper_value: gripper position (0.0 to 1.0)
    is_first_motion: True if this is the init->obj path, False if obj->goal path
    """
    success = True
    path_len = len(path)
    
    for i, target_angles in enumerate(path):
        # Convert from radians to degrees
        angles_deg = np.array(target_angles) * 180.0 / np.pi
        # Last point in path is an endpoint
        is_endpoint = (i == path_len - 1)
        
        # Execute motion with appropriate parameters
        if execution_event is not None and execution_event.is_set():
            print("Path execution interrupted")
            return False
            
        success &= move_to_angles(base, angles_deg, gripper_value, is_endpoint)
        if not success:
            motion_type = "initial pickup" if is_first_motion else "final placement"
            point_type = "endpoint" if is_endpoint else "waypoint"
            print(f"Failed to reach {point_type} during {motion_type} motion:", angles_deg)
            return False
            
        # If this is the endpoint of first motion (at object), briefly pause
        if is_first_motion and is_endpoint:
            print("Reached object pickup position, pausing briefly...")
            time.sleep(0.5)
    
    return True

def main():
    """
    Main function: setup connection and execute angle control
    Usage: 
    1. Single position control:
       rosrun final_project pid_angle_control.py --ip IP -u USERNAME -p PASSWORD j1 j2 j3 j4 j5 j6 [gripper]
       where j1-j6 are joint angles in degrees
       gripper is gripper position (0.0-1.0, optional)
    
    2. Path execution (internal use):
       This mode is used when called from other scripts like RRT
    """
    # Check if being called from command line or another script
    if len(sys.argv) > 1:  # Command line mode
        parser = argparse.ArgumentParser(description='Control Kinova arm joint angles with PID')
        # Connection arguments
        parser.add_argument("--ip", type=str, help="IP address of destination", default="192.168.1.10")
        parser.add_argument("-u", "--username", type=str, help="username to login", default="admin")
        parser.add_argument("-p", "--password", type=str, help="password to login", default="admin")
        # Joint angles and gripper arguments
        parser.add_argument("joint_angles", type=float, nargs=6, help="6 joint angles in degrees")
        parser.add_argument("gripper", type=float, nargs='?', default=0.0, 
                           help="gripper position (0.0-1.0)")

    try:
        args = parser.parse_args()
        
        # Set up connection
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import utilities
        
        with utilities.DeviceConnection.createTcpConnection(args) as router:
            base = BaseClient(router)
            success = move_to_angles(base, args.joint_angles, args.gripper)
            return 0 if success else 1

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
