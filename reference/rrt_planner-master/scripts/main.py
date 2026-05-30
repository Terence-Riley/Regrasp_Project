#!/usr/bin/env python3
import rospy
import numpy as np
from datetime import datetime
import time
import threading

from perception import init_realsense, detect, process_frame
from moveit import init, add_obstacle, set_goal, go_home
from kinematics import NLinkArm
from optimize import BSplineOptimizer
from visualize import plot_cartesian_trajectory

import pid_angle_control
import utilities
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient

obstacle = None
lock = threading.Lock()
obstacle_updated_event = threading.Event()
stop_flag = threading.Event()
current_execution_event = threading.Event()
execution_state = "IDLE"  # ["IDLE", "TO_OBJECT", "TO_GOAL", "TO_HOME"]
current_target = None

def renew_listener(pipeline, align, prev, threshold=0.05, stable_duration=1.0):
    global obstacle
    prev_obstacle = prev
    candidate_obstacle = None
    candidate_start_time = None

    while not stop_flag.is_set():
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        new_obstacle = process_frame(color_image, depth_frame, mode='replan')

        if new_obstacle is None:
            time.sleep(0.05)
            continue

        diff = np.linalg.norm(np.array(new_obstacle) - np.array(prev_obstacle))

        if diff > threshold:
            # New potential value, check if it's the same as last candidate
            if candidate_obstacle is None or np.linalg.norm(np.array(new_obstacle) - np.array(candidate_obstacle)) > threshold:
                candidate_obstacle = new_obstacle
                candidate_start_time = time.time()
            else:
                # Candidate is stable, check if it's stable long enough
                if time.time() - candidate_start_time >= stable_duration:
                    with lock:
                        obstacle = candidate_obstacle
                        obstacle_updated_event.set()
                    prev_obstacle = candidate_obstacle
                    on_obstacle_changed(obstacle)
                    candidate_obstacle = None
                    candidate_start_time = None
        else:
            candidate_obstacle = None
            candidate_start_time = None

        time.sleep(0.05)

def on_obstacle_changed(obstacle):
    global scene, current_execution_event

    # Stop current execution
    current_execution_event.set()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [Trigger] Responding immediately to obstacle change!")
    
    
    # Update obstacle
    scene.remove_world_object()
    rospy.sleep(0.05)
    add_obstacle(scene, obstacle[0], obstacle[1]+0.02, 0.01, 0.12, 0.12, 0.2)

def wait_for_obstacle_stability(duration=3.0):
    global obstacle_updated_event
    start_time = time.time()
    obstacle_updated_event.clear()
    
    while time.time() - start_time < duration:
        if obstacle_updated_event.is_set():
            return False
        time.sleep(0.1)
    return True

def execute_task_segment(group, base, target_pos, optimizer):
    global current_execution_event, execution_state, all_path
    was_interrupted = False  # Track if execution was interrupted by obstacle change
    
    while True:
        if was_interrupted:
            # Only wait for stability after an interruption
            print("Waiting for obstacle position to stabilize...")
            if not wait_for_obstacle_stability(1.0):
                print("Obstacle position changed during stability wait")
                continue
        
        # Plan new path
        print(f"Planning path to {target_pos}")
        path = set_goal(group, target_pos[0], target_pos[1], target_pos[2])
        if path is None:
            print("Failed to plan path")
            return False
            
        # Optimize path
        smooth_path = optimizer.optimize(path)
        all_path = np.vstack((all_path, smooth_path)) if len(all_path)!=0 else smooth_path
        
        # Execute path with interruption checking
        current_execution_event.clear()
        success = pid_angle_control.execute_path(base, smooth_path, current_execution_event)
        
        if success and not current_execution_event.is_set():
            print("Successfully reached target")
            return True
            
        if current_execution_event.is_set():
            print("Execution interrupted, replanning...")
            was_interrupted = True  # Set interrupt flag
        else:
            print("Execution failed")
            return False

def main():
    global obstacle, scene, execution_state, robot, all_path
    pipeline, align = init_realsense()
    
    joint_limits = np.array([(205, 150), (205, 150), (210, 150), (210, 145), (215, 140), (210, 150)]) / 180 * np.pi

    dh_params = [
        [0.0,        0.0 / 1000,   243.3 / 1000,  0.0],
        [np.pi/2,    0.0 / 1000,    30.0 / 1000,  np.pi/2],
        [np.pi,    280.0 / 1000,    20.0 / 1000,  np.pi/2],
        [np.pi/2,    0.0 / 1000,   245.0 / 1000,  np.pi/2],
        [np.pi/2,    0.0 / 1000,    57.0 / 1000,  0.0],
        [-np.pi/2,   0.0 / 1000,   235.0 / 1000, -np.pi/2]
    ]

    robot = NLinkArm(dh_params, joint_limits)

    obj, goal, obstacle = detect(pipeline, align)

    print(f"Detected object: {obj}")
    print(f"Goal position: {goal}")
    print(f"Obstacle position: {obstacle}")

    group, scene, _ = init()

    scene.remove_world_object()
    rospy.sleep(1.0)
    add_obstacle(scene, obstacle[0], obstacle[1]+0.02, 0.01, 0.12, 0.12, 0.2)

    optimizer = BSplineOptimizer(degree=3, num_points=10)

    t = threading.Thread(target=renew_listener, args=(pipeline, align, obstacle))
    t.start()

    all_path = []

    args = utilities.parseConnectionArguments()

    try:
        with utilities.DeviceConnection.createTcpConnection(args) as router:
            base = BaseClient(router)
            pid_angle_control.send_gripper_command(base, 0.2)

            # Move to object
            execution_state = "TO_OBJECT"
            success = execute_task_segment(group, base, [obj[0], obj[1]+0.04, 0.03], optimizer)
            if success:
                pid_angle_control.send_gripper_command(base, 0.8)
                print("Successfully picked up object")

                # Move to goal
                execution_state = "TO_GOAL"
                success = execute_task_segment(group, base, [goal[0]+0.01, goal[1]+0.03, 0.02], optimizer)
                if success:
                    pid_angle_control.send_gripper_command(base, 0)
                    print("Successfully placed object at goal")

                    # Return home
                    execution_state = "TO_HOME"
                    path = go_home(group)
                    if path is not None:
                        smooth_path = optimizer.optimize(path)
                        all_path = np.vstack((all_path, smooth_path)) if len(all_path)!=0 else smooth_path
                        current_execution_event.clear()
                        success = pid_angle_control.execute_path(base, smooth_path, current_execution_event)
                        if success:
                            print("Successfully returned home")
                        else:
                            print("Failed to return home")
                    else:
                        print("Failed to plan path home")
                else:
                    print("Failed to reach goal position")
            else:
                print("Failed to reach object")
    
        plot_cartesian_trajectory(all_path, robot)
    
    except KeyboardInterrupt:
        print("Script interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    stop_flag.set()
    t.join()
    pipeline.stop()
    

if __name__ == "__main__":
    main()
