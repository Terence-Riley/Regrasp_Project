#!/usr/bin/env python3
import sys
import rospy
import numpy as np
import moveit_commander
from moveit_commander import MoveGroupCommander, PlanningSceneInterface
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import Constraints, JointConstraint, PositionConstraint
from shape_msgs.msg import SolidPrimitive
from tf.transformations import quaternion_from_euler

def init():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("gen3_lite_moveit_script", anonymous=True)
    rospy.sleep(1.0)

    group = MoveGroupCommander("arm", robot_description="my_gen3_lite/robot_description", ns="/my_gen3_lite")
    scene = PlanningSceneInterface(ns="/my_gen3_lite")
    robot = moveit_commander.RobotCommander(robot_description="my_gen3_lite/robot_description")

    group.set_planning_time(10)
    group.allow_replanning(True)
    group.set_num_planning_attempts(500)
    group.set_planner_id("RRTstarkConfigDefault")
    group.set_max_velocity_scaling_factor(0.7)
    group.set_max_acceleration_scaling_factor(0.7)

    group.clear_path_constraints()
    group.set_path_constraints(build_constraints())

    return group, scene, robot

def add_obstacle(scene, x, y, z, size_x, size_y, size_z):
    box_pose = PoseStamped()
    box_pose.header.frame_id = "base_link"
    box_pose.pose.orientation.w = 1.0
    box_pose.pose.position.x = x
    box_pose.pose.position.y = y
    box_pose.pose.position.z = z
    box_name = "box"
    scene.add_box(box_name, box_pose, size=(size_x, size_y, size_z))

    rospy.sleep(1.0)

def set_goal(group, x, y, z):
    pose_target = group.get_current_pose().pose
    pose_target.position.x = x
    pose_target.position.y = y
    pose_target.position.z = z

    roll = np.deg2rad(8.3)
    pitch = np.deg2rad(-172.7)
    yaw = np.deg2rad(134.2)
    q = quaternion_from_euler(roll, pitch, yaw)
    pose_target.orientation.x = q[0]
    pose_target.orientation.y = q[1]
    pose_target.orientation.z = q[2]
    pose_target.orientation.w = q[3]

    group.set_pose_target(pose_target)
    while not rospy.is_shutdown():
        success, trajectory, _, _ = group.plan()
        if success:
            print("Trajectory planned successfully")
            joint_trajectory = []
            for point in trajectory.joint_trajectory.points:
                joint_trajectory.append(list(point.positions))
            return np.array(joint_trajectory)
        else:
            print("Failed to plan")
            return None

def go_home(group):
    while not rospy.is_shutdown():
        group.set_named_target("home")
        success, trajectory, _, _ = group.plan()
        if success:
            print("Trajectory planned successfully")
            joint_trajectory = []
            for point in trajectory.joint_trajectory.points:
                joint_trajectory.append(list(point.positions))
            return np.array(joint_trajectory)
        else:
            print("Failed to plan")
            return None

def limit_joint(group, joint_name, target_position, tolerance_above=0.1, tolerance_below=0.1, weight=1.0):
    constraint = JointConstraint()
    constraint.joint_name = joint_name
    constraint.position = target_position
    constraint.tolerance_above = tolerance_above
    constraint.tolerance_below = tolerance_below
    constraint.weight = weight

    constraints = Constraints()
    constraints.joint_constraints.append(constraint)

    group.clear_path_constraints()
    group.set_path_constraints(constraints)

def build_constraints():
    constraints = Constraints()

    joint_constraints = [
        ("joint_2", 0, 0.5, 1.8),
        ("joint_3", 2.6, 0.2, 2.8),
    ]

    for joint_name, target_position, tol_above, tol_below in joint_constraints:
        jc = JointConstraint()
        jc.joint_name = joint_name
        jc.position = target_position
        jc.tolerance_above = tol_above
        jc.tolerance_below = tol_below
        jc.weight = 1.0
        constraints.joint_constraints.append(jc)

    pc = PositionConstraint()
    pc.header.frame_id = "base_link"
    pc.link_name = "end_effector_link"
    pc.weight = 1.0

    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [1.0, 0.9, 0.9]

    box_pose = PoseStamped()
    box_pose.header.frame_id = "base_link"
    box_pose.pose.orientation.w = 1.0
    box_pose.pose.position.x = 0.3
    box_pose.pose.position.y = 0.1
    box_pose.pose.position.z = 0.45

    pc.constraint_region.primitives.append(box)
    pc.constraint_region.primitive_poses.append(box_pose.pose)

    constraints.position_constraints.append(pc)

    return constraints

def main():
    group, scene, robot = init()

    scene.remove_world_object()
    rospy.sleep(1.0)
    add_obstacle(scene, 0.35, -0.05, 0.07, 0.12, 0.12, 0.2)

    pose_target = group.get_current_pose().pose
    print(f"Current pose: x={pose_target.position.x}, y={pose_target.position.y}, z={pose_target.position.z}")
    
    set_goal(group, 0.2, 0.25, 0.01)
    set_goal(group, 0.5, -0.03, 0.01)
    
    group.set_named_target("home")
    group.go(wait=True)
    group.stop()
    group.clear_pose_targets()

if __name__ == "__main__":
    main()
