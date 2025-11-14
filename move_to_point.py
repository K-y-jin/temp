#!/usr/bin/env python3

# Standard Imports
import sys
import threading
from tkinter import Y
import traceback
from typing import Callable, Dict, Generator, List, Optional, Tuple
import cv2
import numpy as np
import numpy.typing as npt
import rclpy

# Third-Party Imports
import stretch_urdf.urdf_utils as uu
import tf2_ros

from cv_bridge import CvBridge
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage

# Local Imports
from nrc_web_teleop.action import MoveToPoint
from nrc_web_teleop_helpers.constants import (
    Joint,
)
from nrc_web_teleop_helpers.conversions import (
    remaining_time,
)
from nrc_web_teleop_helpers.move_to_point_state import MoveToPointState
from nrc_web_teleop_helpers.stretch_ik_control import (
    MotionGeneratorRetval,
    StretchIKControl,
)


class MoveToPointNode(Node):

    def __init__(
        self,
        tf_timeout_secs: float = 0.5,
        action_timeout_secs: float = 60.0,
    ):

        super().__init__("move_to_point")

        # Initialize TF2
        self.tf_timeout = Duration(seconds=tf_timeout_secs)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.static_transform_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.lift_offset: Optional[Tuple[float, float]] = None
        self.wrist_offset: Optional[Tuple[float, float]] = None

        # Create the inverse jacobian controller to execute motions
        urdf_fpaths = uu.generate_ik_urdfs("nrc_web_teleop", rigid_wrist_urdf=False)
        urdf_fpath = urdf_fpaths[0]
        self.controller = StretchIKControl(
            self,
            tf_buffer=self.tf_buffer,
            urdf_path=urdf_fpath,
            static_transform_broadcaster=self.static_transform_broadcaster,
        )

        self.cv_bridge = CvBridge()

        # Subscribe to the Navigation camera's CompressedImage and camera info feed
        self.latest_navigation_camera_image = None
        self.latest_navigation_camera_image_lock = threading.Lock()
        self.navigation_camera_subscriber = self.create_subscription(
            CompressedImage,
            "/navigation_camera/image_raw/rotated/compressed",
            self.navigation_camera_cb,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )

        # Create the action timeout
        self.action_timeout = Duration(seconds=action_timeout_secs)

    def initialize(self) -> bool:
        # Initialize the controller
        ok = self.controller.initialize()
        if not ok:
            self.get_logger().error(
                "Failed to initialize the inverse jacobian controller"
            )
            return False

        # Create the shared resource to ensure that the action server rejects all
        # new goals while a goal is currently active.
        self.active_goal_request_lock = threading.Lock()
        self.active_goal_request = None

        # Create the action server
        self.action_server = ActionServer(
            self,
            MoveToPoint,
            "move_to_point",
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )

        return True

    def navigation_camera_cb(self, ros_image: CompressedImage) -> None:
        with self.latest_navigation_camera_image_lock:
            self.latest_navigation_camera_image = ros_image

    def goal_callback(self, goal_request: MoveToPoint.Goal) -> GoalResponse:
        self.get_logger().info(f"Received request {goal_request}")

        # Reject the goal if no Navigation Camera RGB image has been received yet
        with self.latest_navigation_camera_image_lock:
            if self.latest_navigation_camera_image is None:
                self.get_logger().info(
                    "Rejecting goal request since no Navigation Camera RGB image message has been received yet"
                )
                return GoalResponse.REJECT

        # Reject the goal is there is already an active goal
        # with self.active_goal_request_lock:
        #     if self.active_goal_request is not None:
                
        #         self.get_logger().info(
        #             "Rejecting goal request since there is already an active one"
        #         )
        #         return GoalResponse.REJECT

        # Accept the goal
        self.get_logger().info("Accepting goal request")
        self.active_goal_request = goal_request
        return GoalResponse.ACCEPT

    def cancel_callback(self, _: ServerGoalHandle) -> CancelResponse:
        """
        Always accept client requests to cancel the active goal.

        Parameters
        ----------
        goal_handle: The goal handle.
        """
        self.get_logger().info("Received cancel request, accepting")
        return CancelResponse.ACCEPT

    async def execute_callback(
        self, goal_handle: ServerGoalHandle
    ) -> MoveToPoint.Result:
    
        # Functions to cleanup the action
        terminate_motion_executors = False
        motion_executors: List[Generator[MotionGeneratorRetval, None, None]] = []

        def cleanup() -> None:
            """
            Clean up before returning from the action.
            """
            nonlocal terminate_motion_executors, motion_executors
            self.active_goal_request = None
            self.get_logger().debug("Setting termination flag to True")
            terminate_motion_executors = True
            # Execute the motion executors once more to process cancellation.
            if len(motion_executors) > 0:
                try:
                    for i, motion_executor in enumerate(motion_executors):
                        _ = next(motion_executor)
                except Exception:
                    self.get_logger().debug(traceback.format_exc())

        def action_error_callback(
            error_msg: str = "Goal failed",
            status: int = MoveToPoint.Result.STATUS_FAILURE,
        ) -> MoveToPoint.Result:
            self.get_logger().error(error_msg)
            goal_handle.abort()
            cleanup()
            return MoveToPoint.Result(status=status)

        def action_success_callback(
            success_msg: str = "Goal succeeded",
        ) -> MoveToPoint.Result:
            self.get_logger().info(success_msg)
            goal_handle.succeed()
            cleanup()
            return MoveToPoint.Result(status=MoveToPoint.Result.STATUS_SUCCESS)

        def action_cancel_callback(
            cancel_msg: str = "Goal canceled",
        ) -> MoveToPoint.Result:
            self.get_logger().info(cancel_msg)
            goal_handle.canceled()
            cleanup()
            return MoveToPoint.Result(status=MoveToPoint.Result.STATUS_CANCELLED)

        # Start the timer
        start_time = self.get_clock().now()

        # Initialize the feedback
        feedback = MoveToPoint.Feedback()

        goal_point = None
        # Get the initial goal point
        raw_scaled_x, raw_scaled_y = (
            goal_handle.request.scaled_x,
            goal_handle.request.scaled_y,
        )
        goal_point = np.array([raw_scaled_x, raw_scaled_y])
        self.get_logger().debug(f"##### Initial Goal Point: {goal_point}")

        # Publich_feedback message
        def publish_update_goal_point_feedback():
            self.get_logger().info(f"##### Updated Goal Point: [{feedback.new_scaled_x}, {feedback.new_scaled_y}]")
            feedback.elapsed_time = (self.get_clock().now() - start_time).to_msg()
            goal_handle.publish_feedback(feedback)    

        # Execute the states
        motion_executors: List[Generator[MotionGeneratorRetval, None, None]] = []
        states = MoveToPointState.get_state_machine(setup_mode=True)
        self.get_logger().info(f"All States: {states}")

        state_i = 0
        rate = self.create_rate(5.0)  # 5 Hz

        # Calculate the goal positions
        initial_head_joint_states = self.controller.get_head_joint_states()
        initial_head_pan = initial_head_joint_states[Joint.HEAD_PAN]
        initial_head_tilt = initial_head_joint_states[Joint.HEAD_TILT]

        pan_theta = -1.0 * np.arctan2(raw_scaled_x - 0.5, 0.5) # HFOV 90deg
        beta = np.pi * (127.0/180.0) # VFOV 127deg
        focal_length = 0.5 / np.tan(beta/2.0)
        alpha = -1.0 * np.arctan2(raw_scaled_y-0.5, focal_length)  # tan(alpha) = (y-0.5) / focal_length
        tilt_theta = -np.pi * (33.0/180.0) # -33deg down # unseen x is zero when 33deg down
        feedback.new_scaled_x = 0.5
        feedback.new_scaled_y = focal_length * np.tan(tilt_theta - (initial_head_tilt+alpha)) + 0.5
        
        alpha = np.arctan2(0.5 - feedback.new_scaled_y, focal_length)
        height = 1.24 # about 1 m
        x_dist = height * np.tan(np.pi/2 + tilt_theta + alpha)
        self.get_logger().info(f"##### x_dist: {x_dist}")

        goal_positions = {}
        goal_positions[Joint.BASE_ROTATION] = initial_head_pan + pan_theta
        goal_positions[Joint.HEAD_PAN] = pan_theta
        goal_positions[Joint.HEAD_TILT] = tilt_theta
        goal_positions[Joint.BASE_TRANSLATION] = x_dist
        
        def update_feedback_and_publish_feedback(distance_error: float):
            nonlocal tilt_theta, beta, focal_length, height
            feedback.elapsed_time = (self.get_clock().now() - start_time).to_msg()
            alpha = np.arctan2(distance_error, height) - beta/2
            feedback.new_scaled_y = 0.5 - focal_length * np.tan(alpha)
            feedback.new_scaled_x = 0.5
            # self.get_logger().info(f"##### Feedback: {distance_error}")
            goal_handle.publish_feedback(feedback)

        # Loop
        while rclpy.ok():
            concurrent_states = states[state_i]
            # self.get_logger().info(
            #     f"Executing States: {concurrent_states}", throttle_duration_sec=1.0
            # )
            # Check if a cancel has been requested   
            if goal_handle.is_cancel_requested:
                return action_cancel_callback("Goal canceled")
            # Check if the action has timed out
            if (self.get_clock().now() - start_time) > self.action_timeout:
                return action_error_callback("Goal timed out", MoveToPoint.Result.STATUS_TIMEOUT)

            # Move the robot
            if len(motion_executors) == 0:
                for state in concurrent_states:
                    motion_executor = state.get_motion_executor(
                        controller=self.controller,
                        ik_solution=goal_positions,
                        timeout_secs=remaining_time(
                            self.get_clock().now(),
                            start_time,
                            self.action_timeout,
                            return_secs=True,
                        ),
                        check_cancel=lambda: terminate_motion_executors,
                        err_callback=[update_feedback_and_publish_feedback],
                        success_callback=[publish_update_goal_point_feedback],
                    )
                    if motion_executor is None:
                        return action_success_callback("Goal succeeded")
                    motion_executors.append(motion_executor)
            # Check if the robot is done moving
            else:
                try:
                    for i, motion_executor in enumerate(motion_executors):
                        retval = next(motion_executor)
                        if retval == MotionGeneratorRetval.SUCCESS:
                            motion_executors.pop(i)
                            self.get_logger().info(
                                f"##### Success (State Num {state_i}:{concurrent_states}"
                            )
                            break
                        elif retval == MotionGeneratorRetval.FAILURE:
                            raise Exception("Failed to move to goal pose")
                        else:  # CONTINUE
                            pass
                    if len(motion_executors) == 0:
                        state_i += 1
                except Exception as e:
                    self.get_logger().error(traceback.format_exc())
                    return action_error_callback(
                            f"Error executing the motion generator: {e}",
                        MoveToPoint.Result.STATUS_FAILURE,
                    )

            # Sleep
            rate.sleep()
        
        # Failed to execute MoveToPoint
        return action_error_callback("Failed to execute MoveToPoint")


def main(args: Optional[List[str]] = None):
    rclpy.init(args=args)

    move_to_point = MoveToPointNode()
    move_to_point.get_logger().info("Created!")

    # Use a MultiThreadedExecutor so that subscriptions, actions, etc. can be
    # processed in parallel.
    executor = MultiThreadedExecutor()

    # Spin in the background, as the node initializes
    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(move_to_point,),
        kwargs={"executor": executor},
        daemon=True,
    )
    spin_thread.start()

    # Initialize the node
    move_to_point.initialize()

    # Spin in the foreground
    spin_thread.join()

    move_to_point.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()