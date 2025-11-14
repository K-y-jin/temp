from __future__ import annotations  # Required for type hinting a class within itself

# Standard imports
from enum import Enum
from typing import Callable, Dict, Generator, List, Optional

# Third-party imports
import numpy as np
import numpy.typing as npt
from geometry_msgs.msg import Point, Quaternion, PoseStamped
from std_msgs.msg import Header
# Local imports
from .constants import (
    Joint,
    get_stow_configuration,
)
from .stretch_ik_control import (
    MotionGeneratorRetval,
    StretchIKControl,
    TerminationCriteria,
)
from tf_transformations import quaternion_about_axis

class MoveToPointState(Enum):
    """
    Determine the goal point is reachable.
    First, the robot stow the arm.
    Second, the robot rotate its base heading to the goal point, keeping the camera view fixed.
    Third, the robot move its base to the goal point.

    The below states can be strung together to form a state machine that moves the robot
    to a pregrasp position. The general principle we follow is that the robot should only
    rotate its base and move the lift when its arm is within the base footprint of the robot
    (i.e., the arm length is fully in and the wrist is stowed).
    """

    STOW_ARM = 0
    ROTATE_BASE = 1
    HEAD_PAN = 2
    MOVE_BASE = 3
    HEAD_TILT = 4
    TERMINAL = 5

    @staticmethod
    def get_state_machine(setup_mode: bool = True) -> List[List[MoveToPointState]]:
        states = []
        if setup_mode:
            states.append([MoveToPointState.STOW_ARM])
            states.append([MoveToPointState.ROTATE_BASE, MoveToPointState.HEAD_PAN])
            states.append([MoveToPointState.HEAD_TILT])
            states.append([MoveToPointState.MOVE_BASE])
        states.append([MoveToPointState.TERMINAL])
        return states

    def get_motion_executor(
        self,
        controller: StretchIKControl,
        ik_solution: Dict[Joint, float],
        timeout_secs: float,
        check_cancel: Callable[[], bool] = lambda: False,
        err_callback: Optional[Callable[[npt.NDArray[np.float64]], None]] = None,
        success_callback: Optional[Callable[[npt.NDArray[np.float64]], None]] = None,
    ) -> Optional[Generator[MotionGeneratorRetval, None, None]]:

        # The parameters that are state-dependant
        joints_for_velocity_control = []
        joint_position_overrides = {}
        joints_for_position_control = {}
        velocity_overrides = {}
        error_callback_temp = None
        success_callback_temp = None

        # Configure the parameters depending on the state
        if self == MoveToPointState.TERMINAL:
            return None
        elif self == MoveToPointState.STOW_ARM:
            joints_for_position_control.update(
                get_stow_configuration([Joint.ARM_L0, Joint.ARM_LIFT, Joint.WRIST_PITCH],
                grip_stuff=True)
            )
        elif self == MoveToPointState.HEAD_PAN_TO_GOAL:
            joints_for_position_control[Joint.HEAD_PAN] = ik_solution[Joint.HEAD_PAN]
            velocity_overrides[Joint.HEAD_PAN] = 0.5
            
        elif self == MoveToPointState.ROTATE_BASE:
            success_callback_temp = success_callback[0]
            goal_pose = PoseStamped()
            header = Header()
            header.frame_id = "base_link"
            header.stamp = controller.node.get_clock().now().to_msg()
            goal_pose.header = header

            goal_pose.pose.position = Point(x=0.0, y=0.0, z=0.0)
            base_rotation = ik_solution[Joint.BASE_ROTATION]
            r = quaternion_about_axis(base_rotation, [0, 0, 1])
            goal_pose.pose.orientation = Quaternion(x=r[0], y=r[1], z=r[2], w=r[3])

            joints_for_velocity_control += [Joint.BASE_ROTATION]
            joint_position_overrides.update(
                {
                    joint: position
                    for joint, position in ik_solution.items()
                    if joint != Joint.BASE_ROTATION
                }
            )
        elif self == MoveToPointState.HEAD_PAN:
            joints_for_position_control[Joint.HEAD_PAN] = 0.0
            velocity_overrides[Joint.HEAD_PAN] = controller.joint_vel_abs_lim[
                Joint.BASE_ROTATION
            ][1]
        elif self == MoveToPointState.HEAD_TILT:
            joints_for_position_control[Joint.HEAD_TILT] = ik_solution[Joint.HEAD_TILT] # 33deg down
            velocity_overrides[Joint.HEAD_TILT] = 0.5

        elif self == MoveToPointState.MOVE_BASE:
            error_callback_temp = err_callback[0]
            # move base to the goal point
            goal_pose = PoseStamped()
            header = Header()
            header.frame_id = "base_link"
            header.stamp = controller.node.get_clock().now().to_msg()
            goal_pose.header = header

            goal_pose.pose.position = Point(x=ik_solution[Joint.BASE_TRANSLATION], y=0.0, z=0.0)
            goal_pose.pose.orientation = Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)
            # Joint.BASE_TRANSLATION is not included in the controllable joints
            # So, we cannot use the ZERO_VEL termination criteria
            return controller.translate_base_to_goal_pose(
                goal=goal_pose,
                termination=TerminationCriteria.ZERO_ERR,
                timeout_secs=timeout_secs,
                check_cancel=check_cancel,
                err_callback=error_callback_temp,
                success_callback=success_callback_temp,
            )

        # Create the motion executor
        if len(joints_for_velocity_control) > 0:
            return controller.rotate_base_to_goal_pose(
                goal=goal_pose,
                articulated_joints=joints_for_velocity_control,
                termination=TerminationCriteria.ZERO_VEL,
                joint_position_overrides=joint_position_overrides,
                timeout_secs=timeout_secs,
                check_cancel=check_cancel,
                err_callback=error_callback_temp,
                success_callback=success_callback_temp,
            )
        if len(joints_for_position_control) > 0:
            return controller.move_to_joint_positions(
                joint_positions=joints_for_position_control,
                velocity_overrides=velocity_overrides,
                timeout_secs=timeout_secs,
            )
        return None