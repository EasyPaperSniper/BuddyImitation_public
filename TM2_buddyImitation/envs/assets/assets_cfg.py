import os
from TM2_buddyImitation import TM2_ENVS_DIR, TM2_ROOT_DIR
from TM2_buddyImitation.utils.configclass import BaseConfig


from TM2_buddyImitation.envs.assets.Go2Arx5 import *

current_dir = os.path.dirname(os.path.realpath(__file__))




class Go2Ar_cfg(BaseConfig):
        type='URDF'
        file_dir = TM2_ENVS_DIR +  '/assets/Go2Arx5/urdf/go2_arx5_finray_no_damping.urdf'
        robot_motor_name = GO2AR_MOTOR_NAMES
        robot_default_qpos = GO2AR_INIT_POS + GO2AR_INIT_ROT + GO2AR_INIT_MOTOR_ANGLES
        robot_arm_motor_name = GO2AR_ARM_MOTOR_NAMES
        robot_leg_motor_name = GO2AR_LEG_MOTOR_NAMES
        robot_hip_motor_name = GO2AR_HIP_MOTOR_NAMES
        robot_foot_name = GO2AR_FOOT_NAMES
        action_scale = 0.25
        penalize_contacts_on = [ "base", 'Head_lower', 'Head_upper', 
                                "FL_calf", "FR_calf", "RL_calf", "RR_calf","FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
                                 "link1","link2","link3","link4","link5","link6"]
        terminate_after_contacts_on = [ "base", 'Head_lower', 'Head_upper', 
                                       "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
                                       "FL_calf", "FR_calf", "RL_calf", "RR_calf",
                                       "FL_hip", "FR_hip", "RL_hip", "RR_hip",
                                       "link1","link2","link3","link4","link5","link6"]
        motor_kp = GO2AR_KP
        motor_kv = GO2AR_KD






ASSETS_CFG_DICT = {'Go2Ar': Go2Ar_cfg()}