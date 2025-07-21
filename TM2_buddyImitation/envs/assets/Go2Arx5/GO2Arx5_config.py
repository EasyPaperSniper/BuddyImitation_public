import torch
import numpy as np



GO2AR_LEG_MOTOR_NAMES = [ 
    'FL_hip_joint',    'FR_hip_joint',  'RL_hip_joint',   'RR_hip_joint', 
    'FL_thigh_joint',  'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 
    'FL_calf_joint',   'FR_calf_joint', 'RL_calf_joint',    'RR_calf_joint', 
    ]

GO2AR_HIP_MOTOR_NAMES = [
    'FL_hip_joint',    'FR_hip_joint',  'RL_hip_joint',   'RR_hip_joint', 
]

GO2AR_ARM_MOTOR_NAMES = [ 
    'joint1','joint2', 'joint3', 'joint4', 'joint5', 'joint6',
    ]

GO2AR_MOTOR_NAMES = GO2AR_LEG_MOTOR_NAMES + GO2AR_ARM_MOTOR_NAMES

GO2AR_FOOT_NAMES  = ['FL_foot','FR_foot','RL_foot','RR_foot']

GO2AR_INIT_POS = [0.0, 0.0, 0.35] # x,y,z [m]
GO2AR_INIT_ROT = [1.0, 0.0, 0.0, 0.0] # w,x,y,z [quat]
GO2AR_INIT_MOTOR_ANGLES = [ 
    0.1, -0.1, 0.1, -0.1,
    0.8, 0.8, 1.0, 1.0, 
    -1.5, -1.5, -1.5, -1.5,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    ]



GO2AR_KP = [60]*18
GO2AR_KD = [5]*18