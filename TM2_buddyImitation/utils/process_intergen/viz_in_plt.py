import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
import torch
from TM2_buddyImitation.utils.ratation_conversion import *
from TM2_buddyImitation.utils.quaterion import *
# import cv2


t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
kinematic_chain = [[]]

def plot_t2m( mp_data, result_path, caption):
    mp_joint = []
    for i, data in enumerate(mp_data):
        joint = data[:,:66].reshape(-1,22,3)


        mp_joint.append(joint)

    plot_3d_motion(result_path, t2m_kinematic_chain, mp_joint, title=caption, fps=30)


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(save_path, kinematic_tree, mp_joints, title, figsize=(10, 10), fps=120, radius=4):
    # matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius , radius*2 ])
        ax.set_ylim3d([-radius, radius ])
        ax.set_zlim3d([0, radius ])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    fig = plt.figure(figsize=figsize)
    ax =  fig.add_subplot(projection='3d')#p3.Axes3D(fig)
    init()

    mp_data = []
    for data in mp_joints:
        print(data.shape[0])
    frame_number = min([data.shape[0] for data in mp_joints])
    # print(frame_number)

    # colors = ['red', 'blue', 'black', 'red', 'blue',
    #           'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
    #           'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    #
    colors = ['red', 'green', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

    mp_offset = list(range(-len(mp_joints)//2, len(mp_joints)//2, 1))
    mp_colors = [[colors[i]] * 15 for i in range(len(mp_offset))]

    for i,joints in enumerate(mp_joints):

        # (seq_len, joints_num, 3)
        data = joints.copy().reshape(len(joints), -1, 3)

        MINS = data.min(axis=0).min(axis=0)
        MAXS = data.max(axis=0).max(axis=0)


        #     print(data.shape)

        height_offset = MINS[1]
        data[:, :, 1] -= height_offset
        trajec = data[:, 0, [0, 2]]

        # data[:, :, 0] -= data[0:1, 0:1, 0]
        # data[:, :, 0] += mp_offset[i]
        #
        # data[:, :, 2] -= data[0:1, 0:1, 2]
        mp_data.append({"joints":data,
                        "MINS":MINS,
                        "MAXS":MAXS,
                        "trajec":trajec, })

    #     print(trajec.shape)

    def update(index):
        ax._children =  []
        # ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5

        # plot_xzPlane(-3, 3, 0, -3, 3)
        for pid,data in enumerate(mp_data):
            for i, (chain, color) in enumerate(zip(kinematic_tree, mp_colors[pid])):
                #             print(color)
                if i < 5:
                    linewidth = 5.0
                else:
                    linewidth = 2.0

                ax.plot(data["joints"][index, chain, 0], data["joints"][index, chain, 1], data["joints"][index, chain, 2], linewidth=linewidth,
                          color=color)

        ax.set_xticklabels(['x'])
        ax.set_yticklabels(['y'])
        ax.set_zticklabels(['z'])
        


    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)
    plt.show()

    plt.close()



if __name__ == '__main__':
    file_path ='./TM2_buddyImitation/motion_data/interGen'

    motion_idx = 2091
    motion_names = ['/ori/c1/'+str(motion_idx), '/ori/c2/'+str(motion_idx)]
    mp_data = []
    for motion_name in motion_names:
        mp_data.append(np.load(file_path+motion_name+'.npy')[500:950])
   
    plot_t2m(mp_data, result_path=file_path+'/motion1.mp4',caption='InterGenMotion-'+str(motion_idx))

