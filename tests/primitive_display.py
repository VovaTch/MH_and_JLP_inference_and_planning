import numpy as np
import math
import gtsam
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

np.set_printoptions(precision=3)
plt.rcParams.update({'font.size': 16})

stride_length = 1
motion_primitives = list()
motion_primitives.append(gtsam.Pose3(gtsam.Rot3.Ypr(-0.00, 0.0, 0.0), np.array([stride_length, 0.0, 0.0])))
motion_primitives.append(gtsam.Pose3(gtsam.Rot3.Ypr(+math.pi / 4, 0.0, 0.0), np.array([stride_length, stride_length, 0.0])))
motion_primitives.append(gtsam.Pose3(gtsam.Rot3.Ypr(-math.pi / 4, 0.0, 0.0), np.array([stride_length, -stride_length, 0.0])))
motion_primitives.append(gtsam.Pose3(gtsam.Rot3.Ypr(+math.pi / 2, 0.0, 0.0), np.array([0.0, stride_length, 0.0])))
motion_primitives.append(gtsam.Pose3(gtsam.Rot3.Ypr(-math.pi / 2, 0.0, 0.0), np.array([0.0, -stride_length, 0.0])))

fig = plt.figure(0)
ax = fig.gca()
wedges = []
obj_idx_wedge = 0
plt.plot(0, 0, 'ro')
for motion_primitive in motion_primitives:
    plt.arrow(0.0, 0.0, -motion_primitive.y(), motion_primitive.x())
    ax.annotate("", xy=(-motion_primitive.y(), motion_primitive.x()), xytext=(0, 0),arrowprops = dict(arrowstyle="->"))
    wedge = mpatches.Wedge((-motion_primitive.y(), motion_primitive.x()), 10,
                           motion_primitive.rotation().yaw() * 180 / 3.141 - 60 + 90,
                           motion_primitive.rotation().yaw() * 180 / 3.141 + 60 + 90, ec="none")
    wedges.append(wedge)

collection = PatchCollection(wedges, facecolors='red', alpha=0.2)
ax.add_collection(collection)

ax.set_xlabel('X axis [m]')
ax.set_ylabel('Y axis [m]')
plt.xlim(-stride_length * 2, stride_length * 2)
plt.ylim(-stride_length, stride_length * 2)
plt.tight_layout()
plt.show()