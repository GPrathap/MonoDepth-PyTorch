
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import os


# disp_pp = np.load("/home/geesara/diparity/disparities_pp.npy")
# for i in range(0,len(disp_pp)):
#     disp_to_img = scipy.misc.imresize(disp_pp[i].squeeze(), [256, 512])
#     plt.imsave(os.path.join("/home/geesara/diparity/images", "{}_{}disp.png".format(i, "new_image"))
#                , disp_to_img, cmap='plasma')

disp_pp = np.load("/root/wakemeup/output/disparities.npy")
for i in range(0,len(disp_pp)):
    disp_to_img = scipy.misc.imresize(disp_pp[i].squeeze(), [256, 512])
    plt.imsave(os.path.join("/root/wakemeup/output/", "{}_{}----------------------------.png".format(i, "new_image"))
               , disp_to_img, cmap='plasma')