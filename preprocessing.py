import numpy as np
import cv2
import random, sys, glob, warnings

n_iter = 100

def random_patch(phi,h=128, w=128):
    x1, y1 = (np.inf,np.inf)
    while (x1+w+phi[1]+10>image.shape[1]) or (y1+h+phi[1]+10>image.shape[0]) or (x1+2*phi[0]<0) or (y1+2*phi[0]<0):
        x1, y1 = random.randint(0,image.shape[1]), random.randint(0,image.shape[0])
        if x1+w+phi[1]+10<image.shape[1] and y1+h+phi[0]+10<image.shape[0] and x1+2*phi[0]>0 and y1+2*phi[0]>0:
            x2, y2 = x1+w, y1
            x3, y3 = x2, y2+h
            x4, y4 = x1 , y3
            return np.array([(x1,y1),(x2,y2),(x3,y3),(x4,y4)])
        else:
            pass

def random_perturb(phi=[-30,30]):
    patch = random_patch(phi)
    factors = np.array([random.sample([i for i in range(phi[0],phi[1]+1)], 8)])
    if patch is None:
        patch_warp = None
    else:
        patch_warp = (patch.reshape(1,8)+factors).reshape(4,2)
    return patch, patch_warp, factors.astype(np.float32)


for file in glob.glob('/workspace/omkar_projects/WPI_CV/AutoPano/phase2/coco128/images/train2017/*.jpg'):
    image = cv2.imread(file)
    for i in range(n_iter):
        patch, patch_warp, factors = random_perturb()
        if patch is None:
            pass
        else:
            matrix = cv2.getPerspectiveTransform(patch.astype(np.float32), patch_warp.astype(np.float32))
            mat_inv = np.linalg.inv(matrix)
            result = cv2.warpPerspective(image, mat_inv, (image.shape[1], image.shape[0]))
            img_crop = image[patch[0][1]:patch[2][1], patch[0][0]:patch[1][0]] / 255.0
            img_crop_warp = result[patch[0][1]:patch[2][1], patch[0][0]:patch[1][0]] / 255.0

            patch_add = np.resize(np.concatenate([img_crop, img_crop_warp], axis=2), (128,128,6))
            if patch_add.shape[0]!=128 and patch_add.shape[1]!=128:
                warnings.warn('Patch size other than 128x128x6')
                pass
            else:
                np.save('/workspace/omkar_projects/WPI_CV/AutoPano/phase2/data/input/{}_{}.npy'.format(file.split('/')[-1].split('.')[0],i), patch_add)
                np.save('/workspace/omkar_projects/WPI_CV/AutoPano/phase2/data/gt/{}_{}.npy'.format(file.split('/')[-1].split('.')[0],i), factors)