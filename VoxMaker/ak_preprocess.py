import numpy as np

import os

def ak_normal(source):
    min_s = np.nanmin(source)
    output = source - min_s
    output /= (np.nanmax(output) + 0.00000001)
    return output



def voxelize(pts_file_path, voxel_size):
    text_file = open(pts_file_path, "r")
    lines = text_file.read().split("\n")
    arr_x = np.zeros(len(lines))
    arr_y = np.zeros(len(lines))
    arr_z = np.zeros(len(lines))
    i = 0
    for l in lines:
        s = l.replace('\n', '')
        if s == '':
            continue
        s = s.split(' ')
        arr_x[i] = float(s[0])
        arr_y[i] = float(s[1])
        arr_z[i] = float(s[2])
        i = i + 1

    voxel = interpolate_3d(arr_x, arr_y, arr_z, voxel_size=voxel_size)
    return voxel

def interpolate_3d(x, y, z, voxel_size):
    # [norm_x, norm_y, norm_z] = ak_normal([x, y, z])
    norm_x = ak_normal(x)
    norm_y = ak_normal(y)
    norm_z = ak_normal(z)

    norm_x *= voxel_size
    norm_y *= voxel_size
    norm_z *= voxel_size

    norm_ind_x = norm_x.astype(int)
    norm_ind_y = norm_y.astype(int)
    norm_ind_z = norm_z.astype(int)

    voxels = np.zeros((voxel_size, voxel_size, voxel_size))

    voxels[norm_ind_x, norm_ind_y, norm_ind_z] = 1

    return voxels

def dir_to_voxel(pts_dir, vox_dir, voxel_size):
    pts_files = os.listdir(pts_dir)
    for i in range(len(pts_files)):
        print(i)

        pts_path = pts_dir + '/' + pts_files[i]
        print(pts_path)
        v = voxelize(pts_file_path=pts_path, voxel_size=voxel_size)
        np.save(vox_dir + '/' + pts_files[i], v)


# v = voxelize(
#     'E:\Dataset\shapenetcore_partanno_segmentation_benchmark_v0\shapenetcore_partanno_segmentation_benchmark_v0/02691156\points/train/1a04e3eab45ca15dd86060f189eb133.pts'
#     , 10)

file = '04379243'
dir_to_voxel(
    'E:\Dataset\shapenetcore_partanno_segmentation_benchmark_v0\shapenetcore_partanno_segmentation_benchmark_v0/'+file+'\points/train'
    ,
    'E:\Dataset\shapenetcore_partanno_segmentation_benchmark_v0\shapenetcore_partanno_segmentation_benchmark_v0/'+file+'/vox/train'
    , 50)
dir_to_voxel(
    'E:\Dataset\shapenetcore_partanno_segmentation_benchmark_v0\shapenetcore_partanno_segmentation_benchmark_v0/'+file+'\points/test'
    ,
    'E:\Dataset\shapenetcore_partanno_segmentation_benchmark_v0\shapenetcore_partanno_segmentation_benchmark_v0/'+file+'/vox/test'
    , 50)
dir_to_voxel(
    'E:\Dataset\shapenetcore_partanno_segmentation_benchmark_v0\shapenetcore_partanno_segmentation_benchmark_v0/'+file+'\points/val'
    ,
    'E:\Dataset\shapenetcore_partanno_segmentation_benchmark_v0\shapenetcore_partanno_segmentation_benchmark_v0/'+file+'/vox/val'
    , 50)

print('ok')
