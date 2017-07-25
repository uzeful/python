import lmdb
import caffe
import numpy as np
import cv2
from caffe.proto import caffe_pb2
from lmdb_tools import get_lmdb_value, lmdb_put
#import os

def create_multifocus_lmdb(lmdb_rgbsource_file, lmdb_depsource_file, key_file, lmdb_mf1_file, lmdb_mf2_file, lmdb_label_file, depthreshold=0.7, blurKernelSize=(9,9), blurSigma=3.0):
    # Input: lmdb_rgbsource_file, and lmdb_depsource_file are the source rgb and depth image; key_file stores the keys bounding rgb and depth image, depthreshold denotes bluring parts, blurKernelSize and blurSigma decides the blurring scale
    # Output lmdb_mf2_file and lmdb_mf1_file are the generated multifocus image set database, lmdb_label_file is the ground truth label such as 0, 1

    # read the source images to creating the multi-focus images
    rgbsource_env = lmdb.open(lmdb_rgbsource_file)
    depsource_env = lmdb.open(lmdb_depsource_file)


    # clear the exited lmdb
    #os.system('rm ' + lmdb_mf1_file + ',' + lmdb_mf2_file + ',' + lmdb_label_file)


    # create lmdb file for multifocus image sets
    mf1_env = lmdb.open(lmdb_mf1_file, map_size=int(1e12))
    mf2_env = lmdb.open(lmdb_mf2_file, map_size=int(1e12))
    label_env = lmdb.open(lmdb_label_file, map_size=int(1e12))


    # define datum for storing data
    datum = caffe_pb2.Datum()

    # Read the key_file, which is same of all related lmdbs
    Keys = open(key_file, 'r')
    Lists = Keys.readlines()
    Num = len(Lists)

    # write the multi-focus images one by one
    for ii in range(Num):
        key = Lists[ii].strip('\n')
        rgbValue = get_lmdb_value(rgbsource_env, key)
        depValue = get_lmdb_value(depsource_env, key)

        datum.ParseFromString(rgbValue)
        #rgblabel = datum.label
        rgbData = caffe.io.datum_to_array(datum)

        datum.ParseFromString(depValue)
        deplabel = datum.label
        depData = caffe.io.datum_to_array(datum)

        rgbImage = np.transpose(rgbData, (1,2,0))
        depImage = depData

        normDep = depImage / depImage.max()
        marker = np.array((normDep > 0.5), np.uint8)

        fullBlurImg = cv2.GaussianBlur(rgbImage, blurKernelSize, blurSigma)

        sz = fullBlurImg.shape
        marker3 = np.zeros(sz, np.uint8)
        marker3 = marker
        partBlurImg1 = fullBlurImg * marker3 + rgbImage * (np.ones(sz, np.uint8)-marker3)
        partBlurImg2 = rgbImage * marker3 + fullBlurImg * (np.ones(sz, np.uint8)-marker3)

        cv2.imshow('rgb', rgbImage)
        cv2.imshow('depth', depImage)
        cv2.imshow('blur1', partBlurImg1)
        cv2.imshow('blur2', partBlurImg2)
        cv2.waitKey(1)
        print('{}'.format(key))

        part1 = np.transpose(partBlurImg1, (2, 0, 1))
        part2 = np.transpose(partBlurImg2, (2, 0, 1))
        decmap = marker
        label = deplabel


        datum_write1 = caffe.io.array_to_datum(part1, label)
        lmdb_put(mf1_env, key.encode('ascii'), datum_write1.SerializeToString())
        datum_write2 = caffe.io.array_to_datum(part2, label)
        lmdb_put(mf2_env, key.encode('ascii'), datum_write2.SerializeToString())
        datum_write3 = caffe.io.array_to_datum(decmap, label)
        lmdb_put(label_env, key.encode('ascii'), datum_write3.SerializeToString())

    rgbsource_env.close()
    depsource_env.close()

    mf1_env.close()
    mf2_env.close()
    label_env.close()
