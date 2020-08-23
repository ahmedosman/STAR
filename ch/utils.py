#!/usr/bin/env python
# encoding: utf-8
# Copyright (c) 2016 Max Planck Society. All rights reserved.
# Modified by Ahmed A. A. Osman Feb 2020 

import chumpy as ch 
import numpy as np
import cv2

def axis2quat(p):
    angle =  ch.sqrt(ch.clip(ch.sum(ch.square(p),1), 1e-16,1e16))
    norm_p = p / angle[:,np.newaxis]
    cos_angle = ch.cos(angle/2)
    sin_angle = ch.sin(angle/2)
    qx = norm_p[:,0]*sin_angle
    qy = norm_p[:,1]*sin_angle
    qz = norm_p[:,2]*sin_angle
    qw =  cos_angle -1
    return ch.concatenate([qx[:,np.newaxis],qy[:,np.newaxis],qz[:,np.newaxis],qw[:,np.newaxis]],axis=1)


class Rodrigues(ch.Ch):
    dterms = 'rt'
    
    def compute_r(self):
        return cv2.Rodrigues(self.rt.r)[0]
    
    def compute_dr_wrt(self, wrt):
        if wrt is self.rt:
            return cv2.Rodrigues(self.rt.r)[1].T


def global_rigid_transformation(pose, J, kintree_table):
    results = {}
    pose = pose.reshape((-1, 3))
    id_to_col = {kintree_table[1, i]: i for i in range(kintree_table.shape[1])}
    parent = {i: id_to_col[kintree_table[0, i]] for i in range(1, kintree_table.shape[1])}
    def with_zeros(x):
        return ch.vstack((x, ch.array([[0.0, 0.0, 0.0, 1.0]])))

    results[0] = with_zeros(ch.hstack((Rodrigues(pose[0, :]), J[0, :].reshape((3, 1)))))

    for i in range(1, kintree_table.shape[1]):
        results[i] = results[parent[i]].dot(
            with_zeros(ch.hstack(
                (Rodrigues(pose[i, :]), ((J[i, :] - J[parent[i], :]).reshape((3, 1))))
            )))

    def pack(x):
        return ch.hstack([np.zeros((4, 3)), x.reshape((4, 1))])

    results = [results[i] for i in sorted(results.keys())]
    results_global = results

    if True:
        results2 = [
            results[i] - pack(results[i].dot(ch.concatenate(((J[i, :]), 0))))
            for i in range(len(results))]
        results = results2

    result = ch.dstack(results)
    return result, results_global

def verts_core(pose, v, J, weights, kintree_table, want_Jtr=False):
    A, A_global = global_rigid_transformation(pose, J, kintree_table)
    T = A.dot(weights.T)

    rest_shape_h = ch.vstack((v.T, np.ones((1, v.shape[0]))))

    v = (T[:, 0, :] * rest_shape_h[0, :].reshape((1, -1)) +
         T[:, 1, :] * rest_shape_h[1, :].reshape((1, -1)) +
         T[:, 2, :] * rest_shape_h[2, :].reshape((1, -1)) +
         T[:, 3, :] * rest_shape_h[3, :].reshape((1, -1))).T

    v = v[:, :3]

    class result_meta(object):
        pass

    if not want_Jtr:
        Jtr = None
    else:
        Jtr = ch.vstack([g[:3, 3] for g in A_global])

    meta = result_meta()
    meta.Jtr = Jtr
    meta.A = A
    meta.A_global = A_global
    meta.A_weighted = T

    return v, meta