# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
cimport numpy as np
cimport cython
cdef bint boolean_variable = True
np.import_array()


FLOAT = np.float32

@cython.boundscheck(False)
@cython.wraparound(False)
def computeIntersection(cp1, cp2, s, e):
      # Signed-distance parametric form.  When Sutherland-Hodgman calls this,
      # s and e are on opposite sides of the clip edge, so |denom| = |ds|+|de|
      # and can never be smaller than either individual distance — no blow-up.
      cdef float cx = cp2[0] - cp1[0]
      cdef float cy = cp2[1] - cp1[1]
      cdef float ds = -cy * (s[0] - cp1[0]) + cx * (s[1] - cp1[1])
      cdef float de = -cy * (e[0] - cp1[0]) + cx * (e[1] - cp1[1])
      cdef float denom = ds - de
      cdef float t
      if denom < 1e-8 and denom > -1e-8:
          return [s[0], s[1]]
      t = ds / denom
      if t < 0.0: t = 0.0
      if t > 1.0: t = 1.0
      return [s[0] + t * (e[0] - s[0]), s[1] + t * (e[1] - s[1])]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint inside(cp1, cp2, p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])

@cython.boundscheck(False)
def polygon_clip_unnest(float [:, :] subjectPolygon, float [:, :] clipPolygon):
    """ Clip a polygon with another polygon.

   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**

   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
    outputList = [subjectPolygon[x] for x in range(subjectPolygon.shape[0])]
    cp1 = clipPolygon[-1]
    cdef int lenc = len(clipPolygon)
    cdef int iidx = 0

    # for clipVertex in clipPolygon:
    for cidx in range(lenc):
        clipVertex = clipPolygon[cidx]
        cp2 = clipVertex
        inputList = outputList.copy()
        outputList.clear()
        s = inputList[-1]

        inc = len(inputList)
 
        # for subjectVertex in inputList:
        for iidx in range(inc):
            subjectVertex = inputList[iidx]
            e = subjectVertex
            if inside(cp1, cp2, e):
                if not inside(cp1, cp2, s):
                    outputList.append(computeIntersection(cp1, cp2, s, e))
                outputList.append(e)
            elif inside(cp1, cp2, s):
                outputList.append(computeIntersection(cp1, cp2, s, e))
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            break
    return outputList


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void copy_points(float[:, :] src, float[:, :] dst, Py_ssize_t num_points):
    cdef Py_ssize_t i
    for i in range(num_points):
        dst[i][0] = src[i][0]
        dst[i][1] = src[i][1]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Py_ssize_t add_point(float[:, :] arr, float[:] point, Py_ssize_t num_points):
    # assert num_points < arr.shape[0] - 1
    # for j in range(dim):
    arr[num_points][0] = point[0]
    arr[num_points][1] = point[1]
    num_points = num_points + 1
    return num_points

@cython.boundscheck(False)
@cython.wraparound(False)
cdef Py_ssize_t computeIntersection_and_add(float[:] cp1, float[:] cp2, float[:] s, float[:] e, float[:, :] arr, Py_ssize_t num_points):
      cdef float cx = cp2[0] - cp1[0]
      cdef float cy = cp2[1] - cp1[1]
      cdef float ds = -cy * (s[0] - cp1[0]) + cx * (s[1] - cp1[1])
      cdef float de = -cy * (e[0] - cp1[0]) + cx * (e[1] - cp1[1])
      cdef float denom = ds - de
      cdef float t
      if denom < 1e-8 and denom > -1e-8:
          arr[num_points][0] = s[0]
          arr[num_points][1] = s[1]
      else:
          t = ds / denom
          if t < 0.0: t = 0.0
          if t > 1.0: t = 1.0
          arr[num_points][0] = s[0] + t * (e[0] - s[0])
          arr[num_points][1] = s[1] + t * (e[1] - s[1])
      num_points = num_points + 1
      return num_points

@cython.boundscheck(False)
@cython.wraparound(False)
def polygon_clip_float(float [:, :] subjectPolygon, float [:, :] clipPolygon):
    """
    Assumes subjectPolygon and clipPolygon have 4 vertices
    """
    cdef Py_ssize_t num_clip_points = clipPolygon.shape[0]
    cp1 = clipPolygon[num_clip_points - 1]

    MAX_INTERSECT_POINTS = 10
    num_intersect_points = 0
    outputList_np = np.zeros((MAX_INTERSECT_POINTS, 2), dtype=np.float32)
    cdef float[:, :] outputList = outputList_np

    inputList_np = np.zeros((MAX_INTERSECT_POINTS, 2), dtype=np.float32)
    cdef float[:, :] inputList = inputList_np

    copy_points(subjectPolygon, outputList, subjectPolygon.shape[0])
    cdef Py_ssize_t noutput_list = subjectPolygon.shape[0]
    cdef Py_ssize_t ninput_list = 0
    cdef Py_ssize_t iidx = 0
    
    for cidx in range(num_clip_points):
        clipVertex = clipPolygon[cidx]
        cp2 = clipVertex
        
        copy_points(outputList, inputList, noutput_list)
        ninput_list = noutput_list
        noutput_list = 0

        s = inputList[ninput_list - 1]
        
        for iidx in range(ninput_list):
            e = inputList[iidx]
            if inside(cp1, cp2, e):
                if not inside(cp1, cp2, s):
                    noutput_list = computeIntersection_and_add(cp1, cp2, s, e, outputList, noutput_list)
                    
                noutput_list = add_point(outputList, e, noutput_list)
            elif inside(cp1, cp2, s):
                noutput_list = computeIntersection_and_add(cp1, cp2, s, e, outputList, noutput_list)
            s = e
        cp1 = cp2
        if noutput_list == 0:
            break
    return outputList_np, noutput_list



@cython.boundscheck(False)
@cython.wraparound(False)
def box_intersection(float [:, :, :, :] rect1, 
                    float [:, :, :, :] rect2, 
                    float [:, :, :] non_rot_inter_areas, 
                    int[:] nums_k2, 
                    float [:, :, :] inter_areas,
                    bint approximate):
    """
    rect1 - B x K1 x 8 x 3 matrix of box corners
    rect2 - B x K2 x 8 x 3 matrix of box corners
    non_rot_inter_areas - intersection areas of boxes 
    """
    
    cdef Py_ssize_t B = rect1.shape[0]
    cdef Py_ssize_t K1 = rect1.shape[1]
    cdef Py_ssize_t K2 = rect2.shape[2]


    for b in range(B):
      for k1 in range(K1):
          for k2 in range(K2):
              if k2 >= nums_k2[b]:
                  break
              
              if approximate and non_rot_inter_areas[b][k1][k2] == 0:
                  continue
              
              ##### compute volume of intersection
              inter = polygon_clip_unnest(rect1[b, k1], rect2[b, k2])
              ninter = len(inter)
              if ninter > 0: # there is some intersection between the boxes
                  xs = np.array([x[0] for x in inter]).astype(dtype=FLOAT)
                  ys = np.array([x[1] for x in inter]).astype(dtype=FLOAT)
                  inter_areas[b,k1,k2] = 0.5 * np.abs(np.dot(xs,np.roll(ys,1))-np.dot(ys,np.roll(xs,1)))
    
        
