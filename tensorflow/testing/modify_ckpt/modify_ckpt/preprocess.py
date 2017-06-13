import numpy as np
import h5py
import os
import scipy.io
import tensorflow as tf
import math
from scipy import optimize

def readdata():
    normalpath = './chessData_2017410/chessman_white_queen/normal_white_queen_1.txt'
    chesspath = './chessData_2017410/chessman_white_queen/pointCloud_white_queen_1.txt'
    normal = []
    pointCloud = np.zeros((1,3))
    file =  open(normalpath,'r')
    for line in file:  
        data = np.array(line)
        normal = np.fromstring(data, dtype=float, sep=" ")


    file =  open(chesspath,'r')
    for line in file:    
        data = np.array(line)
        data = np.fromstring(data, dtype=float, sep=" ")    
        pointCloud = np.append(pointCloud,data)


    pointCloud = pointCloud[3:pointCloud.shape[0]]
    N = pointCloud.shape[0] / 3
    pointCloud = np.transpose(pointCloud.reshape(N,3)) 

    return pointCloud, normal

def normalize(vec, maxzero = 1e-12):
    norm_vec = np.linalg.norm(vec)
    if (norm_vec <= maxzero):
      vec_n = zeros(size(vec))
    else:
      vec_n = vec / norm_vec
    
    return vec_n
    

#reference : https://github.com/3dptv/real_time_sobel_fpga_3dptv/blob/master/Matlab/beamsplitter/vrrotvec.m
def vrrotvec(chessnorm,target = [-1,0,0]):
    an = normalize(chessnorm)
    bn = normalize(target)
    
    ax = normalize(np.cross(chessnorm, target))    
    angle = math.acos(np.dot(an, bn)) 
    return ax,angle
    

#reference: https://github.com/3dptv/real_time_sobel_fpga_3dptv/blob/master/Matlab/beamsplitter/vrrotvec2mat.m
def vrrotvec2mat(r):
    s = math.sin(r[3])
    c = math.cos(r[3])
    t = 1 - c
    
    n = normalize(r[0:3])
    x = n[0]
    y = n[1]
    z = n[2]
    m = [ 
     [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y], 
     [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x], 
     [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c  ]
    ]
    return m

def modelnormalize(vertex):
    x = vertex[0,:]
    y = vertex[1,:]
    z = vertex[2,:]
    
    
    scale = 50/ (max(x) - min(x))
    xt = (min(x) + max(x))/2
    yt = (min(y) + max(y))/2
    zt = (min(z) + max(z))/2
    
    x = (x - xt)*scale + xt
    y = (y - yt)*scale + yt
    z = (z - zt)*scale + zt
    
    vertex_new = np.zeros((x.shape[0],3))
    vertex_new[:,0] = x
    vertex_new[:,1] = y
    vertex_new[:,2] = z
   
    return vertex_new

def plot3(a,b,c,mark="o",col="r"):
  from matplotlib import pyplot
  import pylab
  from mpl_toolkits.mplot3d import Axes3D
  pylab.ion()
  fig = pylab.figure()
  ax = Axes3D(fig)
  ax.set_xlabel('x_label')
  ax.set_ylabel('y_label')
  ax.set_zlabel('z_label')
  ax.scatter(a, b, c,marker=mark,color=col)

def circlefit(x,y):
    upper = np.concatenate([x, y, np.ones((1,x.shape[1]))], axis = 0)
    upper = upper.conj().transpose()
    lower = np.array(-(np.power(x,2)+np.power(y,2)))
    lower = lower.conj().transpose()
    a = np.linalg.lstsq(upper,lower)
    xc = -0.5*a[0][0]
    yc = -0.5*a[0][1]
    return xc,yc
    

def rmnoise(vertex):
    xaxis = vertex[:,0]
    xaxis = np.sort(xaxis)
    temp = np.array(np.diff(xaxis[0:199],n=1,axis=0))
    index = np.array(np.where(temp >= 5))

    bottom = xaxis[index[0] +1];
    if not bottom.shape:
        newvertex = vertex[:,np.where(vertex[:,0] > bottom)]
        newvertex[:,0] = newvertex[:,0] - bottom + 1
    else:
        newvertex = vertex
        
    return newvertex

#pointcloud[3][n]
#normal[1][3]
def preprocess(pointCloud = None, desk_normal = None):
    if pointCloud is None and desk_normal is None:
        pointCloud, desk_normal = readdata()
    #find rotation matrix
    cross,ac = vrrotvec(desk_normal)
    r = np.append(cross,ac)
    Rtemp = vrrotvec2mat(r)
    R = np.dot([[1,0,0],[0,0,1],[0,-1,0]], Rtemp)
    
    #rotate point cloud
    pointCloud = np.dot(R, pointCloud)
    pointCloud = pointCloud.conj().transpose()

    #transfer point cloud coordinate and translate to [0,Inf]   
    pointCloud[:,0] = -pointCloud[:,0]
    pointCloud[:,0] = pointCloud[:,0] - min(pointCloud[:,0])+1
    pointCloud[:,1] = pointCloud[:,1] - min(pointCloud[:,1])+1
    pointCloud[:,2] = pointCloud[:,2] - min(pointCloud[:,2])+1
    
    #print pointCloud
    ind = np.where(pointCloud[:,0] < 40)
    xc, yc = circlefit(pointCloud[ind,1],pointCloud[ind,2])

    sy = round(xc[0]) - 25
    sz = round(yc[0]) - 25

    pointCloud = np.round(pointCloud)    
    pointCloud = rmnoise(pointCloud)
        
    x = pointCloud[:,0]
    y = pointCloud[:,1]
    z = pointCloud[:,2]

    y = y - sy
    z = z - sz
    
    answer = np.zeros((50,50,100))
    for pt in range(x.shape[0]):
        if x[pt] > 0 and x[pt] < 100 and y[pt] > 0 and y[pt] < 50 and z[pt] > 0 and z[pt] < 50:
            answer[int(z[pt]),int(y[pt]),int(x[pt])] = 1
        #else:
        #    print(x[pt], y[pt] ,z[pt])
    return answer, pointCloud

