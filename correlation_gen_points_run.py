
import pickle

from tqdm.auto import tqdm
import pandas as pd
import numpy as np

coordsss=[]

# 2pt_correlation_points
# filename='data/2pt_correlation_points_10.pkl'
# fix_x0y0=False
# log2Size=10
# data_count=100
# 
# lattice_size=(2**log2Size,2**log2Size)
# for i in range(data_count):
#     th=np.random.uniform(0,np.pi/2)
#     r=np.exp(np.random.uniform(np.log(1),np.log(min(lattice_size))))
#     x,y=int(np.abs(r*np.cos(th))),int(np.abs(r*np.sin(th)))
#     if x==0 and y==0:
#         x,y=(1,0) if np.random.uniform()<0.5 else (0,1)
#     x0,y0=np.random.randint(0,lattice_size[0]-x),np.random.randint(0,lattice_size[1]-y)
#     if fix_x0y0:
#         x0,y0=(0,0)
#     x1,y1=x0+x,y0+y
#     coordsss.append(((x0,y0),(x1,y1)))
# coordsss=list(sorted(set(coordsss)))

# torus_correlation_points_y
# filename='data/torus_correlation_points_y_10.pkl'
# log2Size=10
# data_count_axis=30

# lattice_size=(2**log2Size,2**log2Size)
# xAxis=list(range(1,5))+list(np.geomspace(5,2**log2Size-1,max(2,data_count_axis//2-4)).astype(int))
# xAxis=[int(x) for x in xAxis]
# xAxis=sorted(set(xAxis+[lattice_size[0]-x for x in xAxis]))
# yAxis=[0]+xAxis
# for x in xAxis:
#     for y in yAxis:
#         x0,y0=0,y
#         x1,y1=x,y
#         coordsss.append(((x0,y0),(x1,y1)))

# smearing between edge
# filename='data/smearing_between_edge_10.pkl'
# lattice_size=(1024,1024)
# for dist in [1,2,4,8,16,32,64,128,256,512]:
#     for start in [1,2,4,8,16,32,64,128,256,512]:
#         x0,y0=start-1,start-1
#         x1,y1=start-1+dist,start-1+dist
#         coordsss.append(((x0,y0),(x1,y1)))


# smearing between edge3
# filename='data/smearing_between_edge_10_3.pkl'
# lattice_size=(1024,1024)
# for blockSize in [1,2,4,8,16,32,64,128,256,512]:
#     for i in range(250):
#         bBlockX=np.random.randint(0,lattice_size[0]//blockSize//2)*blockSize*2
#         bBlockY=np.random.randint(0,lattice_size[1]//blockSize//2)*blockSize*2
#         # find the center of the bigger block
#         cx0,cy0=bBlockX+blockSize-1,bBlockY+blockSize-1
#         # choose a point on the horizontal or vertical center line of the bigger block, in geometric scale
#         r=np.exp(np.random.uniform(np.log(1),np.log(blockSize)))
#         # up down left right
#         dir=np.random.randint(0,4)
#         if dir==0:
#             cx0,cy0=cx0,cy0+int(r)
#         elif dir==1:
#             cx0,cy0=cx0,cy0-int(r)
#         elif dir==2:
#             cx0,cy0=cx0+int(r),cy0
#         elif dir==3:
#             cx0,cy0=cx0-int(r),cy0

#         # choose r from [1,blockSize-1] in geometric
#         r1=np.exp(np.random.uniform(np.log(1),np.log(blockSize)))
#         r2=np.exp(np.random.uniform(np.log(1),np.log(blockSize)))
#         theta1=np.random.uniform(0,2*np.pi)
#         theta2=np.random.uniform(0,2*np.pi)
#         x0,y0=cx0+int(r1*np.cos(theta1)),cy0+int(r1*np.sin(theta1))
#         x1,y1=cx0+int(r2*np.cos(theta2)),cy0+int(r2*np.sin(theta2))
#         # confine them into the bigger block
#         x0,y0=max(bBlockX,min(bBlockX+blockSize*2-1,x0)),max(bBlockY,min(bBlockY+blockSize*2-1,y0))
#         x1,y1=max(bBlockX,min(bBlockX+blockSize*2-1,x1)),max(bBlockY,min(bBlockY+blockSize*2-1,y1))
#         # check if they are the same point
#         if x0==x1 and y0==y1:
#             continue
#         coordsss.append(((x0,y0),(x1,y1)))

filename='data/smearing_corner_10.pkl'
lattice_size=(1024,1024)
def random_rel_to_corner(lx,ly):
    diagonal_length=np.sqrt(lx**2+ly**2)
    r=np.exp(np.random.uniform(np.log(1),np.log(diagonal_length)))
    theta=np.random.uniform(0,2*np.pi)
    x,y=int(r*np.cos(theta)),int(r*np.sin(theta))
    if x<0: x+=lx
    if y<0: y+=ly
    # confine them into the bigger block
    x,y=max(0,min(lx-1,x)),max(0,min(ly-1,y))
    assert x>=0 and x<lx
    assert y>=0 and y<ly
    return x,y
for l in range(0,20):
    for i in range(128):
        lx=2**(l//2) if l%2==0 else 2**(l//2+1)
        ly=2**(l//2)
        if l%2==0:
            lx=2**(l//2)
            ly=2**(l//2)
            BX0=np.random.randint(0,(lattice_size[0]//lx)-1)
            BX1=BX0+1
            BY0=np.random.randint(0,lattice_size[1]//ly)
            BY1=BY0
        else:
            lx=2**(l//2+1)
            ly=2**(l//2)
            BX0=np.random.randint(0,lattice_size[0]//lx)
            BX1=BX0
            BY0=np.random.randint(0,(lattice_size[1]//ly)-1)
            BY1=BY0+1

        x0,y0=random_rel_to_corner(lx,ly)
        x0,y0=BX0*lx+x0,BY0*ly+y0
        x1,y1=random_rel_to_corner(lx,ly)
        x1,y1=BX1*lx+x1,BY1*ly+y1
        #print(lx,ly,BX0,BX1,BY0,BY1,x0,y0,x1,y1)
        assert x0!=x1 or y0!=y1
        assert x0>=0 and x0<lattice_size[0]
        assert x1>=0 and x1<lattice_size[0]
        assert y0>=0 and y0<lattice_size[1]
        assert y1>=0 and y1<lattice_size[1]
        coordsss.append(((x0,y0),(x1,y1)))




# remove the duplicated ones
coordsss=list(sorted(set(coordsss)))
for coordss in coordsss:
    # print coords and distance
    print(coordss,((coordss[0][0]-coordss[1][0])**2+(coordss[0][1]-coordss[1][1])**2)**0.5)
pickle.dump(coordsss,open(filename,'wb'))
print('total correlators:',len(coordsss))
print('saved to',filename)