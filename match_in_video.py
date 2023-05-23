import numpy as np
from numpy import loadtxt
import os.path
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


Rotation = []
Translat = []
load = os.path.isfile("cameraTraj.csv")
# print(load)
if(load):
    trajP = loadtxt('cameraTraj.csv', delimiter=',')
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(trajP[:,0], trajP[:,2], trajP[:,1])
    plt.title("Trayectoria de la cámara durante el video")
    plt.show()
else:
    print("Calculando trayectoria")

# Read video 
cap = cv.VideoCapture('duckling_video.mp4')

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)

#Captura el primer frame
ret, frame = cap.read()

img1 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


while True:
    # Read a frame from the video
    ret, frame = cap.read()
        
    # Check if end of the video
    if not ret:
        break
        
    #  grayscale
    img2 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # Find fundamental matrix
    F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)

    # # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    # Camera matrix
    K = np.matrix([[656.2149,   0., 433.36419],
                    [0., 634.3659, 549.00237266],
                    [0., 0., 1.]])
    KT = K.getT()

    # Calculate Essential matrix
    E1 = np.matmul(KT, F)
    E = np.matmul(E1,K)

    # Find the rotation and translation from each pair of frames
    retval, R, t, mask = cv.recoverPose(E, pts1, pts2, K)

    Rotation.append(R)
    Translat.append(t)

    img_final = cv.drawKeypoints(frame, kp2, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #Declara el frame actual como el frame previo
    img1 = frame

    # Display 
    cv.imshow('Video original con matching keypoints',img_final)

    # Exit 
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

Rotation = np.array(Rotation)
Translat = np.array(Translat)

# print(Rotation)
# print(Translat)

Fullrotation= np.eye(3)
TransInCrdnt = []

for i in range(len(Rotation)):
  
    TransInCrdnt.append( Fullrotation@Translat[i].copy() )
    Fullrotation = Fullrotation@np.linalg.inv(Rotation[i].copy())
    
TransInCrdnt = np.squeeze( np.array(TransInCrdnt) )
TransInCrdnt.shape

traj = []
summ = np.array([0.,0.,0.])

for i in range(TransInCrdnt.shape[0]):
    traj.append(summ)
    summ = summ + TransInCrdnt[i]
    
traj = np.array(traj)

if(load):
    if(traj.shape[0]>trajP.shape[0]):
        np.savetxt('cameraTraj.csv', traj, delimiter=',')
        print("Nueva trayectoria calculada")
else:
    print("Trayectoria calculada")
    np.savetxt('cameraTraj.csv', traj, delimiter=',')
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(traj[:,0], traj[:,2], traj[:,1])
    plt.title("Trayectoria de la cámara durante el video")
    plt.show()

cv.waitKey(0)

cap.release()
cv.destroyAllWindows()
