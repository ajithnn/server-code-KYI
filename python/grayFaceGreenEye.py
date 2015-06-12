#getting the projection of the gradient as a function of the angle 
# defining a pixel as row, col

# get gray face with green lens overlayed - finished version 1
import numpy as np
import cv2
import copy as cp
import os
import cv2.cv as cv
import time
import scipy as sp
from scipy import signal

def cutCircle(center, radius, img):
    siz = np.shape(img)
    x = np.array(range(siz[0]))
    y = np.array(range(siz[1]))
    xi, yi = np.meshgrid(x, y)
    xi = np.power(xi - center[1], 2)
    yi = np.power(yi - center[0], 2)
    dist = np.power(xi + yi, 0.5)
    relMask = np.array(dist < radius, dtype = np.int64)
    outImg = np.multiply(relMask, img)
    return outImg

print sys.argv[1],sys.argv[2]
PathForRoot = sys.argv[1] + '/assets'
VendorPath = '/app/vendor'

try:
    eyeI = cv2.imread(PathForRoot + '/green-big.png')
    releyeI = eyeI[9:82, 9:82, :]
    sizeye = np.shape(releyeI)
    eyerad = 36

    face_cascade = cv2.CascadeClassifier(VendorPath + '/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(VendorPath + '/share/OpenCV/haarcascades/haarcascade_eye.xml')

    #imgpath = '/Users/z001c3k/rahul/work/image/lens/testImgsTrimmd/'
    #imgpath = '/Users/z001c3k/rahul/work/image/lens/newTestImages/'
    #imgoutpath = '/Users/z001c3k/rahul/work/image/lens/outCornerPtsN5/'
    #fname = '/Users/z001c3k/rahul/work/image/lens/badImagesForTraining/Disproportinal_eye_1.jpeg'
    #fname = '/Users/z001c3k/rahul/work/image/lens/testImages/IMG_20150118_080056.jpg'
    #flist = os.listdir(imgpath)
    fname = PathForRoot + "/CurrentImg.jpg"
    flist = []
    flist.append(fname)
    #flist.remove(flist[0])
    c = 0
    #flist = ['image.jpeg']

    for fname in flist:
        c = c+1
        print fname
        b_img = cv2.imread(fname)
        siz = np.shape(b_img)
        if np.float32(np.product(siz))/(640*640*3) < 1:
            scaleDownFactor = 1
            print "image resolution is lesser than desired value"
        else:
            scaleDownFactor = int(np.sqrt(np.float32(np.product(siz))/(640*640*3)))
        nr =int(siz[0]/scaleDownFactor)
        nc = int(siz[1]/scaleDownFactor)

        img = cv2.resize(b_img, (nc, nr))
        #gray_hres = cv2.cvtColor(b_img, cv2.COLOR_BGR2GRAY)
        gray_hres = b_img[:, :, 2]
        #orig_bw = (np.concatenate((gray_hres[:, :, np.newaxis], gray_hres[:, :, np.newaxis], gray_hres[:, :, np.newaxis]), axis = 2))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        allfaces = face_cascade.detectMultiScale(gray, 1.1, 5)
        s = np.shape(allfaces)
        if s[0]<1:
            print 'no faces detected'
        else:
            if s[0]==1:
                face = allfaces
            else:
                for i in np.arange(s[0]):
                    faceFraction = float(allfaces[i,2]*allfaces[i,3])/(nr*nc)
                    if (faceFraction>0.1) and (faceFraction<0.8):
                        face = allfaces[i,:] # this ensures there is only one face if at all detected
                        break        
            (x,y,w,h) = np.squeeze(face)
            #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            (xhres,yhres,whres,hhres) = (int(scaleDownFactor*x),int(scaleDownFactor*y),int(scaleDownFactor*w),int(scaleDownFactor*h)) 
            roi_gray = gray[y:y+h, x:x+w]
            #roi_origbw = orig_bw[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5) # try doing this on the bigger image
            if np.shape(eyes)[0] == 0:
                print 'eye not detected'
            else:
                for (ex,ey,ew,eh) in eyes:
                    eyecx = ey+eh/2
                    if np.squeeze([eyecx>0.2*h] and [eyecx<0.5*h]):
                        #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                        eye_det = roi_color[ey:ey+eh, ex:ex+ew]
                        #eye_det_origbw = roi_origbw[ey:ey+eh, ex:ex+ew]
                        (exhres,eyhres,ewhres,ehhres) = (int(scaleDownFactor*ex),int(scaleDownFactor*ey),int(scaleDownFactor*ew),int(scaleDownFactor*eh))
                        eye_det_gray_hres = gray_hres[yhres+eyhres:yhres+eyhres+ehhres, xhres+exhres:xhres+exhres+ewhres]
                        eye_det_gray_hres_heq = cv2.equalizeHist(eye_det_gray_hres)
                        eye_det_b_img = b_img[yhres+eyhres:yhres+eyhres+ehhres, xhres+exhres:xhres+exhres+ewhres]
                        eye_det_gray_hres_heq_med11 = cv2.medianBlur(eye_det_gray_hres_heq,11)
                        circles = cv2.HoughCircles(eye_det_gray_hres_heq_med11,cv.CV_HOUGH_GRADIENT,4,6,param1=70,param2=35,minRadius=int(ewhres/6.6),maxRadius=int(ewhres/5))
                        while 1:
                            if ([circles[0,0,1]>0.4*h] and [circles[0,0,1]<0.6*h] and [circles[0,0,0]>0.3*h]  and [circles[0,0,0]>0.7*h] ):
                                break
                            else:
                                circles = circles[:, 1:, :]
                        a = None
                        if (type(circles) == type(a)):
                            print ('one iris not detected')                            
                        else:
                            cenRow = int(circles[0,0,1])
                            cenCol = int(circles[0,0,0])
                            cenRad = int(circles[0,0,2])
                            cen = np.array([cenRow, cenCol])
                            dscalefac = np.float32(eyerad)/(cenRad)
                            nreye = int(sizeye[0]/dscalefac)
                            nceye = int(sizeye[1]/dscalefac)
                            smaleyeimg = cv2.resize(releyeI, (nceye, nreye))
                            smaleyeimgOut = cp.copy(smaleyeimg)
                            
                            impg = cp.copy(eye_det_gray_hres_heq_med11)
                            sobelxf = cv2.Sobel(impg,cv2.CV_64F,1,0,ksize=11)
                            sobelyf = cv2.Sobel(impg,cv2.CV_64F,0,1,ksize=11)
                            #maxgrad = 40000
                            maxgrad = np.max([sobelxf, sobelyf])
                            mag = cv2.magnitude(sobelxf/maxgrad, sobelyf/maxgrad)
                            tet = cv2.phase(sobelxf/maxgrad, sobelyf/maxgrad)
                            prj = np.zeros((180), dtype = np.float16)
                            for idx, theta in enumerate(np.linspace(0, np.pi, 180)):
                                p = np.int16(np.round(np.array([cenRow-cenRad*np.sin(theta), cenCol+cenRad*np.cos(theta)])))
                                prj[idx] = mag[p[0], p[1]]*np.cos(np.abs((2*np.pi - tet[p[0], p[1]]) - theta))                    
                            #pr = np.nonzero(prj<0.1)
                            #numAngles = np.shape(pr)[1]
                            prjext = np.hstack((np.tile(prj[0], 11), prj, np.tile(prj[179], 11)))
                            y3 = sp.signal.convolve(prjext, (np.float16(1)/9)*np.ones((9,)))
                            y4 = y3[15:210-15]
                            y5 = y4[1:] - y4[0:179]
                            zero_crossings = np.where(np.diff(np.sign(y5)))[0]
                            y6 = y5[1:] - y5[0:178]
                            #y7 = np.hstack((np.tile(y6[0], 5), y6, np.tile(y6[177], 5)))
                            #y8 = sp.signal.convolve(y7, (np.float16(1)/5)*np.ones((5,)))
                            #y9 = y8[7:192-7]
                            y7 = np.hstack((np.tile(y6[0], 3), y6, np.tile(y6[177], 3)))
                            y8 = sp.signal.convolve(y7, (np.float16(1)/3)*np.ones((3,)))
                            y9 = y8[4:186-4]
                            
                            inflectionPts = zero_crossings[y9[zero_crossings]>0.0005]
                            numAngles = np.shape(inflectionPts)[0]
                                                    
                            if numAngles:
                                rightAngle = inflectionPts[0]
                                leftAngle = inflectionPts[numAngles-1]
                                pleft = np.array([cenRow-cenRad*np.sin(np.deg2rad(leftAngle)), cenCol+cenRad*np.cos(np.deg2rad(leftAngle))])
                                pright = np.array([cenRow-cenRad*np.sin(np.deg2rad(rightAngle)), cenCol+cenRad*np.cos(np.deg2rad(rightAngle))])
                                if (leftAngle - rightAngle)>100:
                                    print('change needed')
                                    meanAngle = (leftAngle+rightAngle)/2
                                    p1 = np.array([cenRow-cenRad*np.sin(np.deg2rad(meanAngle)), cenCol+cenRad*np.cos(np.deg2rad(meanAngle))])                            
                                    # (2p1+p2)/3 = center
                                    p2 = 3*cen - 2*p1                                
                                    #print(p1)
                                    #print(cen)
                                    #print(p2)
                                    smaleyeimgOut[:,:,0] = cutCircle(p2 - [cenRow-cenRad, cenCol-cenRad], int(round(np.linalg.norm(p2 - pleft))), smaleyeimg[:,:,0])
                                    smaleyeimgOut[:,:,1] = cutCircle(p2 - [cenRow-cenRad, cenCol-cenRad], int(round(np.linalg.norm(p2 - pleft))), smaleyeimg[:,:,1])
                                    smaleyeimgOut[:,:,2] = cutCircle(p2 - [cenRow-cenRad, cenCol-cenRad], int(round(np.linalg.norm(p2 - pleft))), smaleyeimg[:,:,2])
                                    #debug viz
                                    #plt.figure()
                                    #marks = [leftAngle, rightAngle]
                                    #plt.plot(np.linspace(0, 179, 180), prj, 'b', np.linspace(0, 179, 180), y4, 'g', marks, prj[marks], 'rD', hold=False)
                                    #plt.xlabel(fname)
                                    #plt.show()
                                    #cv2.circle(eye_det_b_img, (int(round(pleft[1])), int(round(pleft[0]))), 2, (0,255,0), 2)
                                    #cv2.circle(eye_det_b_img, (int(round(pright[1])), int(round(pright[0]))), 2, (0,0,255), 2)
                                    #cv2.circle(eye_det_b_img, (int(round(p1[1])), int(round(p1[0]))), 2, (255,0,255), 2)
                                    #cv2.circle(eye_det_b_img, (int(round(p2[1])), int(round(p2[0]))), int(round(np.linalg.norm(p2 - pleft))), (255,0,0), 2)
                        ret1, mask1 = cv2.threshold(smaleyeimgOut[:,:,1], 5, 255, cv2.THRESH_BINARY)
                        nmask = cv2.bitwise_not(mask1)                    

                        iris_roi = eye_det_b_img[cenRow-cenRad:cenRow-cenRad+nreye, cenCol-cenRad:cenCol-cenRad+nceye, :] # removed a plus one                    
                        iris_roi[:,:,0] = cv2.bitwise_and(iris_roi[:,:,0],iris_roi[:,:,0],mask = nmask)
                        iris_roi[:,:,1] = cv2.bitwise_and(iris_roi[:,:,1],iris_roi[:,:,1],mask = nmask)
                        iris_roi[:,:,2] = cv2.bitwise_and(iris_roi[:,:,2],iris_roi[:,:,2],mask = nmask)
                        iris_roi = cv2.add(iris_roi,smaleyeimgOut/2)
                        eye_det_b_img[cenRow-cenRad:cenRow-cenRad+nreye, cenCol-cenRad:cenCol-cenRad+nceye, :] = iris_roi# removed a plus one
        
                        #cv2.circle(eye_det_b_img, (np.float32(circles[0,0,0]/scaleDownFactor), np.float32(circles[0,0,1]/scaleDownFactor)), np.float32(circles[0,0,2]/scaleDownFactor), (255,0,0), 2)                        
                        
                            
                    else:
                        print 'bad eye detected'
       
        #orig_bw[]
except Exception,e:
    fo = open(PathForRoot + '/' + sys.argv[3], "w+")
    fo.write(str(e))
    fo.close()

cv2.imwrite(imgoutpath + fname, b_img)


    
    