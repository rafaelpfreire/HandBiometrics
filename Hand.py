import cv2
import sys
import numpy as np
import pdb as pb

class HandDebug(object):
    def __init__(self):
        pass

    @staticmethod
    def saveImg(img, name):
        filename = "../output/" + name
        cv2.imwrite(filename, img)
        
    @staticmethod
    def resizeImg(img, s):
        # Preservers the aspect ratio
        r = float(s)/float(img.shape[1])
        dim = (int(s), int(img.shape[0]*r))
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return resized

    @staticmethod
    def showImg(img, wname):
        show = HandDebug.resizeImg(img, 1200)
        cv2.imshow(wname, show)
        cv2.waitKey()

    @staticmethod
    def printTestVector(out, binary, candidate, lsize):
        x0 = candidate[1]
        y0 = candidate[0]
        roi = binary[(x0-20):(x0+20),(y0-20):(y0+20)]

        dx = cv2.Sobel(roi, cv2.CV_64F,1,0,ksize=5)
        dy = cv2.Sobel(roi, cv2.CV_64F,0,1,ksize=5)

        cv2.imshow('Sobel',HandDebug.resizeImg(dx,500))
        cv2.imshow('ROI',HandDebug.resizeImg(roi,500))

        print "dx: {} / dy: {}".format(dx[20,20], dy[20,20])

        a = np.arctan(dy[20,20]/(dx[20,20]+np.finfo(float).eps))
        v = np.array((np.cos(a), np.sin(a)))

        v1 = candidate
        v2 = candidate - lsize*v
        p1 = (int(v1[0]), int(v1[1]))
        p2 = (int(v2[0]), int(v2[1]))

        cv2.line(out, p1, p2, (255,255,255), 2)


    @staticmethod
    def printTestCircumference(img, pt):

        for nPoints, radius in zip([4,8,16],[10,20,30]):
            x = pt[0] + np.array(radius * np.cos(np.arange(nPoints) * 2*np.pi/nPoints), np.int)
            y = pt[1] + np.array(radius * np.sin(np.arange(nPoints) * 2*np.pi/nPoints), np.int)

            cv2.circle(img,(pt[0],pt[1]),2,(0,0,0),2)
            for m in range(0, nPoints):
                cv2.circle(img,(x[m],y[m]),2,(0,0,0),2)

class Finger(object):

    tip = None
    base = None
    size = None

    def __init__(self, tip, base):
        self.tip = tip
        self.base = base
        self.size = np.sqrt(np.power(tip,2) + np.power(base,2))

counter = 0

class Hand(object):

    whitch = "RIGHT"
    thumb  = None
    index  = None
    midle  = None
    ring   = None
    little = None

    def __init__(self, imgPath, which):

        if ("RIGHT" == which) or ("LEFT" == which):
            self.witch = which

        image = self._resizeImg(cv2.imread(imgPath), 600.0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        seg = self._skinColorSegmentation(image)
        edge = cv2.Canny(seg, 50, 100)
        valleys = self._valleyDetection(seg, edge)

        for v in valleys:
            cv2.circle(image,v,2,(0,0,255),2)
        #HandDebug.printTestCircumference(gray, valleys[0])
        #HandDebug.printTestCircumference(image, valleys[0])
        #HandDebug.printTestCircumference(image, valleys[0])
        #HandDebug.printTestVector(image, seg, valleys[0], 50)

        out = np.concatenate((seg,edge), axis=1)
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        out = np.concatenate((image,out), axis=1)
        #HandDebug.showImg(out, 'Image')
        global counter
        HandDebug.saveImg(out, 'output-{}.png'.format(counter))
        counter = counter + 1

    def _resizeImg(self, img, s):
        # Preservers the aspect ratio
        r = s/img.shape[1]
        dim = (int(s), int(img.shape[0]*r))
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return resized

    def _skinColorSegmentation(self, image):
        ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        thr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #ycbcr = cv2.GaussianBlur(ycbcr, (9,9), 0)

        for i in range(0, ycbcr.shape[0]):
            for j in range(0, ycbcr.shape[1]):
                cb = ycbcr[i][j][2]
                cr = ycbcr[i][j][1]
                if ((cb >= 77) and (cb <= 127) and
                   (cr >= 133) and (cr <= 174)):
                    thr[i][j] = 255
                else:
                    thr[i][j] = 0

        # TODO: Histogram equalization??

        # Filling holes
        #thr = cv2.medianBlur(thr, 5)
        #fill = thr.copy()
        #h, w = thr.shape[:2]
        #mask = np.zeros((h+2, w+2), np.uint8)
        #cv2.floodFill(fill, mask, (0,0), 255);
        #nfill = cv2.bitwise_not(fill)
        #ret = thr | nfill

        ke = np.ones((3,3), np.uint8)
        kc = np.ones((5,5), np.uint8)

        ret = cv2.medianBlur(thr, 5)
        ret = cv2.erode(ret, ke)
        ret = cv2.morphologyEx(ret, cv2.MORPH_CLOSE, kc, iterations=1)

        return ret

    def _valleyDetection(self, binary, edge):

        b = 10
        a = 10 + b
        u = 10 + a
        v = list()
        ePts = list()
        valleys = list()

        # Assign the edge points
        for i in range(u, edge.shape[0]-u):
            for j in range(u, edge.shape[1]-u):
                if edge[i][j] == 255:
                    ePts.append((i,j))

        # Iterate through the edge points in order
        # to find a valley between two fingers. It
        # must satisfy each one of the 4 conditions
        for e in ePts:
            if self._checkValleyCandidate(binary, e, b, 4, 2):
                if self._checkValleyCandidate(binary, e, a, 8, 4):
                    if self._checkValleyCandidate(binary, e, u, 16, 7):
                        if self._checkNonHandDirection(binary, ePts, e, 40):
                            valleys.append((e[1],e[0]))

        # Postprocessing: get rid of valley points that
        # are too close to each other
        #if len(valleys) > 2:
        #    temp = list()
        #    prev = valleys[0]

        #    for nxt in valleys[1:-1]:
        #        x = nxt[0]-prev[0]
        #        y = nxt[1]-prev[1]
        #        d = np.sqrt(x*x+ y*y)

        #        if d <= 10:
        #            temp.append(prev)
        #        else:
        #            v.append(temp[int(len(temp)/2)])
        #            temp[:] = [nxt]

        #        prev = nxt
        v = valleys

        return v

    def _getTestCircumference():
        pass

    def _getNonHandDirection():
        pass

    # TODO: Better method of numerical differentiation
    def _checkNonHandDirection(self, binary, edge, candidate, lsize):
        # Inputs
        #    - binary:        Binary image
        #    - candidate:     Valley candidate point
        #    - lsize:         Size in pixels of the check line towards the non-hand direction
        ret = True
        #x0 = candidate[0]
        #y0 = candidate[1]
        #roi = binary[(x0-20):(x0+20),(y0-20):(y0+20)]

        #dx = cv2.Sobel(roi, cv2.CV_64F,1,0,ksize=5)
        #dy = cv2.Sobel(roi, cv2.CV_64F,0,1,ksize=5)

        #a = - np.arctan(dy[20,20]/dx[20,20])
        #v = (np.cos(a), np.sin(a))

        idx0 = edge.index(candidate)
        idx1 = idx0+3

        if idx1 > len(edge):
            idx1 = idx0-3

        x0 = float(edge[idx0][0])
        y0 = float(edge[idx0][1])
        x1 = float(edge[idx1][0]) + np.finfo(float).eps
        y1 = float(edge[idx1][1]) + np.finfo(float).eps

        df = (y1 - y0)/(x1 - x0)
        a = np.arctan(df) + np.pi/2
        v = (np.cos(a), np.sin(a))

        shape = binary.shape

        for i in range(0,lsize):
            x = int(x0 + i*v[0])
            y = int(y0 + i*v[1])
            if (x >= shape[0]) or (y >= shape[1]):
                ret = False
                break
            elif binary[x][y] == 255:
                ret = False
                break

        return ret

    def _checkValleyCandidate(self, img, candidate, radius, nPoints, nCheckPoints):
        # Inputs:
        #    - img:           Image that is beign tested
        #    - candidate:     Valley candidate point. Center of the test circunference
        #    - radius:        Test circumference radius
        #    - nPoints:       Number of points of the test circumference
        #    - nCheckPoints:  Maximum number of consecutive non-hand region checks to consider
        #                     a candidate point as a valley
        ret = False

        x = candidate[0] + np.array(radius * np.cos(np.arange(nPoints) * 2*np.pi/nPoints), np.int)
        y = candidate[1] + np.array(radius * np.sin(np.arange(nPoints) * 2*np.pi/nPoints), np.int)
        p = [img[u][w] for u,w in zip(x,y)]

        # Iterate through the test cicumference two times and
        # check each point for a hand or a non-hand region
        # counting the occurences
        checkCount = 0
        maxCount = 0
        pIterate = p
        pIterate.extend(p)

        for nextP in pIterate:

            if nextP == 0:
                checkCount += 1
                if checkCount > maxCount:
                    maxCount = checkCount
            else:
                checkCount = 0

        if (maxCount > 0) and (maxCount < nCheckPoints):
            ret = True

        return ret
