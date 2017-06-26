import os
from Hand import Hand

import numpy as np

dataPath = '../dataset'
outPath = '../output'

if not os.path.isdir(outPath):
    os.mkdir(outPath)

#for hand in os.listdir(dataPath):
handPics = os.listdir(dataPath)
#hand = Hand(dataPath + '/' + handPics[0], "RIGHT")
#hand = Hand(dataPath + '/' + handPics[1], "RIGHT")
##hand = Hand(dataPath + '/' + handPics[2], "RIGHT")
#hand = Hand(dataPath + '/' + handPics[3], "RIGHT")
#hand = Hand(dataPath + '/' + handPics[4], "RIGHT")
#hand = Hand(dataPath + '/' + handPics[5], "RIGHT")
#hand = Hand(dataPath + '/' + handPics[6], "RIGHT")
#hand = Hand(dataPath + '/' + handPics[7], "RIGHT")
##hand = Hand(dataPath + '/' + handPics[8], "RIGHT")

for i in (0,1,3,4,5,6,7,9,10,11,12,13,14,16,17,18,19,20,21,22,23):
    hand = Hand(dataPath + '/' + handPics[i], "RIGHT")

#x = np.arange(10)
#y = x[x > 5]
#y = np.arange(20)
#x = np.array(10*np.cos(np.arange(4)*np.pi/2), np.int)
#print y
