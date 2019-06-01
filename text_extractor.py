import cv2
from matplotlib import pyplot as plt
from random import randint

class ReceiptTextExtractor:

    debug_mode = False

    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        pass

    def cntY(self, elem):
        x,y,w,h = cv2.boundingRect(elem)
        return y

    def extract_text(self, img):
        img_height, img_width, img_channels = img.shape

        #find contours that are text sized
        smallTextSize = 100
        largeTextSize = 10000
        #these contours are our targets
        imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,10)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        sizes = []
        targets = []
        for cnt in contours:
            size = cv2.contourArea(cnt)
            if size >= smallTextSize and size <= largeTextSize:
                sizes.append(size)
                targets.append(cnt)
        
        #sort target contours on y height in image
        targets = sorted(targets, key=self.cntY)
        prevY = 0
        yLean = 70
        
        #group by line
        groups = []
        line_group = []
        for target in targets:
            x,y,w,h = cv2.boundingRect(target)
            if prevY < y-yLean or prevY > y+yLean:
                prevY = y
                groups.append(line_group)
                line_group = []
            else:
                line_group.append(target)
            
        groups.append(line_group)

        for group in groups:
            color = (100,50,randint(100, 255))
            smallestX = 100000
            largestX = 0
            smallestY = 100000
            largestY = 0
            for cnt in group:
                x,y,w,h = cv2.boundingRect(cnt)
                
                if(x < smallestX):
                    smallestX = x
                elif(x > largestX):
                    largestX = x
                if(y < smallestY):
                    smallestY = y
                elif(y > largestY):
                    largestY = y
            
            smallestX = max(0,smallestX - 50)
            smallestY = max(0,smallestY - 50)
            largestX = min(largestX + 50, img_width)
            largestY = min(largestY + 50, img_height)

            height = largestY-smallestY
            width = largestX-smallestX
            P1 = (smallestX,smallestY)
            P2 = (smallestX+width,smallestY+height)
            cv2.rectangle(img,P1,P2,color,4)
            segment = img[smallestY:smallestY+height, smallestX:smallestX+width]
            seg_height, seg_width, seg_chan = segment.shape
            # if(seg_height != 0 and seg_width != 0):
                # plt.imshow(segment, cmap = 'gray')
                # plt.show()
        