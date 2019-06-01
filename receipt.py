import cv2
from matplotlib import pyplot as plt
from receipt_preprocessor import ReceiptPreprocessor
from random import randint

def cntY(elem):
    x,y,w,h = cv2.boundingRect(elem)
    return y

def main():
    preprocessor = ReceiptPreprocessor(debug_mode=True)
    folder_name = "receipt_imgs"
    # imgs = ['rec0.jpg', 'rec1.jpg', 'rec2.jpg', 'rec4.jpg', 'rec5.jpeg', 'rec7.jpg', 'rec9.jpg']
    imgs = ['rec2.jpg']
    for img_name in imgs:
        #setup window
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 600,600)

        #read image
        img = cv2.imread(folder_name + "/" + img_name, 1)
        img_height, img_width, img_channels = img.shape

        #save original
        original = img.copy()

        img = preprocessor.preprocess(img)

        # cv2.imshow("image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        #find text test
        imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # imgray = cv2.GaussianBlur(imgray,(5,5),0)
        thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,10)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        sizes = []
        targets = []
        for cnt in contours:
            size = cv2.contourArea(cnt)
            if size >= 150 and size <= 10000:
                sizes.append(size)
                targets.append(cnt)
        
        targets = sorted(targets, key=cntY)
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
            # cv2.rectangle(img,P1,P2,color,4)
            segment = img[smallestY:smallestY+height, smallestX:smallestX+width]
            seg_height, seg_width, seg_chan = segment.shape
            if(seg_height != 0 and seg_width != 0):
                plt.imshow(segment, cmap = 'gray')
                plt.show()

        
        # num_bins = 50
        # n, bins, patches = plt.hist(sizes, num_bins, facecolor='blue', alpha=0.5)
        # plt.show()

        plt.imshow(img, cmap = 'gray')
        plt.show()

        #show original and preprocessed image
        # plt.subplot(121),plt.imshow(original, cmap = 'gray')
        # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122),plt.imshow(thresh, cmap='gray')
        # plt.title('Output Image'), plt.xticks([]), plt.yticks([])
        # plt.show()

        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

main()