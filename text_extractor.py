import cv2
from matplotlib import pyplot as plt
from random import randint
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class ReceiptTextExtractor:
    """ 
    Helper Class responsible for parsing text from receipt image 
    """

    debug_mode = False

    def __init__(self, debug_mode=False):
        """ 
        Default Constructor

        Parameters: 
        debug_mode (bool): true for verbose logging

        """
        
        self.debug_mode = debug_mode
        pass

    def find_text_boxes(self,img):
        """ 
        Find contours that are text sized

        Parameters:
        img (image): image to find contours in

        Returns:
        ([Contours]): text size contours sorted by Y position

        """

        smallTextSize = 100
        largeTextSize = 16000
        
        #these contours are our targets
        imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,10)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        sizes = []
        targets = []

        #check bounding box is within size limits to be target
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            P1 = (x,y)
            P2 = (x+w,y+h)
            size = w*h
            sizes.append(size)
            if size >= smallTextSize and size <= largeTextSize:
                targets.append((x, y, w, h))
        
        #sort target contours on y height in image
        targets = sorted(targets, key=self.cntY)
        return targets

    def color_groups(self,img,text_boxes):
        """ 
        Draws colored contours in image to represent Y position

        Parameters:
        img (image): image to draw contours on
        text_boxes ([Contours]): contours around text areas

        """

        #top text box
        t_x, t_y, t_w, t_h = text_boxes[0]

        #bottom text box
        b_x, b_y, b_w, b_h = text_boxes[-1]

        #map y position to color
        OldRange = (b_y - t_y)  
        NewRange = (255 - 0)  

        #draw colored box for each contour
        for text_box in text_boxes:
            x, y, w, h = text_box
            NewValue = (((y - t_y) * NewRange) / OldRange)
            color = (NewValue,150,NewValue/2)
            P1 = (x,y)
            P2 = (x+w,y+h)
            cv2.rectangle(img,P1,P2,color,6)

    def cntY(self, elem):
        """ 
        Element Comparator by Y position

        Returns: 
        (int): Y position

        """
        x,y,w,h = elem
        return y

    def cntX(self,elem):
        """ 
        Element Comparator by X position

        Returns: 
        (int): X position

        """
        x,y,w,h = elem
        return x

    def group_text(self,img,text_boxes):
        """ 
        Group Text Boxes withing Y threshold together as
        One rectangle contour spanning all boxes

        Parameters:
        img (image): image to group contours from
        text_boxes ([Contours]): contours around text areas

        Returns:
        ([Rectangle]): segments of text with similar Y position

        """

        segments = []
        groups = []
        current_group = []
        similar_thres = 100

        for text_box in text_boxes:
            x, y, w, h = text_box

            #if current line is empty start new group
            if len(current_group) == 0:
                current_group.append(text_box)
                continue
            
            #first elemnt in current group
            r_x, r_y, r_w, r_h = current_group[0]
            
            if abs(y-r_y) <= similar_thres:
                #add to current group if withing threshold
                current_group.append(text_box)
            else:
                #otherwise create a new group
                groups.append(current_group)
                current_group = [text_box]

        #create one rectangle contour spanning all boxes in each group
        for text_group in groups:
            top = text_group[0]
            bottom = text_group[-1]
            
            group = sorted(text_group, key=self.cntX)
            left = group[0]
            right = group[-1]

            padding = 25

            #x, y, w, h
            #P1---P3
            #|    |
            #P2---P4
            smallestX = left[0]-padding
            largestX = right[0]+right[2]+padding
            smallestY = top[1]-padding
            largestY = bottom[1]+bottom[3]+padding

            P1 = (smallestX,smallestY)
            P4 = (largestX,largestY)
            if debug_mode:
                cv2.rectangle(img,P1,P4,(0,255,0),6)

            segment = img[smallestY:largestY, smallestX:largestX]
            segments.append(segment)

        return segments
            

    def extract_text(self, img):
        """ 
        Parse lines of text from an image

        First find the text boxes
        Then find segments of text on same line
        Use OCR to convert each segment into a String

        Parameters:
        img (image): image to extract text lines from

        """
        img_height, img_width, img_channels = img.shape

        text_boxes = self.find_text_boxes(img)

        segments = self.group_text(img,text_boxes)

        for segment in segments:
            text = pytesseract.image_to_string(Image.fromarray(segment))
            if "total" in text.lower():
                print(text)
        print("-----------------------")
