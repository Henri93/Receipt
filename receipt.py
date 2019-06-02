import cv2
from matplotlib import pyplot as plt
from receipt_preprocessor import ReceiptPreprocessor
from text_extractor import ReceiptTextExtractor

def main():
    folder_name = "receipt_imgs"
    imgs = ['rec0.jpg', 'rec1.jpg', 'rec2.jpg', 'rec4.jpg', 'rec5.jpeg', 'rec7.jpg', 'rec9.jpg']
#    imgs = ['rec2.jpg']

    preprocessor = ReceiptPreprocessor(debug_mode=False)
    text_extractor = ReceiptTextExtractor(debug_mode=True)


    for img_name in imgs:
        #setup window
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 600,600)

        #read image
        img = cv2.imread(folder_name + "/" + img_name, 1)
        
        #save original
        original = img.copy()

        img = preprocessor.preprocess(img)

        text_extractor.extract_text(img)

        #show original and preprocessed image
        # plt.subplot(121),plt.imshow(original, cmap = 'gray')
        # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122),plt.imshow(img, cmap='gray')
        # plt.title('Output Image'), plt.xticks([]), plt.yticks([])
        # plt.show()

        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

main()