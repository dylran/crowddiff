import numpy as np
import cv2
import os
import argparse

from PIL import Image


def get_arg_parser():
    parser = argparse.ArgumentParser('Count circles in a density map', add_help=False)

    # Dataset parameters
    parser.add_argument('--data_dir', default='./results', type=str,
                        help='Path to the groundtruth density maps')
    parser.add_argument('--result_dir', default='', type=str,
                        help='Path to the predicted density maps')
    
    # Output parameters
    parser.add_argument('--output_dir', default='', type=str,
                        help='Path to the output of the code')
    
    # kernel parameters
    parser.add_argument('--thresh', default=200, type=int,
                        help='Threshold value for the kernel')

    return parser


def main(args):
    path = args.data_dir

    img_list = os.listdir(path)
    for name in img_list:
        image = cv2.imread(os.path.join(path, name),0)
        pred = image[:,256:-256]
        gt = image[:,-256:]
        
        pred_count = get_circle_count(pred, args.thresh)
        gt_count = get_circle_count(gt, args.thresh)

        print(name, ' pred: ',pred_count, ' gt: ',gt_count)
        # break
    pass


def get_circle_count(image, threshold, draw=False):

    # Denoising
    denoisedImg = cv2.fastNlMeansDenoising(image)

    # Threshold (binary image)
    # thresh – threshold value.
    # maxval – maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
    # type – thresholding type
    th, threshedImg = cv2.threshold(denoisedImg, 200, 255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU) # src, thresh, maxval, type

    # Perform morphological transformations using an erosion and dilation as basic operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    morphImg = cv2.morphologyEx(threshedImg, cv2.MORPH_OPEN, kernel)

    # Find and draw contours
    contours, _ = cv2.findContours(morphImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if draw:
        contoursImg = cv2.cvtColor(morphImg, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(contoursImg, contours, -1, (255,100,0), 3)

        Image.fromarray(contoursImg, mode='RGB').show()

    return len(contours)-1 # remove the outerboarder countour

if __name__=='__main__':
    parser = argparse.ArgumentParser('Count the number of circles in a density', parents=[get_arg_parser()])
    args = parser.parse_args()
    main(args)


# for dirname in os.listdir("images/"):

#     for filename in os.listdir("images/" + dirname + "/"):

#         # Image read
#         img = cv2.imread("images/" + dirname + "/" + filename, 0)

#         # Denoising
#         denoisedImg = cv2.fastNlMeansDenoising(img)

#         # Threshold (binary image)
#         # thresh – threshold value.
#         # maxval – maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
#         # type – thresholding type
#         th, threshedImg = cv2.threshold(denoisedImg, 200, 255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU) # src, thresh, maxval, type

#         # Perform morphological transformations using an erosion and dilation as basic operations
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
#         morphImg = cv2.morphologyEx(threshedImg, cv2.MORPH_OPEN, kernel)

#         # Find and draw contours
#         contours, hierarchy = cv2.findContours(morphImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         contoursImg = cv2.cvtColor(morphImg, cv2.COLOR_GRAY2RGB)
#         cv2.drawContours(contoursImg, contours, -1, (255,100,0), 3)

#         cv2.imwrite("results/" + dirname + "/" + filename + "_result.tif", contoursImg)
#         textFile = open("results/results.txt","a")
#         textFile.write(filename + " Dots number: {}".format(len(contours)) + "\n")
#         textFile.close()
