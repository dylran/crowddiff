import os
from PIL import Image
import argparse
import numpy as np
import torch as th
from einops import rearrange
import cv2


def get_arg_parser():
    parser = argparse.ArgumentParser('Parameters for the evaluation', add_help=False)

    parser.add_argument('--data_dir', default='primary_datasets/shtech_A/test_data/images', type=str,
                        help='Path to the original image directory')
    parser.add_argument('--result_dir', default='experiments/shtech_A', type=str,
                        help='Path to the diffusion results directory')
    parser.add_argument('--output_dir', default='experiments/evaluate', type=str,
                        help='Path to the output directory')
    parser.add_argument('--image_size', default=256, type=int,
                        help='Crop size')

    return parser


def config(dir):
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass


def main(args):
    data_dir = args.data_dir
    result_dir = args.result_dir
    output_dir = args.output_dir
    image_size = args.image_size

    config(output_dir)

    img_list = os.listdir(data_dir)
    result_list = os.listdir(result_dir)

    mae, mse = 0, 0

    for index, name in enumerate(img_list):
        image = Image.open(os.path.join(data_dir, name)).convert('RGB')

        crops, gt_count = get_crops(result_dir, name.split('_')[-1], image, result_list)

        pred = crops[:,:, image_size:-image_size,:].mean(-1)
        gt = crops[:,:, -image_size:,:].mean(-1)
        
        pred = remove_background(pred)

        pred = combine_crops(pred, image, image_size)
        gt = combine_crops(gt, image, image_size)

        pred_count = get_circle_count(pred)

        pred = np.repeat(pred[:,:,np.newaxis],3,-1)
        gt = np.repeat(gt[:,:,np.newaxis],3,-1)
        image = np.asarray(image)

        gap = 5
        red_gap = np.zeros((image.shape[0],gap,3), dtype=int)
        red_gap[:,:,0] = np.ones((image.shape[0],gap), dtype=int)*255

        image = np.concatenate([image, red_gap, pred, red_gap, gt], axis=1)
        # Image.fromarray(image, mode='RGB').show()
        cv2.imwrite(os.path.join(output_dir,name), image[:,:,::-1])
        
        mae += abs(pred_count-gt_count)
        mse += abs(pred_count-gt_count)**2

        if index == -1:
            print(name)
            break

    print(f'mae: {mae/(index+1) :.2f} and mse: {np.sqrt(mse/(index+1)) :.2f}')


def remove_background(crops):
    def count_colors(image):

        colors_count = {}
        # Flattens the 2D single channel array so as to make it easier to iterate over it
        image = image.flatten()
        # channel_g = channel_g.flatten()  # ""
        # channel_r = channel_r.flatten()  # ""

        for i in range(len(image)):
            I = str(int(image[i]))
            if I in colors_count:
                colors_count[I] += 1
            else:
                colors_count[I] = 1
        
        return int(max(colors_count, key=colors_count.__getitem__))+5

    for index, crop in enumerate(crops):
        count = count_colors(crop)
        crops[index] = crop*(crop>count)

    return crops


def get_crops(path, index, image, result_list, image_size=256):
    w, h = image.size
    ncrops = ((h-1+image_size)//image_size)*((w-1+image_size)//image_size)
    crops = []

    gt_count = 0
    for _ in range(ncrops):
        crop = f'{index.split(".")[0]}-{_+1}'
        for _ in result_list:
            if _.startswith(crop):
                break

        crop = Image.open(os.path.join(path,_))
        # crop = Image.open()
        crops.append(np.asarray(crop))
        gt_count += float(_.split(' ')[-1].split('.')[0])
    crops = np.stack(crops)
    if len(crops.shape) < 4:
        crops = np.expand_dims(crops, 0)
    
    return crops, gt_count
    

def combine_crops(density, image, image_size):
    w,h = image.size
    p1 = (h-1+image_size)//image_size
    density = th.from_numpy(density)
    density = rearrange(density, '(p1 p2) h w-> (p1 h) (p2 w)', p1=p1)
    den_h, den_w = density.shape

    start_h, start_w = (den_h-h)//2, (den_w-w)//2
    end_h, end_w = start_h+h, start_w+w
    density = density[start_h:end_h, start_w:end_w]
    # print(density.max(), density.min())
    # density = density*(density>0)
    # assert False
    return density.numpy().astype(np.uint8)


def get_circle_count(image, threshold=0, draw=False):

    # Denoising
    denoisedImg = cv2.fastNlMeansDenoising(image)

    # Threshold (binary image)
    # thresh – threshold value.
    # maxval – maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
    # type – thresholding type
    th, threshedImg = cv2.threshold(denoisedImg, threshold, 255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU) # src, thresh, maxval, type

    # Perform morphological transformations using an erosion and dilation as basic operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    morphImg = cv2.morphologyEx(threshedImg, cv2.MORPH_OPEN, kernel)

    # Find and draw contours
    contours, _ = cv2.findContours(morphImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if draw:
        contoursImg = cv2.cvtColor(morphImg, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(contoursImg, contours, -1, (255,100,0), 3)

        Image.fromarray(contoursImg, mode='RGB').show()

    return max(len(contours)-1,0) # remove the outerboarder countour


# def get_circle_count_and_sample(samples, thresh=0):

    count = [], []
    for sample in samples:
        pred_count = get_circle_count(sample. thresh)
        mae.append(th.abs(pred_count-gt_count))
        count.append(th.tensor(pred_count))
    
    mae = th.stack(mae)
    count = th.stack(count)

    index = th.argmin(mae)

    return index, mae[index], count[index], gt_count


if __name__=='__main__':
    parser = argparse.ArgumentParser('Combine the results and evaluate', parents=[get_arg_parser()])
    args = parser.parse_args()
    main(args)