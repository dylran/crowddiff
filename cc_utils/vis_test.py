import os
from PIL import Image
import argparse
import numpy as np
import torch as th
from einops import rearrange
import cv2
from glob import glob
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import torch.nn as nn


def get_arg_parser():
    parser = argparse.ArgumentParser('Parameters for the evaluation', add_help=False)

    parser.add_argument('--data_dir', default='primary_datasets/shtech_A/test_data/images', type=str,
                        help='Path to the original image directory')
    parser.add_argument('--result_dir', default='experiments/cc-qnrf-1', type=str,
                        help='Path to the diffusion results directory')
    parser.add_argument('--output_dir', default='experiments/evaluate-qnrf', type=str,
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
    # data_dir = args.data_dir
    result_dir = args.result_dir
    output_dir = args.output_dir
    image_size = args.image_size

    config(output_dir)

    # img_list = os.listdir(data_dir)
    result_list = os.listdir(result_dir)
    result_list = glob(os.path.join(result_dir,'*.jpg'))

    kernel = create_density_kernel(11,2)
    normalizer = kernel.max()
    kernel = GaussianKernel(kernel_weights=kernel, device='cpu')

    mae, mse = 0, 0

    for index, name in enumerate(result_list):
        image = np.asarray(Image.open(name).convert('RGB'))
        image = np.split(image, 3, axis=1)
        gt, image, density = image[0], image[1], image[2]
        
        gt = gt[:,:,0]
        density = density[:,:,0]

        gt = 1.*(gt>125)
        density = 1.*(density>125)

        density = th.tensor(density)
        density = density.unsqueeze(0).unsqueeze(0)
        density = kernel(density).detach().numpy()
        # density = th.stack(density_maps)
        # density = density.transpose(1,2,0)

        gt = th.tensor(gt)
        gt = gt.unsqueeze(0).unsqueeze(0)
        gt = kernel(gt).detach().numpy()

        density = ((density/normalizer).clip(0,1)*255).astype(np.uint8)
        gt = ((gt/normalizer).clip(0,1)*255).astype(np.uint8)

        # density = np.repeat(density[:,:,np.newaxis],axis=-1,repeats=3)
        # gt = np.repeat(gt[:,:,np.newaxis], axis=-1, repeats=3)

        # req_image = np.concatenate([density, image, gt], axis=1)
        req_image = [density, image, gt]
        name = os.path.basename(name)
        # assert False
        # cv2.imwrite(os.path.join(output_dir,name), req_image[:,:,::-1])
        fig, ax = plt.subplots(ncols=3, nrows=1, tight_layout=True)
        for index, figure in enumerate(req_image):
            ax[index].imshow(figure)
            ax[index].axis('off')
        # plt.show()
        # assert False
        # extent = plt.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(output_dir,name).replace('jpg','png'),bbox_inches='tight')
        plt.close()
        # Image.fromarray(req_image, mode='RGB').show()
    #     plt.figure()
    #     plt.imshow(req_image)
    # plt.show()
        # print(image.dtype)

        # gt = th.stack(gt)
        # gt = gt.transpose(1,2,0)
        
        # print(density.shape, gt.shape)
        

        # assert False

    #     crops, gt_count = get_crops(result_dir, name.split('_')[-1], image, result_list)

    #     pred = crops[:,:, image_size:-image_size,:].mean(-1)
    #     gt = crops[:,:, -image_size:,:].mean(-1)
        
    #     pred = remove_background(pred)

    #     pred = combine_crops(pred, image, image_size)
    #     gt = combine_crops(gt, image, image_size)

    #     pred_count = get_circle_count(pred)

    #     pred = np.repeat(pred[:,:,np.newaxis],3,-1)
    #     gt = np.repeat(gt[:,:,np.newaxis],3,-1)
    #     image = np.asarray(image)

    #     gap = 5
    #     red_gap = np.zeros((image.shape[0],gap,3), dtype=int)
    #     red_gap[:,:,0] = np.ones((image.shape[0],gap), dtype=int)*255

    #     image = np.concatenate([image, red_gap, pred, red_gap, gt], axis=1)
    #     # Image.fromarray(image, mode='RGB').show()
    #     cv2.imwrite(os.path.join(output_dir,name), image[:,:,::-1])
        
    #     mae += abs(pred_count-gt_count)
    #     mse += abs(pred_count-gt_count)**2

    #     if index == -1:
    #         print(name)
    #         break

    # print(f'mae: {mae/(index+1) :.2f} and mse: {np.sqrt(mse/(index+1)) :.2f}')


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


def create_density_kernel(kernel_size, sigma):
    
    kernel = np.zeros((kernel_size, kernel_size))
    mid_point = kernel_size//2
    kernel[mid_point, mid_point] = 1
    kernel = gaussian_filter(kernel, sigma=sigma)

    return kernel


class GaussianKernel(nn.Module):

    def __init__(self, kernel_weights, device):
        super().__init__()
        self.kernel = nn.Conv2d(1,1,kernel_weights.shape, bias=False, padding=kernel_weights.shape[0]//2)
        kernel_weights = th.tensor(kernel_weights).unsqueeze(0).unsqueeze(0)
        with th.no_grad():
            self.kernel.weight = nn.Parameter(kernel_weights)
    
    def forward(self, density):
        return self.kernel(density).squeeze()


if __name__=='__main__':
    parser = argparse.ArgumentParser('Combine the results and evaluate', parents=[get_arg_parser()])
    args = parser.parse_args()
    main(args)