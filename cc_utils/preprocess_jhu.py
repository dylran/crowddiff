import torch
import os
import glob
import argparse

import pandas as pd
import numpy as np
import torch.nn as nn

from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from einops import rearrange

import cv2
import sys

import matplotlib.pyplot as plt



def get_arg_parser():
    parser = argparse.ArgumentParser('Prepare image and density datasets', add_help=False)

    # Datasets path
    parser.add_argument('--dataset', default='jhu_plus')
    parser.add_argument('--data_dir', default='primary_datasets/', type=str,
                        help='Path to the original dataset')
    parser.add_argument('--mode', default='train', type=str,
                        help='Indicate train or test folders')
    
    # Output path
    parser.add_argument('--output_dir', default='datasets/intermediate', type=str,
                        help='Path to save the results')
    
    # Gaussian kernel size and kernel variance
    parser.add_argument('--kernel_size', default='', type=str,
                        help='Size of the Gaussian kernel')
    parser.add_argument('--sigma', default='', type=str,
                        help='Variance of the Gaussian kernel')
    
    # Crop image parameters
    parser.add_argument('--image_size', default=256, type=int,
                        help='Size of the crop images')
    
    # Device parameter
    parser.add_argument('--ndevices', default=4, type=int)

    # Image output
    parser.add_argument('--with_density', action='store_true')

    # count bound
    parser.add_argument('--lower_bound', default=0, type=int)
    parser.add_argument('--upper_bound', default=np.Inf, type=int)

    # condition parameter
    parser.add_argument('--weather', default='clear', type=str)

    return parser


def create_q_samples(image, num_steps, maxvar, minvar):
    import matplotlib.pyplot as plt
    plt.imshow(image[:,:,0]/0.04)
    plt.axis('off')
    plt.savefig(f'experiments/final/q sample transparent/0.jpg',dpi=1200,format='jpg',
                bbox_inches='tight',pad_inches=0.0)
    img = Image.open(f'experiments/final/q sample transparent/0.jpg')
    img = img.convert('RGBA')
    datas = img.getdata()
    newData = []

    for item in datas:
        newData.append((item[0],item[1],item[2],200))

    img.putdata(newData)
    img.save(f'experiments/final/q sample transparent/0.png')



    q_samples = np.repeat(image, axis=-1, repeats=num_steps+1)/0.04
    beta_schedule = np.linspace(minvar, maxvar, num_steps) * (1000/num_steps)
    for index, beta in enumerate(beta_schedule):
        x_prev = q_samples[:,:,index] *2 -1
        x_cur = np.sqrt(1-beta)*x_prev + np.sqrt(beta)*np.random.randn(x_prev.shape[0],x_prev.shape[1])
        q_samples[:,:,index+1] = (x_cur + 1)*0.5
        plt.imshow(x_cur)
        plt.axis('off')
        plt.savefig(f'experiments/final/q sample transparent/{np.round(beta,2)}.jpg',dpi=1200,format='jpg',
                    bbox_inches='tight',pad_inches=0.0)
        img = Image.open(f'experiments/final/q sample transparent/{np.round(beta,2)}.jpg')
        img = img.convert('RGBA')
        datas = img.getdata()
        newData = []
    
        for item in datas:
            newData.append((item[0],item[1],item[2],200))
    
        img.putdata(newData)
        img.save(f'experiments/final/q sample transparent/{np.round(beta,2)}.png')
        # assert False
        # plt.show()
        
    # assert False
        



def main(args):

    # dataset directiors
    data_dir = os.path.join(args.data_dir, args.dataset)
    mode = args.mode

    # output directory
    output_dir = os.path.join(args.output_dir)

    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass

    # density kernel parameters
    kernel_size_list, sigma_list = get_kernel_and_sigma_list(args)
    
    # normalization constants
    normalizer = 0.008

    # crop image parameters
    image_size = args.image_size

    # device parameter
    device = 'cpu'

    # distribution of crowd count
    crowd_bin = dict()

    with open(os.path.join(data_dir,mode,'image_labels.txt')) as f:
        img_list = [line.rstrip() for line in f]
    
    # img_list = sorted(glob.glob(os.path.join(data_dir,mode,'*.jpg')))
    

    sub_list = setup_sub_folders(img_list, output_dir, ndevices=args.ndevices)

    kernel_list = []
    kernel_list = [create_density_kernel(kernel_size_list[index], sigma_list[index]) for index in range(len(sigma_list))]
    normalizer = [kernel.max() for kernel in kernel_list]

    kernel_list = [GaussianKernel(kernel, device) for kernel in kernel_list]
    weather_index = {'clear':0,
                     'fog':1,
                     'rain':2,
                     'snow':3
                     }
    weather_index = weather_index[args.weather]

    count = 0

    for device, img_list in enumerate(sub_list):
        for file in img_list:
            count += 1
            if count%10==0:
                # print(count)
                pass
            name, weather = file.split(',')[0], int(file.split(',')[3])
            # if weather != weather_index:
            #     continue
            # print(name)
            if name != '0718':
                continue
            
            file = os.path.join(data_dir,mode,'images',name+'.jpg')
            # print(file)
            # load the images and locations
            image = Image.open(file).convert('RGB')

            # check if number of crops are more than 50
            # samples, _,_,_ = create_crops(np.asarray(image)[np.newaxis,:,:,:], args)
            # if samples > 50:                
            #     continue
            # else:
            #     print(name, samples)

            file = file.replace('images','gt').replace('.jpg','.txt')
            with open(file) as f:
                lines = [line.rstrip() for line in f]
            locations = [[int(line.split(' ')[0]),int(line.split(' ')[1])] for line in lines]
            try:
                locations = np.stack(locations)
            except ValueError:
                continue
            
            index = (len(locations)-args.lower_bound)//400
            # try:
            #     if crowd_bin[str(index)] > 0:
            #         continue
            #     crowd_bin[str(index)] += 1
            # except KeyError:
            #     crowd_bin[str(index)] = 1
            #     print(f'new bin: {len(crowd_bin.keys())}')
            

            # resize the image and rescale locations
            if image_size == -1:
                image = np.asarray(image)
            else:
                if mode == 'train' or mode=='test':
                    image, locations = resize_rescale_info(image, locations, image_size)
                else:
                    image = np.asarray(image)
            
            # create dot map
            density = create_dot_map(locations, image.shape)
            
            for _ in range(5):
                random_loc = locations + np.random.randn(locations.shape[0], locations.shape[1])*2
                dot_map = (np.zeros_like(image)*255).astype(np.uint8)
                circle_map = (np.ones_like(image)*255).astype(np.uint8)
                color = (255,0,0)
                if _ == 4:
                    random_loc = locations
                    color = (0,0,255)
                for point in random_loc:
                    print(point)
                    point = (int(point[0]), int(point[1]))
                    print(point)
                    cv2.circle(circle_map, point, radius=3, color=color, thickness=-1)
                    # cv2.circle(dot_map, point, radius=1, color=color, thickness=-1)
                    dot_map[point[1],point[0],0] = 1
                mask = 255 - np.prod(circle_map, axis=-1, keepdims=True)**(1/3)
                dot_map = np.sum(dot_map, axis=-1, keepdims=False).astype(np.float64)
                density = torch.tensor(dot_map)

                density = density.unsqueeze(0).unsqueeze(0)
                density_maps = [kernel(density) for kernel in kernel_list]
                density = torch.stack(density_maps).detach().numpy()
                dot_map = density.transpose(1,2,0)
                plt.imshow(dot_map)


                plt.axis('off')
                plt.savefig(f'experiments/final/q sample transparent/density{_}.jpg',dpi=1200,format='jpg',
                            bbox_inches='tight',pad_inches=0.0)
                img = Image.open(f'experiments/final/q sample transparent/density{_}.jpg')
                img = img.convert('RGBA')
                datas = img.getdata()
                newData = []

                for item in datas:
                    newData.append((item[0],item[1],item[2],200))

                img.putdata(newData)
                img.save(f'experiments/final/q sample transparent/density{_}.png')


                
                circle_map = circle_map[:,:,::-1]
                circle_map = np.concatenate([circle_map,mask], axis=-1)
                
                img = Image.fromarray(circle_map[:,:,:-1].astype(np.uint8)).convert('RGBA')
                datas = img.getdata()
 
                newData = []
            
                for item in datas:
                    if item[0] == 255 and item[1] == 255 and item[2] == 255:
                        newData.append((127, 127, 127, 100))
                    else:
                        newData.append(item)
            
                img.putdata(newData)
                img.save(f'experiments/final/q sample transparent/realize{_}.png', "PNG")
                # cv2.imwrite(f'experiments/final/q sample transparent/realize{_}.jpg',dot_map)
                # plt.imshow(dot_map)
                # plt.axis('off')
                # plt.savefig(f'experiments/final/q sample transparent/realize{_}.png',dpi=1200,
                            # bbox_inches='tight',pad_inches=0.0, transparent=True)
                    # print(dot_map.dtype, dot_map.shape)
            assert False

            density = torch.tensor(density)

            density = density.unsqueeze(0).unsqueeze(0)
            density_maps = [kernel(density) for kernel in kernel_list]
            density = torch.stack(density_maps).detach().numpy()
            density = density.transpose(1,2,0)

            # create image crops
            if image_size == -1:
                images, densities = np.expand_dims(image, 0), np.expand_dims(density, 0)
            else:
                if mode == 'train' or mode == 'test':
                    overlap = 0
                    images = create_overlapping_crops(image, image_size, overlap)
                    densities = create_overlapping_crops(density, image_size, overlap)
                else:
                    images, densities = create_non_overlapping_crops(image, density, image_size)
            
            # print(file)
            index = os.path.basename(file).split('.')[0]

            path = os.path.join(output_dir,f'part_{device+1}',mode)
            den_path = path.replace(os.path.basename(path), os.path.basename(path)+'_den')

            try:
                os.mkdir(path)
                os.mkdir(den_path)
            except FileExistsError:
                pass
            
            q_samples = create_q_samples(density, num_steps=10, maxvar=0.01, minvar=0.0002)

            for sub_index, (image, density) in enumerate(zip(images, densities)):
                if density.sum() < 5:
                    continue
                file = os.path.join(path,str(index)+'-'+str(sub_index+1)+'.jpg')
                if args.with_density:
                    req_image = [(255-(density[:,:,index]/normalizer[index]*255.)).clip(0,255).astype(np.uint8) for index in range(len(normalizer))]
                    req_image = torch.tensor(np.asarray(req_image))
                    req_image = rearrange(req_image, 'c h w -> h (c w)')
                    req_image = req_image.detach().numpy()
                    if len(req_image.shape) < 3:
                        req_image = req_image[:,:,np.newaxis]
                    req_image = np.repeat(req_image,3,-1)
                    image = np.concatenate([image, req_image],axis=1)
                
                image = np.concatenate(np.split(image, 2, axis=1), axis=0) if args.with_density else image
                Image.fromarray(image, mode='RGB').save(file)
                density = rearrange(torch.tensor(density), 'h w c -> h (c w)').detach().numpy()
                file = os.path.join(den_path,str(index)+'-'+str(sub_index+1)+'.csv')
                density = pd.DataFrame(density.squeeze())
                density.to_csv(file, header=None, index=False)
                # print(file)

            

    print(count)
    print(normalizer)
    print(len(crowd_bin.keys()))
    print(crowd_bin)


def create_crops(image, args):
    """Create image crops from the crowd dataset
    inputs: crowd image, density map
    outputs: model_kwargs and crowd count
    """

    # create a padded image
    image = create_padded_image(image, 256)

    return image.shape


def create_padded_image(image, image_size):

    image = image.transpose(0,-1,1,2)
    _, c, h, w = image.shape
    image = torch.tensor(image)
    p1, p2 = (h-1+image_size)//image_size, (w-1+image_size)//image_size
    pad_image = torch.full((1,c,p1*image_size, p2*image_size),0, dtype=image.dtype)

    start_h, start_w = (p1*image_size-h)//2, (p2*image_size-w)//2
    end_h, end_w = h+start_h, w+start_w

    pad_image[:,:,start_h:end_h, start_w:end_w] = image
    pad_image = rearrange(pad_image, 'n c (p1 h) (p2 w) -> (n p1 p2) c h w', p1=p1, p2=p2)

    return pad_image


def get_kernel_and_sigma_list(args):

    kernel_list = [int(item) for item in args.kernel_size.split(' ')]
    sigma_list = [float(item) for item in args.sigma.split(' ')]

    return kernel_list, sigma_list


def get_circle_count(image, normalizer=1, threshold=0, draw=False):

    image = ((image / normalizer).clip(0,1)*255).astype(np.uint8)
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

    return len(contours)-1 # remove the outerboarder countour


def create_dot_map(locations, image_size):

    density = np.zeros(image_size[:-1])

    for a,b in locations:
        x, y = int(a), int(b)
        try:
            density[y,x] = 1.
        except:
            pass
    return density


def create_density_kernel(kernel_size, sigma):
    
    kernel = np.zeros((kernel_size, kernel_size))
    mid_point = kernel_size//2
    kernel[mid_point, mid_point] = 1
    kernel = gaussian_filter(kernel, sigma=sigma)

    return kernel


def resize_rescale_info(image, locations, image_size):

    w,h = image.size
    # check if the both dimensions are larger than the image size
    if h < image_size or w < image_size:
        scale = np.ceil(max(image_size/h, image_size/w))
        h, w = int(scale*h), int(scale*w)
        locations = locations*scale

    # h_scale, w_scale = image_size/h, image_size/w
    # locations[:,0] = locations[:,0]*w_scale
    # locations[:,1] = locations[:,1]*h_scale
    # w,h = image_size, image_size
    # assert False
    
    
    image = image.resize((w,h))
    
    return np.asarray(image), locations

"""
def create_overlapping_crops(image, density, image_size):
    h,w,_ = image.shape
    h_pos = int((h-1)//image_size) + 1
    w_pos = int((w-1)//image_size) + 1

    end_h = h - image_size
    end_w = w - image_size

    start_h_pos = np.linspace(0, end_h, h_pos, dtype=int)
    start_w_pos = np.linspace(0, end_w, w_pos, dtype=int)    
    
    image_crops, density_crops = [], []
    for start_h in start_h_pos:
        for start_w in start_w_pos:
            end_h, end_w = start_h+image_size, start_w+image_size
            image_crops.append(image[start_h:end_h, start_w:end_w,:])
            density_crops.append(density[start_h:end_h, start_w:end_w])

    image_crops = np.asarray(image_crops)
    density_crops = np.asarray(density_crops)

    return image_crops, density_crops
"""

def create_non_overlapping_crops(image, density, image_size):
    
    h, w = density.shape
    h, w = (h-1+image_size)//image_size, (w-1+image_size)//image_size
    h, w = h*image_size, w*image_size
    pad_density = np.zeros((h,w), dtype=density.dtype)
    pad_image = np.zeros((h,w,image.shape[-1]), dtype=image.dtype)

    start_h = (pad_density.shape[0] - density.shape[0])//2
    end_h = start_h + density.shape[0]
    start_w = (pad_density.shape[1] - density.shape[1])//2
    end_w = start_w + density.shape[1]
    
    pad_density[start_h:end_h, start_w:end_w] = density
    pad_image[start_h:end_h, start_w:end_w] = image

    pad_density = torch.tensor(pad_density)
    pad_image = torch.tensor(pad_image)

    pad_density = rearrange(pad_density, '(p1 h) (p2 w) -> (p1 p2) h w', h=image_size, w=image_size).numpy()
    pad_image = rearrange(pad_image, '(p1 h) (p2 w) c -> (p1 p2) h w c', h=image_size, w=image_size).numpy()   

    return pad_image, pad_density


def create_overlapping_crops(image, crop_size, overlap):
    """
    Create overlapping image crops from the crowd image
    inputs: model_kwargs, arguments

    outputs: model_kwargs and crowd count
    """
    
    X_points = start_points(size=image.shape[1],
                            split_size=crop_size,
                            overlap=overlap
                            )
    Y_points = start_points(size=image.shape[0],
                            split_size=crop_size,
                            overlap=overlap
                            )

    image = arrange_crops(image=image, 
                          x_start=X_points, y_start=Y_points,
                          crop_size=crop_size
                          )
    
    return image


def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            if split_size == size:
                break
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


def arrange_crops(image, x_start, y_start, crop_size):
    crops = []
    for i in y_start:
        for j in x_start:
            split = image[i:i+crop_size, j:j+crop_size, :]
            crops.append(split)
    try:
        crops = np.stack(crops)
    except ValueError:
        print(image.shape)
        for crop in crops:
            print(crop.shape)
    # crops = rearrange(crops, 'n b c h w-> (n b) c h w')
    return crops



def setup_sub_folders(img_list, output_dir, ndevices=4):
    per_device = len(img_list)//ndevices
    sub_list = []
    for device in range(ndevices-1):
        sub_list.append(img_list[device*per_device:(device+1)*per_device])
    sub_list.append(img_list[(ndevices-1)*per_device:])
    
    for device in range(ndevices):
        sub_path = os.path.join(output_dir, f'part_{device+1}')
        try:
            os.mkdir(sub_path)
        except FileExistsError:
            pass

    return sub_list


class GaussianKernel(nn.Module):

    def __init__(self, kernel_weights, device):
        super().__init__()
        self.kernel = nn.Conv2d(1,1,kernel_weights.shape, bias=False, padding=kernel_weights.shape[0]//2)
        kernel_weights = torch.tensor(kernel_weights).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            self.kernel.weight = nn.Parameter(kernel_weights)
    
    def forward(self, density):
        return self.kernel(density).squeeze()


if __name__=='__main__':
    parser = argparse.ArgumentParser('Prepare image and density dataset', parents=[get_arg_parser()])
    args = parser.parse_args()
    main(args)