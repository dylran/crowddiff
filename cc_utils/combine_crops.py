import os
import argparse
from glob import glob

import numpy as np

from PIL import Image


def get_arg_parser():
    parser = argparse.ArgumentParser('Combine image crops for test image', add_help=False)

    # Datasets path
    parser.add_argument('--data_dir', default='', type=str,
                        help='Path to the original dataset')
    parser.add_argument('--den_dir', default='', type=str,
                        help='Path to the density results of cropped images')
    parser.add_argument('--output_dir', default='', type=str,
                        help='Path to save the results')

    return parser


def main(args):

    # create output folder
    try:
        os.mkdir(args.output_dir)
    except FileExistsError:
        pass

    # load the image file list
    img_list = sorted(glob(os.path.join(args.data_dir,'*.jpg')))

    for index in range(1,len(os.listdir(args.data_dir))+1):
        h_pos, w_pos = get_crop_pos(args, index)
        density = get_density_maps(args, index, h_pos.size*w_pos.size)
        
        density = combine_crops(density, h_pos, w_pos, image_size=256)
        density = Image.fromarray(density, mode='L')

        path = os.path.join(args.output_dir, str(index)+'.jpg')
        density.save(path)
        break
        

def combine_crops(crops, h_pos, w_pos, image_size):
    density = np.zeros((h_pos[-1]+image_size, w_pos[-1]+image_size), dtype=np.uint8)
    count = 0
    for start_h in h_pos:
        for start_w in w_pos:
            end_h = start_h + image_size
            end_w = start_w + image_size
            density[start_h:end_h, start_w:end_w] = crops[count]
            count += 1
    return density


def get_crop_pos(args, index, image_size=256):
    
    path = os.path.join(args.data_dir,'IMG_'+str(index)+'.jpg')
    image = Image.open(path)
    
    image = resize_rescale_image(image, image_size)

    w,h = image.size
    h_pos = int((h-1)//image_size) + 1
    w_pos = int((w-1)//image_size) + 1

    end_h = h - image_size
    end_w = w - image_size

    start_h_pos = np.linspace(0, end_h, h_pos, dtype=int)
    start_w_pos = np.linspace(0, end_w, w_pos, dtype=int)

    return start_h_pos, start_w_pos


def resize_rescale_image(image, image_size):

    w, h = image.size # image is a PIL
    # check if the both dimensions are larger than the image size
    if h < image_size or w < image_size:
        scale = np.ceil(max(image_size/h, image_size/w))
        h, w = int(scale*h), int(scale*w)
    
    return image.resize((w,h))


def get_density_maps(args, index, crops):
    density = []
    for sub_index in range(crops):
        path = os.path.join(args.den_dir, str(index)+'-'+str(sub_index+1)+'.jpg')
        density.append(load_density_map(path))
    density = np.asarray(density)
    
    return density


def load_density_map(path):

    density = np.asarray(Image.open(path).convert('L'))
    return density


if __name__=='__main__':
    parser = argparse.ArgumentParser('Combine crop density images', parents=[get_arg_parser()])
    args = parser.parse_args()
    main(args)