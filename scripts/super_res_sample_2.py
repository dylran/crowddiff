"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import argparse
import os
import glob
import cv2

from PIL import Image
import pandas as pd

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

from einops import rearrange

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from cc_utils.utils import *

from matplotlib import pyplot as plt

import random

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")



def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading data...")
    data = load_data_for_worker(args.data_dir, args.batch_size, args.normalizer, args.pred_channels, args.file)

    logger.log("creating samples...")
    
    for _ in os.listdir(args.data_dir):

        model_kwargs = next(data)       
        data_parameter = DataParameter(model_kwargs, args)
        model_kwargs['low_res'] = model_kwargs['low_res'][:]
        
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        while data_parameter.resample:
        # while data_parameter.cycles < 1:
            data_parameter.update_cycle()
            # set_seed()
            samples = diffusion.p_sample_loop(
                model,
                (model_kwargs['low_res'].size(0), args.pred_channels, model_kwargs['low_res'].size(2), model_kwargs['low_res'].size(3)),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )            
            data_parameter.evaluate(samples, model_kwargs)
            samples = Denormalize(samples.cpu())
            fil_samples = samples.clone()
            for index, _ in enumerate(samples):
                _ = (_.numpy()*255).astype(np.uint8).squeeze()
                samples[index] = th.from_numpy(_).unsqueeze(0)
                _ = remove_background(_)
                fil_samples[index] = th.from_numpy(_).unsqueeze(0)
            sample = data_parameter.combine_overlapping_crops(samples)
            sample = sample.numpy().transpose(-2,-1,0)
            fil_sample = data_parameter.combine_overlapping_crops(fil_samples)
            fil_sample = fil_sample.numpy().transpose(-2,-1,0)
            
            image = data_parameter.combine_overlapping_crops(model_kwargs['low_res'])
            image = Denormalize(image.cpu()).numpy().transpose(-2,-1,0)
            image = (image*255).astype(np.uint8)

            density = data_parameter.density[:,:,np.newaxis]/args.normalizer
            density = (density*255).astype(np.uint8)

            # sample = sample/(sample.max()+1e-12)
            # density = density/(density.max()+1e-12)
            
            req_image = np.concatenate([density, sample, fil_sample], 1)
            req_image = np.repeat(req_image, repeats=3, axis=-1)
            req_image = np.concatenate([image, req_image],1)

            cv2.imwrite(os.path.join(args.log_dir, f'{data_parameter.name}.jpg'), req_image[:,:,::-1])


            # sample = Denormalize(sample).numpy().transpose(-2,-1,-3).squeeze()
            # samples= Denormalize(samples.cpu()).numpy().transpose(0,-2,-1,-3).squeeze()
            # for index, _ in enumerate(samples):
            #     plt.imshow(_)
            #     plt.savefig(os.path.join(args.log_dir, f'{count}_{index}.jpg'))
            # plt.savefig(os.path.join(args.log_dir, f'{count}.jpg'))
            
        # print(samples.shape)
            
        # data_parameter.evaluate(samples, model_kwargs)
        
        
        

        # assert False
        # model_kwargs['low_res'] = crowd_img
        # model_kwargs['gt_count'] = int(np.sum(crowd_count))
        # model_kwargs['crowd_den'] = crowd_den
        # model_kwargs['name'] = name
        # model_kwargs = combine_crops(result, model_kwargs, dims, mae)
        
        # save_visuals(model_kwargs, args)

    logger.log("sampling complete")


def evaluate_samples(samples, model_kwargs, crowd_count, order, result, mae, dims, cycles):

    samples = samples.cpu().numpy()
    for index in range(order.size):
        p_result, p_mae = evaluate_sample(samples[index], crowd_count[order[index]], name=f'{index}_{cycles}')
        if np.abs(p_mae) < np.abs(mae[order[index]]):
            result[order[index]] = p_result
            mae[order[index]] = p_mae
    
    indices = np.where(np.abs(mae[order])>0)
    order = order[indices]
    model_kwargs['low_res'] = model_kwargs['low_res'][indices]

    pred_count = combine_crops(result, model_kwargs, dims, mae)['pred_count']
    del model_kwargs['pred_count'], model_kwargs['result']
    
    resample = False if len(order)==0 else True
    resample = False if np.sum(np.abs(mae[order]))<25 else True

    print(f'mae: {mae}')
    print(f'cum mae: {np.sum(np.abs(mae[order]))} comb mae: {np.abs(pred_count-np.sum(crowd_count))} cycle:{cycles}')

    return model_kwargs, order, result, mae, resample


def evaluate_sample(sample, count, name=None):
    
    sample = sample.squeeze()
    sample = (sample+1)
    sample = (sample/(sample.max()+1e-8))*255
    sample = sample.clip(0,255).astype(np.uint8)
    sample = remove_background(sample)
    
    pred_count = get_circle_count(sample, name=name, draw=True)

    return sample, pred_count-count


def remove_background(crop):
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

    count = count_colors(crop)
    crop = crop*(crop>count)

    return crop


def get_circle_count(image, threshold=0, draw=False, name=None):

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
        contoursImg = np.zeros_like(morphImg)
        contoursImg = np.repeat(contoursImg[:,:,np.newaxis],3,-1)
        for point in contours:
            x,y = point.squeeze().mean(0)
            if x==127.5 and y==127.5:
                continue
            cv2.circle(contoursImg, (int(x),int(y)), radius=3, thickness=-1, color=(255,255,255))
        threshedImg = np.repeat(threshedImg[:,:,np.newaxis], 3,-1)
        morphImg = np.repeat(morphImg[:,:,np.newaxis], 3,-1)
        image = np.concatenate([contoursImg, threshedImg, morphImg], axis=1)
        cv2.imwrite(f'experiments/target_test/{name}_image.jpg', image)
    return max(len(contours)-1,0) # remove the boarder


def create_crops(model_kwargs, args):
    
    image = model_kwargs['low_res']
    density = model_kwargs['high_res']

    model_kwargs['dims'] = density.shape[-2:]

    # create a padded image
    image = create_padded_image(image, args.large_size)
    density = create_padded_image(density, args.large_size)

    model_kwargs['low_res'] = image
    model_kwargs['high_res'] = density

    model_kwargs['crowd_count'] = th.sum((model_kwargs['high_res']+1)*0.5*args.normalizer, dim=(1,2,3)).cpu().numpy()
    model_kwargs['order'] = np.arange(model_kwargs['low_res'].size(0))

    model_kwargs = organize_crops(model_kwargs)
        
    return model_kwargs


def organize_crops(model_kwargs):
    indices = np.where(model_kwargs['crowd_count']>0)
    model_kwargs['order'] = model_kwargs['order'][indices]
    model_kwargs['low_res'] = model_kwargs['low_res'][indices]

    return model_kwargs
        

def create_padded_image(image, image_size):

    _, c, h, w = image.shape
    p1, p2 = (h-1+image_size)//image_size, (w-1+image_size)//image_size
    pad_image = th.full((1,c,p1*image_size, p2*image_size),-1, dtype=image.dtype)

    start_h, start_w = (p1*image_size-h)//2, (p2*image_size-w)//2
    end_h, end_w = h+start_h, w+start_w

    pad_image[:,:,start_h:end_h, start_w:end_w] = image
    pad_image = rearrange(pad_image, 'n c (p1 h) (p2 w) -> (n p1 p2) c h w', p1=p1, p2=p2)

    return pad_image


def combine_crops(crops, model_kwargs, dims, mae, image_size=256):

    crops = th.tensor(crops).squeeze()
    p1, p2 = (dims[0]-1+image_size)//image_size, (dims[1]-1+image_size)//image_size
    crops = rearrange(crops, '(p1 p2) h w -> (p1 h) (p2 w)',p1=p1, p2=p2)
    crops = crops.numpy()
    
    start_h, start_w = (crops.shape[0]-dims[0])//2, (crops.shape[1]-dims[1])//2
    end_h, end_w = start_h+dims[0], start_w+dims[1]
    model_kwargs['result'] = crops[start_h:end_h, start_w:end_w]

    model_kwargs['pred_count'] = get_circle_count(crops.astype(np.uint8))
    
    return model_kwargs


def save_visuals(model_kwargs, args):

    crowd_img = model_kwargs["low_res"]
    crowd_img = ((crowd_img + 1) * 127.5).clamp(0, 255).to(th.uint8)
    crowd_img = crowd_img.permute(0, 2, 3, 1)
    crowd_img = crowd_img.contiguous().cpu().numpy()[0]

    crowd_den = model_kwargs['crowd_den']
    crowd_den = (crowd_den + 1) * args.normalizer/2
    crowd_den = crowd_den*255.0/(th.max(crowd_den)+1e-8)
    crowd_den = crowd_den.clamp(0, 255).to(th.uint8)
    crowd_den = crowd_den.permute(0, 2, 3, 1)
    crowd_den = crowd_den.contiguous().cpu().numpy()[0]

    sample = model_kwargs['result'][:,:,np.newaxis]

    gap = 5
    red_gap = np.zeros((crowd_img.shape[0],gap,3), dtype=int)
    red_gap[:,:,0] = np.ones((crowd_img.shape[0],gap), dtype=int)*255

    if args.pred_channels == 1:
        sample = np.repeat(sample, 3, axis=-1)
        crowd_den = np.repeat(crowd_den, 3, axis=-1)

    req_image = np.concatenate([crowd_img, red_gap, sample, red_gap, crowd_den], axis=1)
    print(model_kwargs['name'])
    path = f'{model_kwargs["name"][0].split(".")[0].split("-")[0]} {model_kwargs["pred_count"] :.0f} {model_kwargs["gt_count"] :.0f}.jpg'
    cv2.imwrite(os.path.join(args.log_dir, path), req_image[:,:,::-1])


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        per_samples=1,
        use_ddim=True,
        data_dir="", # data directory
        model_path="", # model path
        log_dir=None, # output directory
        normalizer=0.2, # density normalizer
        pred_channels=3,
        thresh=200, # threshold for circle count
        file='', # specific file number to test
        overlap=0.5, # overlapping ratio for image crops
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def load_data_for_worker(base_samples, batch_size, normalizer, pred_channels, file_name, class_cond=False):
    if file_name == '':
        img_list = sorted(glob.glob(os.path.join(base_samples,'*.jpg')))
    else:
        img_list = sorted(glob.glob(os.path.join(base_samples,f'*/*/{file_name}-*.jpg')))
    den_list = []
    for _ in img_list:
        den_path =  _.replace('test','test_den')
        den_path = den_path.replace('.jpg','.csv')
        den_list.append(den_path)

    image_arr, den_arr = [], []
    for file in img_list:
        image = Image.open(file)
        image_arr.append(np.asarray(image))

        file = file.replace('test','test_den').replace('jpg','csv')
        image = np.asarray(pd.read_csv(file, header=None).values)
        image = image/normalizer
        image = np.repeat(image[:,:,np.newaxis],pred_channels,-1)
        den_arr.append(image)

    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer, den_buffer = [], []
    label_buffer = []
    name_buffer = []
    while True:
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i]), den_buffer.append(den_arr[i])
            name_buffer.append(os.path.basename(img_list[i]))
            if class_cond:
                # label_buffer.append(label_arr[i])
                pass
            if len(buffer) == batch_size:
                batch = th.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                den_batch = th.from_numpy(np.stack(den_buffer)).float()
                den_batch = den_batch
                den_batch = 2*den_batch - 1
                den_batch = den_batch.permute(0, 3, 1, 2)
                res = dict(low_res=batch,
                           name=name_buffer,
                           high_res=den_batch
                           )
                if class_cond:
                    res["y"] = th.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer, name_buffer, den_buffer = [], [], [], []


if __name__ == "__main__":
    main()
