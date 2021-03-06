import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset_utils.image_dataset import ImageDataset
from dataset_utils.concat_img_bck import ConcatImgBck
import dataset_utils.augumentation as A
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image
from models.model import WholeNet

import hydra

@hydra.main(config_path="configs", config_name="test_image.yaml")
def main(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device : "+str(device))

    #prepare dataset
    image_rgb_data = ImageDataset(config["src_path"],"RGB") #source data
    image_bck_data = ImageDataset(config["bck_path"],"RGB")#,transforms=T.RandomRotation(degrees=(180,180))) #background data
    image_rgb_bck = ConcatImgBck(image_rgb_data, image_bck_data, transforms=A.PairCompose([A.PairApply(nn.Identity()), A.PairApply(T.ToTensor())]))

    test_dataset = DataLoader(image_rgb_bck, batch_size=1, pin_memory=True)

    #prepare model
    model = WholeNet().to(device)
    model.eval()

    #load pretrained model
    pretrained_state_dict = torch.load(config["pretrained_model"],map_location=device)
    
    matched , total = 0, 0
    original_state_dict = model.state_dict()
    for key in original_state_dict.keys():
        total +=1
        if key in pretrained_state_dict and original_state_dict[key].shape == pretrained_state_dict[key].shape:
            original_state_dict[key] = pretrained_state_dict[key]
            matched += 1
    model.load_state_dict(original_state_dict)
    print(f'Loaded pretrained state_dict: {matched}/{total} matched')

    #prepare output directory
    if not os.path.exists(config["output_path"]):
        os.makedirs(config["output_path"])
        print("created output dir : "+config["output_path"])
    print("========start inference=========")
    #Timer
    times = []
    #inference
    with torch.no_grad():
        for i ,(src, bck) in enumerate(tqdm(test_dataset)):
            src = src.to(device)
            bck = bck.to(device)

            filename = image_rgb_bck.img_dataset.filenames[i]
            filename = os.path.relpath(filename, config["src_path"])
            filename = os.path.splitext(filename)[0]
            
            #start timer
            tic = time.perf_counter()
            #inference
            alp, fgr, _, _, err, ref = model(src, bck)
            #stop timer
            toc = time.perf_counter()
            print(f'Matting time: {toc-tic} sec')
            times.append(toc-tic)
            
            if 'com' in config["output_type"]:
                com = torch.cat([fgr * alp.ne(0), alp], dim=1)
                filepath = os.path.join(config["output_path"], filename + '_com.png')
                com = to_pil_image(com[0])
                com.save(filepath)
                print("saved "+filepath)
            if 'alp' in config["output_type"]:
                filepath = os.path.join(config["output_path"], filename + '_alp.jpg')
                alp = to_pil_image(alp[0])
                alp.save(filepath)
                print("saved "+filepath)
            if 'fgr' in config["output_type"]:
                filepath = os.path.join(config["output_path"], filename + '_fgr.jpg')
                fgr = to_pil_image(fgr[0])
                fgr.save(filepath)
                print("saved "+filepath)
            if 'err' in config["output_type"]:
                err = F.interpolate(err, src.shape[2:], mode='bilinear', align_corners=False)
                filepath = os.path.join(config["output_path"], filename + '_err.jpg')
                err = to_pil_image(err[0])
                err.save(filepath)
                print("saved "+filepath)
            if 'ref' in config["output_type"]:
                ref = F.interpolate(ref, src.shape[2:], mode='nearest')
                filepath = os.path.join(config["output_path"], filename + '_ref.jpg')
                ref = to_pil_image(ref[0])
                ref.save(filepath)
                print("saved "+filepath)
    
    if len(times)>1:
        times.pop(0) #ignore first inference time because it is flawed for unknown reason
        avg_time = sum(times) / len(times)
        print(f"Average matting time: {avg_time}")

if __name__ == "__main__":
    main()