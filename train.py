import torch
from torch.optim import Optimizer
from torch import nn
from torchmetrics import Dice,JaccardIndex

import os
import argparse
from tqdm import tqdm
import logging

from src import functions,logg,load
from src.model import Model

BIG_LOSS = 1000

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--device','-d',type=int,default=1)
    parser.add_argument('--config','-c',type=str,default="configs/config_drone.json")

    args = parser.parse_args()

    return args

def train(model:Model,
          optim:Optimizer,
          train_loader,
          device:torch.device,
          experiment,
          config):
    
    model.train()

    metrics=dict(dice=Dice(num_classes=config["model"]["classes"]).to(device),
                 iou=JaccardIndex(task='multiclass',num_classes=config["model"]["classes"]).to(device))

    total_loss = 0
    total_iou = 0
    total_dice = 0
    total_batches = len(train_loader)

    for image,mask in tqdm(train_loader,total=total_batches):

        loss,dice,iou = functions.train_step(model,optim,image,mask,metrics,device) 

        experiment.log({
            "train loss step":loss,
            "train dice step":dice,
            "train iou step":iou
        })
        
        total_loss += loss
        total_dice += dice
        total_iou += iou


    output = dict(loss=total_loss/total_batches,
                  dice=total_dice/total_batches,
                  iou=total_iou/total_batches)

    return output

def evaluate(model:nn.Module,
             val_loader,
             device:torch.device,
             experiment,
             config):
    
    model.eval()

    metrics=dict(dice=Dice(num_classes=config["model"]["classes"]).to(device),
                 iou=JaccardIndex(task='multiclass',num_classes=config["model"]["classes"]).to(device))

    total_loss = 0
    total_iou = 0
    total_dice = 0
    total_batches = len(val_loader)

    for image,mask in tqdm(val_loader,total=total_batches):

        loss,dice,iou = functions.val_step(model,image,mask,metrics,device)

        total_loss += loss
        total_dice += dice
        total_iou += iou


    origin_size=config["training"]["resize"]
    pred = functions.predict_single_image(image[0],model,device,origin_size,config["dataset"])
    logg.log_wandb_val_images(mask[0],pred,experiment,config)

    output = dict(loss=total_loss/total_batches,
                  dice=total_dice/total_batches,
                  iou=total_iou/total_batches)

    return output

if __name__ == '__main__':

    args = get_args()

    config = logg.load_config(args.config)
    
    logg.config_logging()
    
    experiment = logg.config_wandb(config,"Mask4All")
    
    device = torch.device(f"cuda:{args.device}")
    
    train_loader,val_loader,test_loader = load.load_dataset(config)

    model = load.load_model(config,device=device)

    optim = load.load_optim(config,model=model)

    best_loss = BIG_LOSS

    for epoch in range(config["training"]["epochs"]):
        
        logging.info(f"Epoch {epoch+1}: ")
        
        output_train = train(model,optim,train_loader,device,experiment,config)
        output_val = evaluate(model,val_loader,device,experiment,config)

        logg.logging_info_epoch(output_train,output_val,experiment)    

        ckpt = functions.generate_checkpoint(model,optim)

        name_save_dir = f'output/{config["dataset"]["nombre"]}'

        if not os.path.exists(name_save_dir):
            os.makedirs(name_save_dir)

        if output_val["loss"] < best_loss:
            best_loss = output_val["loss"]
            torch.save(ckpt,os.path.join(name_save_dir,f"best_{config['model']['name']}.pt"))
        torch.save(ckpt,os.path.join(name_save_dir,f"last_{config['model']['name']}.pt"))
        



        









    
