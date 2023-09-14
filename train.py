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
from src.functions import generate_mask,generate_images

BIG_LOSS = 1000
NO_WANDB = True


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--device','-d',type=int,default=1)
    parser.add_argument('--config','-c',type=str,default="configs/config_drone.json")

    args = parser.parse_args()

    return args

def train(model:Model,
          optim:Optimizer,
          train_loader,
          experiment,
          config):
    
    model.train()

    metrics=dict(dice=Dice(num_classes=config["model"]["classes"]),
                 iou=JaccardIndex(task='multiclass',num_classes=config["model"]["classes"]))

    total_loss = 0
    total_iou = 0
    total_dice = 0
    total_batches = len(train_loader)

    for inputs in tqdm(train_loader,total=total_batches):

        loss,dice,iou = functions.train_step(model,optim,inputs,metrics) 

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

def evaluate(model:Model,
             val_loader,
             experiment,
             config):
    
    model.eval()

    metrics=dict(dice=Dice(num_classes=config["model"]["classes"]),
                 iou=JaccardIndex(task='multiclass',num_classes=config["model"]["classes"]))

    total_loss = 0
    total_iou = 0
    total_dice = 0
    total_batches = len(val_loader)

    for inputs in tqdm(val_loader,total=total_batches):

        loss,dice,iou = functions.val_step(model,inputs,metrics)

        total_loss += loss
        total_dice += dice
        total_iou += iou


    origin_size=config["training"]["resize"]
    original_size = (origin_size,origin_size)
    image = generate_images(inputs)[0]
    mask = generate_mask(inputs)[0]
    pred = model.predict(image.unsqueeze(0),original_size)
    logg.log_wandb_val_images(mask,pred,experiment,config)

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
        
        output_train = train(model,optim,train_loader,experiment,config)
        output_val = evaluate(model,val_loader,experiment,config)

        logg.logging_info_epoch(output_train,output_val,experiment)    

        ckpt = functions.generate_checkpoint(model,optim)

        name_save_dir = f'output/{config["dataset"]["nombre"]}'

        if not os.path.exists(name_save_dir):
            os.makedirs(name_save_dir)

        if output_val["loss"] < best_loss:
            best_loss = output_val["loss"]
            torch.save(ckpt,os.path.join(name_save_dir,f"best_{config['model']['name']}.pt"))
        torch.save(ckpt,os.path.join(name_save_dir,f"last_{config['model']['name']}.pt"))
        



        









    
