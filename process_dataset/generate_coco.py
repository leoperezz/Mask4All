from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
from os.path import join
import cv2
from PIL import Image
from tqdm import tqdm


def filterDataset(folder, classes=None, mode='train'):    
    # initialize COCO api for instance annotations
    annFile = '{}/annotations/instances_{}2017.json'.format(folder, mode)
    coco = COCO(annFile)
    
    images = []
    if classes!=None:
        # iterate for each individual class in the list
        for className in classes:
            # get all images containing given categories
            catIds = coco.getCatIds(catNms=className)
            imgIds = coco.getImgIds(catIds=catIds)
            images += coco.loadImgs(imgIds)
    
    else:
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)
    
    # Now, filter out the repeated images
    unique_images = []
    for i in tqdm(range(len(images))):
        if images[i] not in unique_images:
            unique_images.append(images[i])
            
    random.shuffle(unique_images)
    dataset_size = len(unique_images)
    
    return unique_images, dataset_size, coco


def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return None

def getImage(imageObj, img_folder, input_image_size):
    # Read and normalize an image
    train_img = io.imread(img_folder + '/' + imageObj['file_name'])
    # Resize
    train_img = cv2.resize(train_img, input_image_size)
    if (len(train_img.shape)==3 and train_img.shape[2]==3): # If it is a RGB 3 channel image
        return train_img
    else: # To handle a black and white image, increase dimensions to 3
        stacked_img = np.stack((train_img,)*3, axis=-1)
        return stacked_img
    
def getNormalMask(imageObj, classes, coco, catIds, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    cats = coco.loadCats(catIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        className = getClassName(anns[a]['category_id'], cats)
        pixel_value = classes.index(className)+1
        new_mask = cv2.resize(coco.annToMask(anns[a])*pixel_value, input_image_size)
        train_mask = np.maximum(new_mask, train_mask)

    return train_mask  
    
def getBinaryMask(imageObj, coco, catIds, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        new_mask = cv2.resize(coco.annToMask(anns[a]), input_image_size)
        
        #Threshold because resizing may cause extraneous values
        new_mask[new_mask >= 0.5] = 1
        new_mask[new_mask < 0.5] = 0

        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask


def save_array(image:np.ndarray,folder:str,name_img:str):
    
    img = Image.fromarray(image.astype(np.uint8))
    img.save(join(folder,name_img))



def dataGeneratorCoco(images, 
                      classes, 
                      coco, 
                      folder, 
                      input_image_size, 
                      mode, 
                      mask_type,
                      save_folder):
    
    img_folder = '{}/images/{}'.format(folder, mode)
    dataset_size = len(images)
    catIds = coco.getCatIds(catNms=classes)
    folder_img = join(save_folder,mode,"images")
    folder_mask = join(save_folder,mode,"masks")
    
    for i in tqdm(range(dataset_size)): #initially from 0 to batch_size, when c = 0
        
        imageObj = images[i]
            
        train_img = getImage(imageObj, img_folder, input_image_size)
        
        if mask_type=="binary":
            train_mask = getBinaryMask(imageObj, coco, catIds, input_image_size)
            
        elif mask_type=="normal":
            train_mask = getNormalMask(imageObj, classes, coco, catIds, input_image_size)      

        img_name = "img_{:07d}.jpg".format(i)
        mask_name = "img_{:07d}.png".format(i)
        
        save_array(train_img,folder_img,img_name)
        save_array(train_mask,folder_mask,mask_name)
        


if __name__ == '__main__':

    folder = '/home/lperez/code/Satellite/data/coco-dataset'
    classes = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    ]
    modes = ['val']
    input_image_size = (256,256)
    mask_type = 'normal'
    save_folder = "/home/lperez/code/Satellite/data/coco-dataset-process"
    for mode in modes:
        images, dataset_size, coco = filterDataset(folder, classes,  mode)
        print(f'dataset size: {dataset_size}')
        dataGeneratorCoco(images, classes, coco, folder,input_image_size, mode, mask_type,save_folder)

















