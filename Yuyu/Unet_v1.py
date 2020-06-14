###########################################################################################
# Unet version 1
# Reference: https://github.com/zhixuhao/unet
# Date: 2020-06-14
# Author: Xiong Yuyu
###########################################################################################
#Load Packages
import os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import *
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import albumentations as A
import time

#Set Path
DATA_DIR = './'
x_train_dir = os.path.join(DATA_DIR, 'train_frames/train')
y_train_dir = os.path.join(DATA_DIR, 'train_masks/train')

x_valid_dir = os.path.join(DATA_DIR, 'val_frames/val')
y_valid_dir = os.path.join(DATA_DIR, 'val_masks/val')

#Preprocessing
# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['0', '1', '2', '3', '4', '5']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            with_shape_assert= False
    ):
        self.image_ids = os.listdir(images_dir)
        self.mask_ids = os.listdir(masks_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.image_ids] #store image paths to a list
        
        self.masks_fps =[]
        for i in self.image_ids:
            for j in self.mask_ids:
                if i[:-4] == j[0:16]:
                    self.masks_fps.append(os.path.join(masks_dir,j)) #store mask paths to a list
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes] #dont understand!
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.with_shape_assert = with_shape_assert
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0) #read mask in grayscale (5120,5120)
        
        # extract certain classes from mask (e.g. 0 or 1 or 2)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float') #(5120,5120,6)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        #this is to check the shape of image and ask after the agumentation and preprprocessing
        if self.with_shape_assert:
            assert image.shape ==(512, 512, 3)
            assert mask.shape ==(512,512,6)
            
        return image, mask
        
    def __len__(self):
        return len(self.image_ids)
    
    def allItemsToStr(self):
        result = "imgs are " + str(len(self.image_ids)) + " masks " + str(len(self.mask_ids))
        # for each img, result + img path/ name.../ size
        return result
    
    def __str__(self):
        return 'Dataset: len is: ' + str(len(self.image_ids)) + " item summary: " + self.allItemsToStr() 

class Dataloader(tf.keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integer number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
        https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/keras/utils/data_utils.py#L444-L454
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        # check batch shape
        assert len(batch)==2
        assert batch[0].shape ==(self.batch_size, 512, 512, 3)
        assert batch[1].shape ==(self.batch_size, 512, 512, 6)
        
        return batch[0], batch[1]
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
    
    def dataToStr(self,data):
        return "img= " + str(data[0].shape) + " mask= " + str(data[1].shape)
    
    def allDataSetToStr(self):
        result = ''
        for data in self.dataset:
            result + ' ## ' + self.dataToStr(data)
        
        return result
    
    def __str__(self):
        return 'Data loader ' + str(len(self.indexes)) + " batch size " + str(self.batch_size) + " data set " + str(len(self.dataset)) + self.allDataSetToStr()

#Augmentation
# define heavy augmentations
def get_training_augmentation():
    train_transform = [
        
        A.Resize(height=512, width=512, interpolation=cv2.INTER_NEAREST),

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        #A.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
        A.RandomCrop(height=512, width=512, always_apply=True),

        #A.IAAAdditiveGaussianNoise(p=0.2),
        #A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                #A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                #A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        # A.OneOf(
        #     [
        #         A.IAASharpen(p=1),
        #         A.Blur(blur_limit=3, p=1),
        #         A.MotionBlur(blur_limit=3, p=1),
        #     ],
        #     p=0.9,
        # ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.Resize(height=512, width=512, interpolation=cv2.INTER_NEAREST),
        #A.PadIfNeeded(512, 512)
    ]
    return A.Compose(test_transform)


def get_preprocessing():
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=normalize),
    ]
    return A.Compose(_transform)

def normalize(img, max_pixel_value=255.0, **args ):
    return img/255.0

#Define dice_coef
def dice_coef(y_true, y_pred,smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)


#Unet Model
#layers
def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(6, 1, activation = 'softmax')(conv9)

    model = Model(inputs, conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = [dice_coef,'accuracy',MeanIoU(num_classes=6)])
    #
    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

model = unet(input_size=(512,512,3))

BATCH_SIZE = 2
CLASSES = ['0', '1','2','3','4','5']
EPOCHS = 5
n_classes = len(CLASSES)

## Load Data
train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    classes=CLASSES, 
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(),
    with_shape_assert= True,
)

# Dataset for validation images
valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    classes=CLASSES, 
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(),
    with_shape_assert= True,
)

train_dataloader = Dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloader(valid_dataset, batch_size=1, shuffle=False)

model_checkpoint = ModelCheckpoint('unet_v1_checkpoint.hdf5', monitor='loss',verbose=1, save_best_only=True)
callback = EarlyStopping(monitor='loss',patience=3,verbose=1)

#Train
#Train Model
start_time = time.time()
unet1 = model.fit(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=EPOCHS, 
    validation_data=valid_dataloader, 
    callbacks=[model_checkpoint,callback], 
    validation_steps=len(valid_dataloader),verbose=1)

end_time = time.time()
print(f"training duration: {end_time - start_time}")
model.save("UNet_v1.h5", overwrite=True) 


