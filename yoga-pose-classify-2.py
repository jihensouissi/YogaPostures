#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.utils import shuffle           
from tqdm import tqdm


# In[86]:


from PIL import Image
from PIL import ImageFile
from tensorflow.keras.preprocessing.image import ImageDataGenerator


ImageFile.LOAD_TRUNCATED_IMAGES = True
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# # 1/ Data Requirements & Data Collection
# 

# ##  the data was collected from a forwarded folder "YOGA" containing 2 sets one for the training which contains 5 yoga positions supported with images for each position and second folder for the testing respectively

# In[ ]:





# In[ ]:





# # 2/ Data Understanding & Data Preparation
# 

# In[89]:


traindir = 'C:/Users/AHMED/Desktop/YOGA/content/cleaned/DATASET/TRAIN'
validdir = 'C:/Users/AHMED/Desktop/YOGA/content/cleaned/DATASET/TEST'


# In[91]:


print('Number of positions to be predicted in the training set : ',len(os.listdir(traindir)))
print('Number of positions to be predicted in the testing set : ',len(os.listdir(validdir)))


# In[92]:


class_dir = 'C:/Users/AHMED/Desktop/YOGA/content/cleaned/DATASET/TRAIN/plank'


# In[ ]:





# # Visualiztion of the content of the training and testing folders  

# In[ ]:





# In[99]:


class_names = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

IMAGE_SIZE = (150,150)
class_names


# # FROM THE TRAINING SET 

# In[110]:


for pose in class_names:
    image_dir = f'{traindir}/{pose}'
    images = os.listdir(image_dir)
 
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Images for {pose} category . . . .', fontsize=20)
 
    for i in range(3):
        k = np.random.randint(0, len(images))
        img = np.array(Image.open(f'{traindir}/{pose}/{images[k]}'))
        ax[i].imshow(img)
        ax[i].axis('off')
    plt.show()


# # FROM THE TESTING SET 

# In[111]:


for pose in class_names:
    image_dir = f'{validdir}/{pose}'
    images = os.listdir(image_dir)
 
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Images for {pose} category . . . .', fontsize=20)
 
    for i in range(3):
        k = np.random.randint(0, len(images))
        img = np.array(Image.open(f'{validdir}/{pose}/{images[k]}'))
        ax[i].imshow(img)
        ax[i].axis('off')
    plt.show()


# In[ ]:





# In[105]:


fig, axes = plt.subplots(nrows=3, ncols=3,figsize=(10,5), subplot_kw={'xticks':[], 'yticks':[]})
for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(os.path.join(class_dir,os.listdir(class_dir)[i])))
    ax.set_title(os.listdir(traindir)[2])
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# ## This function goes through all files in the yoga folder, and transfer all the files which are not a "JPG" type to JPG files

# In[67]:


def png_to_jpg(basedir):
    for foldername in os.listdir(basedir):
        folder_path = (basedir+'/' + foldername)
        print("here"+folder_path)
        for filename in os.listdir(folder_path):

            extension = os.path.splitext(filename)[1]
            if extension != ".jpg":
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path)
                if not img.mode == 'RGB':
                    img = img.convert('RGB')

                img.save(os.path.splitext(img_path)[0] + ".jpg")
                os.remove(img_path)
                img.close()
    print("All PNG Files Converted to JPG")


# ##  This function goes through all files in the yoga folder, checks if each file is a valid image, and removes corrupted files.
#  

# In[77]:



def removeCorruptedImages(path):
    for filename in os.listdir(path):
        try:
            img = Image.open(os.path.join(path,filename))
            img.verify() 
        except (IOError, SyntaxError) as e:
            print('Bad file:', filename)
            os.remove(os.path.join(path,filename))


# In[75]:


png_to_jpg(base_dir)
png_to_jpg(base_dirtested)


# In[80]:


#removeCorruptedImages(os.path.join(traindir,'TREE'))
removeCorruptedImages(os.path.join(traindir,'DOWNDOG'))
removeCorruptedImages(os.path.join(traindir,'WARRIOR2'))
removeCorruptedImages(os.path.join(traindir,'GODDESS'))
removeCorruptedImages(os.path.join(traindir,'PLANK'))


removeCorruptedImages(os.path.join(validdir,'TREE'))
removeCorruptedImages(os.path.join(validdir,'DOWNDOG'))
removeCorruptedImages(os.path.join(validdir,'WARRIOR2'))
removeCorruptedImages(os.path.join(validdir,'GODDESS'))
removeCorruptedImages(os.path.join(validdir,'PLANK'))


# In[ ]:





# In[87]:


train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)


# In[88]:


batch_size=8
print("For Training: ")
train_datagen = train_datagen.flow_from_directory(
                  directory = traindir,
                  target_size=(300,300),
                  batch_size=batch_size,
                  shuffle=True,
                  color_mode="rgb",
                  class_mode='categorical')

print("\nFor Testing: ")
val_datagen = test_datagen.flow_from_directory(
                directory = validdir,
                target_size=(300,300),
                batch_size=batch_size,
                shuffle=False,
                color_mode="rgb",
                class_mode='categorical')


# ### we can see that for the training set, the number of files is reduces from 1075 to 1064 and for the testing from 466 files to 458 files 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[70]:


def load_data():
   
    datasets = ["C:/Users/AHMED/Desktop/YOGA/content/cleaned/DATASET/TRAIN" , "C:/Users/AHMED/Desktop/YOGA/content/cleaned/DATASET/TEST"]#a list of directories that contain the image data to be loaded.
    output = []   ##  A list to hold the loaded image data
    
    # Iterate through the training and testing set.
    for dataset in datasets:
        
        images = [] 
        labels = []
        
        print("Loading {}".format(dataset))
        
        # Iterate through each Subfolder corresponding to a category  
        for folder in os.listdir(dataset):
         
        ###For each directory, the function loads all images from the subfolders representing each category of images
            label = class_names_label[folder]
            ## function "class_names_label" used to map each category name to a numerical label
            
            
            ##The function then loops through each subfolder and its contents using the os library
            # Iterate through each image in our folder
            
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
              
                # Image path should be obtained
                img_path = os.path.join(os.path.join(dataset, folder), file)
                
     ##For each image, we read the image using OpenCV, resizes the image, and appends it to a list named "images"
                image = cv.imread(img_path)
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                image = cv.resize(image, IMAGE_SIZE) 
                
                
    ##The corresponding label for the image is obtained from the dictionary, and it is appended to a list named "labels"
                # Append the image along with its label to the output
                images.append(image)
                labels.append(label)
                
        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')
        
    ## "images" and "labels" lists are converted to NumPy arrays and shuffled using the "shuffle" function.
        images, labels = shuffle(images, labels)
        
    ##The shuffled "images" and "labels" arrays are appended as a tuple to the "output" list.
        output.append((images, labels))

    return output
# output is a list of tuples, where each tuple contains two NumPy arrays: one array containing the image data, and the other containing the corresponding labels


# ###   OpenCV (cv) pour charger et redimensionner les images
# ###   la bibliothèque numpy pour stocker les images et les étiquettes dans des tableaux
# ###   la fonction shuffle de la bibliothèque scikit-learn pour mélanger les données.

# In[ ]:





# In[72]:


TRAIN_SPLIT = 0.7  # set the value of TRAIN_SPLIT to 0.7 (70% train, 30% test)

# code that uses TRAIN_SPLIT


# ###### train_images, train_labels, test_images, and test_labels will contain the image and label data for the training and test sets, respectively

# In[106]:


(train_images, train_labels), (test_images, test_labels) = load_data()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # A visualization of the distribution of images among different classes in the training and testing sets

# In[107]:



#  train_counts and test_counts are assigned to the corresponding counts of images for each class in the train and test sets
#  The "_" variable is used to discard the unique values
_, train_counts = np.unique(train_labels, return_counts=True)
_, test_counts = np.unique(test_labels, return_counts=True)
#  he NumPy unique function is used to find the unique values and their corresponding counts in the training and test sets of labels

pd.DataFrame({'train': train_counts,'test': test_counts}, index=class_names).plot.bar()
##   The plot.bar() is called to create a bar chart showing the distribution of images
##   among different classes in the training and test sets. 

plt.show()


# ## Each bar represents a class, and the height of the bar represents the number of images belonging to that class. The chart has two groups of bars, one for the training set and one for the test set.

# # 3/ Data Modeling & Model Evaluation
# 

# In[62]:


train_images = train_images / 255.0
test_images = test_images / 255.0


# In[63]:


### le modèle de réseau de neurones convolutif (ConvNet) pour la classification d'images à partir de données 
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('sigmoid'))


# In[64]:


model.summary()


# In[65]:


model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])


# In[66]:


history = model.fit(train_images, train_labels, batch_size = 28, epochs=15, validation_split = 0.3)


# # To visualize how our model is performing, we plot the accuracy and the loss during the training

# In[30]:


def plot_performance(history):

    fig = plt.figure(figsize=(15,8))

    # Plot accuracy
    plt.subplot(221)
    plt.plot(history.history['accuracy'],'bo--', label = "acc")
    plt.plot(history.history['val_accuracy'], 'ro--', label = "val_acc")
    plt.title("Training_accuracy vs Validation_accuracy")
    plt.ylabel("ACCURACY")
    plt.xlabel("epochs")
    plt.legend()

    # Plot loss_function
    plt.subplot(222)
    plt.plot(history.history['loss'],'bo--', label = "loss")
    plt.plot(history.history['val_loss'], 'ro--', label = "val_loss")
    plt.title("Training_loss vs Validation_loss")
    plt.ylabel("LOSS")
    plt.xlabel("epochs")

    plt.legend()
    plt.show()


# In[31]:


plot_performance(history)


# # The plot shows that The model is not perfect as we can clearly observe the Validation loss slightly flattening though the training loss is going down and the accuracy of the model is improving.
# # Reason could be insufficient training data and hence it could be improved by adding more training data.

# In[32]:


test_loss = model.evaluate(test_images, test_labels)


# # We now make predictions on the test data set.

# In[33]:


predictions = model.predict(test_images)
pred_labels = np.argmax(predictions,axis=1)  ...
# np.argmax is used since each prediction would be an array of probabilities and we need to pick the max value
    
     
pred_labels


# # Plotting the images along with their actual class and predicted class would give us a proper idea about how our model is making predictions.

# In[50]:


fig, ax = plt.subplots(6,5, figsize = (15,15))
ax = ax.ravel()

for i in range(0,30):  
    ax[i].imshow(test_images[i])
    ax[i].set_title(f"predicted class: {class_names[pred_labels[i]]} \n Actual Class: {class_names[test_labels[i]]}")
    ax[i].axis('off')
plt.subplots_adjust(wspace=0.65)


# # We took this sample of images to show the result of modeling applied on our dataset

# 
