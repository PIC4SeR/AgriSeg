import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from utils.preprocess import unstandardize_image

def visualize_images(img, gt, pred, denormalize=False, vineyard=False, imagenet=False):
    class_colors = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), 
                             (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0), 
                             (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), 
                             (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), 
                             (0, 0, 230), (119, 11, 32), (0, 0, 0)]
    
    if pred is not None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 8))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(8, 8))
        
    if denormalize:
        img = unstandardize_image(img, imagenet=imagenet)
        
    if not vineyard:
        gt = tf.gather(class_colors, tf.cast(gt, tf.int32))

    axes[0].imshow(img)
    axes[0].set_title('Actual Image')

    axes[1].imshow(gt)
    axes[1].set_title('Masked Image')
    
    if pred is not None:
        axes[2].imshow(pred)
        axes[2].set_title('Predicted Image')
        

        
        
def visualize(X,y,index=0):
    img = np.array(X[index])
    mask = np.array(y[index])
    fig, axes = plt.subplots(1, 3, figsize=(12, 8))
    # show the mask over the image
    axes[0].imshow(img)
    axes[0].set_title('RGB Image')
    
    axes[1].imshow(mask)
    axes[1].set_title('Masked Image')
    
    axes[2].imshow(img)
    axes[2].imshow(mask, alpha=0.4)
    axes[2].set_title('RGB + Maks')
        

def plot_history(history, metric='accuracy'):
    acc = history.history[metric]
    val_acc = history.history['val_' + metric]
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1,len(acc)+1)

    plt.title('Training and validation accuracy')
    plt.plot(epochs, acc, color='blue', label='Train')
    plt.plot(epochs, val_acc, color='orange', label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    _ = plt.figure()
    plt.title('Training and validation loss')
    plt.plot(epochs, loss, color='blue', label='Train')
    plt.plot(epochs, val_loss, color='orange', label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
  