#Laboratory 1
#1
#librery
import io 
import os
#from Google import Create_Service
import cv2
from cv2 import imread
import numpy as np
import matplotlib.pyplot as plt
import io 
from skimage.metrics import structural_similarity as ssim
from PIL import Image

#How distant is your face from the average? How would you measure it?
#upload my photo, resize 256x256 and grayscale
def my_photo_re():
    try:
        my_photo =cv2.imread("./imagenes/SantiagoRG.jpeg",0)
        my_reshape = cv2.resize(my_photo, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        return my_reshape
    except:
        None

def average_photos(my_pic=my_photo_re()):
    try:
        my_pic=my_pic
        #Count pictures from drive
        initial_count = 0
        dir = r".\imagenes"
        for path in os.listdir(dir):
            if os.path.isfile(os.path.join(dir, path)):
                initial_count += 1
        #Read and convert imagen with sixe 256x256 and grayscale
        list_resize_gray = []
        for path in os.listdir(dir):
            try:
                img = cv2.imread(os.path.join(dir, path),0)
                res = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            except Exception as e:

                print(str(e))
            list_resize_gray.append(res)
        average_face =np.array(list_resize_gray)
        return np.mean(average_face, axis=0)
    except:
        return None

def dis_img(my_pic=my_photo_re(),img_average=average_photos()):
    distant_imagen = np.linalg.norm(my_pic.flatten() - img_average.flatten())
    return abs(distant_imagen)

def dis_img2(my_pic=my_photo_re(),img_average=average_photos()):
    mse = np.mean((my_pic - img_average) ** 2)
    return abs(mse)

def metric_differnt(my_pic=my_photo_re(),img_average=average_photos()):
    return ssim(my_pic, img_average, multichannel=True, data_range=my_pic.max() - my_pic.min())

def plot_pls(name):
    """
    Funtion for plot a sigle imagen
    name : path + name
    """
    plt.savefig(name)
    img=cv2.imread(name)
    res, im_png = cv2.imencode(".jpeg", img)
    return im_png

def plot_all(plots,title,name,row=1,col=1,fig_1=10,fig_2=8):
    """
    Funtion for plot diferentes plots in one
    plot: list of path's plots
    title : list of titles of plots
    name : str of name of the final plot
    row: rows of the final plot
    col: cols of the final plot
    fig_1: first element for size of plot,
    fig_2: second element for size of plot
    """
    axes=[]
    j=0
    fig=plt.figure(figsize=(fig_1,fig_2))
    for i in plots:
        img=cv2.imread(i)
        axes.append(fig.add_subplot(row, col, j+1))
        subplot_title=(title[j])
        axes[-1].set_title(subplot_title)  
        axes[-1].set_axis_off()
        plt.imshow(img)
        fig.tight_layout()
        j+=1
    plt.savefig(name)
    img=cv2.imread(name)
    res, im_png = cv2.imencode(".jpeg", img)
    return im_png

def plot_progressive(X):
    """
    Funtion for plot diferentes progessive components in personal photo
    X: Photo 256x256 grayscale
    """
    i=1
    rows = 4
    cols = 4
    axes=[]
    j=1
    fig=plt.figure(figsize=(12,10))
    s_future=[0] * 256
    while i<256:
        i*=2
        u,s,vh = np.linalg.svd(X)     #U:unitary matrix left singular vectors as columns, s:the singular values,Vh:singular vector like rows
        s=s[:i]
        for k in range(i):
            s_future[k] = s[k]
        #s_cleaned = np.array([si if si > 250 else 0 for si in s_future])
        s_cleaned = s_future
        img_denoised = np.array(np.dot(u * s_cleaned, vh), dtype=int)
        res, im_png = cv2.imencode(".jpeg", img_denoised)
        axes.append( fig.add_subplot(rows, cols, j) )
        subplot_title=("Imagen with "+str(i))
        axes[-1].set_title(subplot_title)  
        plt.imshow(img_denoised,cmap='gray')
        fig.tight_layout()    
        #plt.show()
        j+=1
    return img_denoised

def plot_progressive_unitary(X):
    """
    Funtion for plot diferentes progessive components in personal photo
    X: Photo 256x256 grayscale
    """
    i=1
    rows = 4
    cols = 4
    images_conse=[]
    j=1
    s_future=[0] * 256
    while i<256:
        i*=2
        u,s,vh = np.linalg.svd(X)     #U:unitary matrix left singular vectors as columns, s:the singular values,Vh:singular vector like rows
        s=s[:i]
        for k in range(i):
            s_future[k] = s[k]
        s_cleaned = s_future
        img_denoised = np.array(np.dot(u * s_cleaned, vh), dtype=int)
        images_conse.append(img_denoised)
        j+=1
    return images_conse

def open_imagen(file):
    with Image.open(file.file) as img:
        img_gray = img.convert('L') # convert to grayscale
        img_array = np.array(img_gray)
        img_array = [img_array.flatten()] # reshape to one-dimensional array
        return img_array