#FastAPI
from typing import Union
from fastapi import FastAPI,UploadFile
from starlette.responses import StreamingResponse
#Download img
from modules.download_img import dowload_process
#Num
import numpy as np
from modules.matrix import matrix_a
from modules.imagen import my_photo_re,average_photos,dis_img,dis_img2,plot_pls,plot_all,plot_progressive,metric_differnt,plot_progressive_unitary, open_imagen
#Imagen
import cv2
from cv2 import imread
import io 
#unsupervised
import modules.unsupervised as un
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
#SVD for imagen
from scipy.linalg import svd
#Mnist
from modules.sklearn import apply_logistic_regression,apply_logistic_regression_any,load_nmist_dataset
from modules.dimen_reduction import plot_reduction_SVD,plot_reduction_PCA,plot_reduction_TSNE,reduction_SVD,reduction_PCA,reduction_TSNE
from modules.dimen_reduction import plot_reduction_SVD_sk,plot_reduction_PCA_sk,plot_reduction_TSNE_sk,reduction_SVD_sk,reduction_PCA_sk,reduction_TSNE_sk
#Time
import time



app = FastAPI()

#Download images
@app.get("/0_Download_images")
def download_imgages():
    msn = dowload_process()
    return {'Status:': msn}

#Rectangular matrix
@app.post("/1_Matrix")
def random_matrix(row = 3,col = 3):
    mat,rank,trace,det,invert,w_1, v_1, w_2, v_2 = matrix_a(row,col)
    return {'Matrix:': str(mat).replace("\n", ""), 
            'The rank is:' : str(rank).replace("\n", ""), 
            'The trace is:' : str(trace).replace("\n", ""), 
            'The determinant is:': str(det).replace("\n", ""),
            'Can you invert A?':'Yes, if determinant different from 0',
            'How?':'with the funtion np.linalg.inv from numpy',
            'Matrix invert is:' : str(invert).replace("\n", ""),
            'Eigenvalue 1': str(w_1).replace("\n", ""), 
            'Eigenvalue 2': str(w_2).replace("\n", ""), 
            'Eigenvector 1:' : str(v_1).replace("\n", ""), 
            'Eigenvector 2:' : str(v_2).replace("\n", ""),
            'How are eigenvalues and eigenvectors of A’A and AA’ related?': "have the same eigenvalues λ. A'Av = λv",
            'What interesting differences can you notice between A’A and AA’?': "about size A'A has the same number of columns as rows as matrix A has, AA' has the same number of rows as columns as A has",
            '.': "about rank The matrix A'A has the same rank as the original matrix A, while the rank of AA' can be smaller if A has more rows than columns.",
            '..':"The matrix A'A is symmetric and positive semidefinite, while AA' is symmetric but not necessarily positive semidefinite."}

#My photo
@app.get("/2_My_photo")
def my_photo_img():
    my_img = my_photo_re()
    # Returns a cv2 image array from the document vector
    res, im_png = cv2.imencode(".jpeg", my_img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

#Average of photos
@app.get("/2_Average_imagen")
def Average_imagen():
    my_img = average_photos()
    # Returns a cv2 image array from the document vector
    res, im_png = cv2.imencode(".jpeg", my_img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

#distant
@app.get("/2_distant")
def distant_img():
    distance = dis_img()
    mse_distance = dis_img2()
    return {"The Euclidean distance from the image to the mean is:": distance,
            "calculating the MSE": mse_distance}

#unsupervised Python package
@app.get("/3_unsupervised_first_example_SVD_PCA_TSNE")
def unsupervised_all():
    # Load DataSet
    wine_data = load_wine()
    X, y = wine_data['data'], wine_data['target']
    print(list(wine_data['target_names']))
    # initial figure
    fig, ax = plt.subplots(figsize=(7, 4))
    fig = plt.scatter(X[:,0], X[:,2], c=y)
    plt.legend(handles=fig.legend_elements()[0], 
            labels=list(wine_data['target_names']))
    plot_pls(".\plots\plot_wine.jpeg")
    # Normalise the data
    scaler = StandardScaler()
    scaler.fit(X)
    X_normalised = scaler.transform(X)
    # apply SVD
    # Normalise the data
    svd = un.SVD(n_components=2)
    # transform the data using the SVD object
    X_transformed = svd.transform(X_normalised)
    fig, ax = plt.subplots(figsize=(7, 4))
    fig = plt.scatter(X_transformed[:,0], X_transformed[:,1], c=y)
    plt.legend(handles=fig.legend_elements()[0], 
            labels=list(wine_data['target_names']))
    plot_pls(".\plots\plot_wine_svd.jpeg")
    # apply PCA
    pca = un.PCA(n_components=2)
    pca.fit(X_normalised)
    # transform the data using the PCA object
    X_transformed = pca.transform(X_normalised)
    fig, ax = plt.subplots(figsize=(7, 4))
    fig = plt.scatter(X_transformed[:,0], X_transformed[:,1], c=y)
    plt.legend(handles=fig.legend_elements()[0], 
            labels=list(wine_data['target_names']))
    plot_pls(".\plots\plot_wine_PCA.jpeg")
    # apply TSNE
    # Apply tsne now
    tsne = un.tsne()
    tsne.fit_tsne(X_normalised, y)
    # transform the data using the PCA object
    X_transformed = tsne.fit_tsne(X_normalised, y)
    fig, ax = plt.subplots(figsize=(7, 4))
    fig = plt.scatter(X_transformed[:,0], X_transformed[:,1], c=y)
    plt.legend(handles=fig.legend_elements()[0], 
            labels=list(wine_data['target_names']))
    plot_pls(".\plots\plot_wine_TSNE.jpeg")
    im_png = plot_all([".\plots\plot_wine.jpeg",".\plots\plot_wine_svd.jpeg",".\plots\plot_wine_PCA.jpeg",".\plots\plot_wine_TSNE.jpeg"], 
                      ["Date Origin", "Apply SVD", "Apply PCA", "Apply TSNE"], 
                      ".\plots\plot_wine_all.jpeg", 
                      row=2, col=2)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpeg")

#unsupervised Python package
@app.get("/4_Progressive_SVD")
def unsupervised_svd():
    # Load poto
    my_img = my_photo_re()
    # Returns a cv2 image array from the document vector
    img_denoised = plot_progressive(my_img)
    res, im_png = cv2.imencode(".jpeg", img_denoised)
    plt.savefig(".\plots\pro_svd_.jpeg")
    img=cv2.imread(".\plots\pro_svd_.jpeg")
    res, im_png = cv2.imencode(".jpeg", img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpeg")

#unsupervised Python package
@app.get("/4_Progressive_SVD_quantify")
def unsupervised_svd_quanti():
    my_img = my_photo_re()
    images_conse=plot_progressive_unitary(my_img)
    i=1
    j=0
    result={'similarity': '1 same imagen, 0 difirent imagen'}
    while i<256:
        i*=2
        img=images_conse[j]
        result['For imagen with '+str(i)+' components the similarity is']=metric_differnt(my_img,img)
        j+=1
    return result

#5 Mnist
@app.get("/5_Train_Mnist_LogisticRegression")
def train_msnist_LR():
    inicio = time.time()
    model,score= apply_logistic_regression()
    time_ejecution=time.time()-inicio
    return {"Score Train Mnist 0's & 8's with logist regression:":score,
            "Execution time in seconds:":time_ejecution} 

#6 dimensionality reduction
@app.get("/6_dimensionality_reduction_SVD_PCA_TSNE")
def dimensionality_reduction():
    inicio = time.time()
    plot_reduction_SVD()
    print("Tiempo SVD:",time.time()-inicio)
    inicio = time.time()
    plot_reduction_PCA()
    print("Tiempo PCA:",time.time()-inicio)
    inicio = time.time()
    plot_reduction_TSNE(flag=False)
    print("Tiempo TSNE:",time.time()-inicio)
    im_png = plot_all([".\plots\plot_reduct_train_svd.jpeg",".\plots\plot_reduct_test_svd.jpeg",".\plots\plot_reduct_train_pca.jpeg",".\plots\plot_reduct_test_pca.jpeg",".\plots\plot_reduct_train_tsne.jpeg",".\plots\plot_reduct_test_tsne.jpeg"], 
                      ["SVD train", "SVD test","PCA train","PCA test","TSNE train","TSNE test"], 
                      ".\plots\plot_reduc_dimen_all.jpeg", 
                      row=3, col=2,
                      fig_1=10, fig_2=8)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpeg")

#6 dimensionality reduction performance
@app.get("/6_dimensionality_reduction_SVD_PCA_TSNE_performance")
def dimensionality_reduction_performance():
    inicio = time.time()
    X_train_transformed,y_train,X_test_transformed,y_test= reduction_SVD()
    model,SVD_score = apply_logistic_regression_any(X_train_transformed,y_train,X_test_transformed,y_test)
    Tiempo_SVD_modeloLR=time.time()-inicio

    inicio = time.time()
    X_train_transformed,y_train,X_test_transformed,y_test= reduction_PCA()
    model,PCA_score = apply_logistic_regression_any(X_train_transformed,y_train,X_test_transformed,y_test)
    Tiempo_PCA_modeloLR=time.time()-inicio

    inicio = time.time()
    X_train_transformed,y_train,X_test_transformed,y_test= reduction_TSNE()
    model,TSNE_score = apply_logistic_regression_any(X_train_transformed,y_train,X_test_transformed,y_test)
    Tiempo_TSNE_modeloLR=time.time()-inicio
    
    return {"Score for reduction SVD:":SVD_score,
            "Execution time SVD in seconds:":Tiempo_SVD_modeloLR,
            "Score for reduction PCA:":PCA_score,
            "Execution time PCA in seconds:":Tiempo_PCA_modeloLR,
            "Score for reduction TSNE:":TSNE_score,
            "Execution time TSNE in seconds:":Tiempo_TSNE_modeloLR}

#6 dimensionality reduction
@app.get("/7_dimensionality_reduction_SVD_PCA_TSNE_Scikit_Learn")
def dimensionality_reduction_Scikit_Learn():
    plot_reduction_SVD_sk()
    plot_reduction_PCA_sk()
    plot_reduction_TSNE_sk()
    im_png = plot_all([".\plots\plot_reduct_train_svd_sk.jpeg",".\plots\plot_reduct_test_svd_sk.jpeg",".\plots\plot_reduct_train_pca_sk.jpeg",".\plots\plot_reduct_test_pca_sk.jpeg",".\plots\plot_reduct_train_tsne_sk.jpeg",".\plots\plot_reduct_test_tsne_sk.jpeg"], 
                      ["SVD train Scikit-Learn", "SVD test Scikit-Learn","PCA train Scikit-Learn","PCA test Scikit-Learn","TSNE train Scikit-Learn","TSNE test Scikit-Learn"], 
                      ".\plots\plot_reduc_dimen_all_sk.jpeg", 
                      row=3, col=2,
                      fig_1=10, fig_2=8)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpeg")

#6 dimensionality reduction performance
@app.get("/7_dimensionality_reduction_SVD_PCA_TSNE_Scikit_Learn_performance")
def dimensionality_reduction_performance_Scikit_Learn():
    inicio = time.time()
    X_train_transformed,y_train,X_test_transformed,y_test= reduction_SVD_sk()
    model, SVD_score = apply_logistic_regression_any(X_train_transformed,y_train,X_test_transformed,y_test)
    Tiempo_SVD_modeloLR=time.time()-inicio

    inicio = time.time()
    X_train_transformed,y_train,X_test_transformed,y_test= reduction_PCA_sk()
    model,PCA_score = apply_logistic_regression_any(X_train_transformed,y_train,X_test_transformed,y_test)
    Tiempo_PCA_modeloLR=time.time()-inicio

    inicio = time.time()
    X_train_transformed,y_train,X_test_transformed,y_test= reduction_TSNE_sk()
    model,TSNE_score = apply_logistic_regression_any(X_train_transformed,y_train,X_test_transformed,y_test)
    Tiempo_TSNE_modeloLR=time.time()-inicio
    
    return {"Score for reduction SVD Scikit-Learn:":SVD_score,
            "Execution time SVD Scikit-Learn in seconds:":Tiempo_SVD_modeloLR,
            "Score for reduction PCA Scikit-Learn:":PCA_score,
            "Execution time PCA Scikit-Learn in seconds:":Tiempo_PCA_modeloLR,
            "Score for reduction TSNE Scikit-Learn:":TSNE_score,
            "Execution time TSNE Scikit-Learn in seconds:":Tiempo_TSNE_modeloLR,}

#11 prediction 
@app.post("/11_prediction_0_8_unsupervised")
async def prediction_unsupervised(file: UploadFile):
    inicio = time.time()
    result={}
    img_array=open_imagen(file)
    inicio = time.time()
    X_train, y_train, X_test, y_test = load_nmist_dataset()
    X_train_transformed,y_train,X_test_transformed,y_test= reduction_SVD()
    model_SVD, SVD_score = apply_logistic_regression_any(X_train_transformed,y_train,X_test_transformed,y_test)
    result['Prediction with SVD unsupervised'] = model_SVD.predict(img_array)[0]
    print("Tiempo SVD prediction:",time.time()-inicio)

    inicio = time.time()
    X_train, y_train, X_test, y_test = load_nmist_dataset()
    X_train_transformed,y_train,X_test_transformed,y_test= reduction_PCA()
    model_PCA, PCA_score = apply_logistic_regression_any(X_train_transformed,y_train,X_test_transformed,y_test)
    result['Prediction with PCA unsupervised'] = model_PCA.predict(img_array)[0]
    print("Tiempo PCA prediction:",time.time()-inicio)

    inicio = time.time()
    X_train, y_train, X_test, y_test = load_nmist_dataset()
    X_train_transformed,y_train,X_test_transformed,y_test= reduction_TSNE()
    model_TSNE, TSNE_score = apply_logistic_regression_any(X_train_transformed,y_train,X_test_transformed,y_test)
    result['Prediction with TSNE unsupervised'] = model_TSNE.predict(img_array)[0]
    print("Tiempo TSNE prediction:",time.time()-inicio)

    print("Tiempo TSNE & modeloLR & scikit-learns:",time.time()-inicio)
    return result
    