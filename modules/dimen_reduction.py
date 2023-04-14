import modules.unsupervised as un
from modules.sklearn import load_nmist_dataset
import matplotlib.pyplot as plt
import cv2
from cv2 import imread
import io 
from modules.imagen import plot_all 
#Reduct with sklearn
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#My implementation

def reduction_SVD():
    X_train, y_train, X_test, y_test = load_nmist_dataset()
    svd = un.SVD(n_components=2)
    X_train_transformed = svd.transform(X_train)
    X_test_transformed = svd.transform(X_test)
    return X_train_transformed,y_train,X_test_transformed,y_test

def reduction_PCA(n_components=2):
    X_train, y_train, X_test, y_test = load_nmist_dataset()
    pca = un.PCA(n_components=n_components)
    pca.fit(X_train)
    X_train_transformed = pca.transform(X_train)
    pca.fit(X_test)
    X_test_transformed = pca.transform(X_test)
    return X_train_transformed,y_train,X_test_transformed,y_test

def reduction_TSNE(limit=200,flag=True):
    print("flag2:",flag)
    X_train, y_train, X_test, y_test = load_nmist_dataset()
    X_train = X_train[:limit]
    y_train = y_train[:limit]
    X_test = X_test[:limit]
    y_test = y_test[:limit]
    tsne = un.tsne(flag)
    X_train_transformed = tsne.fit_tsne(X_train,y_train)
    X_test_transformed = tsne.fit_tsne(X_test,y_test)
    return X_train_transformed,y_train,X_test_transformed,y_test

def plot_reduction_SVD():
    values_reduction_SVD_train,y_train,values_reduction_SVD_test,y_test = reduction_SVD()
    train_color = ['y' if i == '0' else 'tab:purple' for i in y_train]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig = plt.scatter(values_reduction_SVD_train[:,0], values_reduction_SVD_train[:,1], c=train_color)
    plt.savefig(".\plots\plot_reduct_train_svd.jpeg")
    #Test
    train_color = ['y' if i == '0' else 'tab:purple' for i in y_test]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig = plt.scatter(values_reduction_SVD_test[:,0], values_reduction_SVD_test[:,1], c=train_color)
    plt.savefig(".\plots\plot_reduct_test_svd.jpeg")

def plot_reduction_PCA():
    values_reduction_PCA_train,y_train,values_reduction_PCA_test,y_test = reduction_PCA()
    train_color = ['y' if i == '0' else 'tab:purple' for i in y_train]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig = plt.scatter(values_reduction_PCA_train[:,0], values_reduction_PCA_train[:,1], c=train_color)
    plt.savefig(".\plots\plot_reduct_train_pca.jpeg")
    #Test
    train_color = ['y' if i == '0' else 'tab:purple' for i in y_test]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig = plt.scatter(values_reduction_PCA_test[:,0], values_reduction_PCA_test[:,1], c=train_color)
    plt.savefig(".\plots\plot_reduct_test_pca.jpeg")

def plot_reduction_TSNE(flag):
    print("flag1:",flag)
    values_reduction_TSNE_train,y_train,values_reduction_TSNE_test,y_test = reduction_TSNE(700,flag)
    train_color = ['y' if i == '0' else 'tab:purple' for i in y_train]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig = plt.scatter(values_reduction_TSNE_train[:,0], values_reduction_TSNE_train[:,1], c=train_color)
    plt.savefig(".\plots\plot_reduct_train_tsne.jpeg")
    #Test
    train_color = ['y' if i == '0' else 'tab:purple' for i in y_test]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig = plt.scatter(values_reduction_TSNE_test[:,0], values_reduction_TSNE_test[:,1], c=train_color)
    plt.savefig(".\plots\plot_reduct_test_tsne.jpeg")

#sklearn implementation
def reduction_SVD_sk():
    X_train, y_train, X_test, y_test = load_nmist_dataset()
    svd = SVD(n_components=2)
    svd.fit(X_train)
    X_train_transformed = svd.transform(X_train)
    svd.fit(X_test)
    X_test_transformed = svd.transform(X_test)
    return X_train_transformed,y_train,X_test_transformed,y_test

def reduction_PCA_sk():
    X_train, y_train, X_test, y_test = load_nmist_dataset()
    pca = PCA(n_components=2)
    pca.fit(X_train)
    X_train_transformed = pca.transform(X_train)
    pca.fit(X_test)
    X_test_transformed = pca.transform(X_test)
    return X_train_transformed,y_train,X_test_transformed,y_test

def reduction_TSNE_sk(limit=200):
    X_train, y_train, X_test, y_test = load_nmist_dataset()
    #X_train = X_train[:limit]
    #y_train = y_train[:limit]
    #X_test = X_test[:limit]
    #y_test = y_test[:limit]
    tsne = TSNE()
    X_train_transformed = tsne.fit_transform(X_train,y_train)
    X_test_transformed = tsne.fit_transform(X_test,y_test)
    return X_train_transformed,y_train,X_test_transformed,y_test

def plot_reduction_SVD_sk():
    values_reduction_SVD_train,y_train,values_reduction_SVD_test,y_test = reduction_SVD_sk()
    train_color = ['y' if i == '0' else 'tab:purple' for i in y_train]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig = plt.scatter(values_reduction_SVD_train[:,0], values_reduction_SVD_train[:,1], c=train_color)
    plt.savefig(".\plots\plot_reduct_train_svd_sk.jpeg")
    #Test
    train_color = ['y' if i == '0' else 'tab:purple' for i in y_test]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig = plt.scatter(values_reduction_SVD_test[:,0], values_reduction_SVD_test[:,1], c=train_color)
    plt.savefig(".\plots\plot_reduct_test_svd_sk.jpeg")

def plot_reduction_PCA_sk():
    values_reduction_PCA_train,y_train,values_reduction_PCA_test,y_test = reduction_PCA_sk()
    train_color = ['y' if i == '0' else 'tab:purple' for i in y_train]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig = plt.scatter(values_reduction_PCA_train[:,0], values_reduction_PCA_train[:,1], c=train_color)
    plt.savefig(".\plots\plot_reduct_train_pca_sk.jpeg")
    #Test
    train_color = ['y' if i == '0' else 'tab:purple' for i in y_test]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig = plt.scatter(values_reduction_PCA_test[:,0], values_reduction_PCA_test[:,1], c=train_color)
    plt.savefig(".\plots\plot_reduct_test_pca_sk.jpeg")

def plot_reduction_TSNE_sk():
    values_reduction_TSNE_train,y_train,values_reduction_TSNE_test,y_test = reduction_TSNE_sk(700)
    train_color = ['y' if i == '0' else 'tab:purple' for i in y_train]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig = plt.scatter(values_reduction_TSNE_train[:,0], values_reduction_TSNE_train[:,1], c=train_color)
    plt.savefig(".\plots\plot_reduct_train_tsne_sk.jpeg")
    #Test
    train_color = ['y' if i == '0' else 'tab:purple' for i in y_test]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig = plt.scatter(values_reduction_TSNE_test[:,0], values_reduction_TSNE_test[:,1], c=train_color)
    plt.savefig(".\plots\plot_reduct_test_tsne_sk.jpeg")