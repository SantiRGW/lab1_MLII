import os
direc_cre = ".\credentials"
direc_img = ".\imagenes"
direc_plt = ".\plots"
def create_folders():
    try:
        os.stat(direc_cre)
    except:
        os.mkdir(direc_cre)
    try:
        os.stat(direc_img)
    except:
        os.mkdir(direc_img)
    try:
        os.stat(direc_plt)
    except:
        os.mkdir(direc_plt)