from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import io
import os
from pathlib import Path
from modules.Google import Create_Service

def dowload_process():
    # Carpeta de destino donde se descargan los archivos
    destiny = ".\imagenes"

    # ID de la carpeta de Google Drive que se desea descargar
    carpet_id = "1f1aZ4i1lYsRaW9ID76iHfGztKdmAsg21"

    CLIENTE = ".\credentials\credentials.json"
    API_NAME = "drive"
    API_VERSION = "v3"
    SCOPES = ['https://www.googleapis.com/auth/drive'] 

    service = Create_Service(CLIENTE,API_NAME,API_VERSION,SCOPES)

    # Obtener lista de archivos de la carpeta especificada
    query = f"'{carpet_id}' in parents and trashed = false"
    results = service.files().list(q=query, fields="nextPageToken, files(id, name)").execute()
    items = results.get("files", [])

    # Descargar cada archivo
    for item in items:
        archivo_id = item["id"]
        archivo_nombre = item["name"]
        request = service.files().get_media(fileId=archivo_id)
        archivo_descargado = io.BytesIO()
        downloader = MediaIoBaseDownload(archivo_descargado, request)
        done = False

        while not done:
            status, done = downloader.next_chunk()
            print("Download progesss {0}".format(status.progress()*100))
        # Guardar el archivo en la carpeta de destino
        with open(os.path.join(destiny, archivo_nombre), 'wb') as f:
            f.write(archivo_descargado.getbuffer())
    return "Images dowload ok"