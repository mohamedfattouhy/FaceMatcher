# MANAGE ENVIRONNEMENT
import os
import requests
import tarfile
import shutil


def uncompress_and_move_lfw_dataset() -> None:

    # URL to load data from
    url = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'

    # Download file from URL
    response = requests.get(url)
    with open('lfw.tgz', 'wb') as file:
        file.write(response.content)

    # Uncompress Tar GZ Labelled Faces
    with tarfile.open('lfw.tgz', 'r:gz') as tar:
        tar.extractall()

    NEG_PATH = os.path.join('data', 'negative')

    # Move LFW Images to the following repository data/negative
    for directory in os.listdir('lfw'):
        for file in os.listdir(os.path.join('lfw', directory)):
            EX_PATH = os.path.join('lfw', directory, file)
            NEW_PATH = os.path.join(NEG_PATH, file)
            os.replace(EX_PATH, NEW_PATH)

    # Remove the 'lfw' directory
    shutil.rmtree('lfw')
