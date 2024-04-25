# RoboRouter

The CV side of this project contains our image reprojection, stitching, and costmap generation. See the ImageProcessing Notebook for the example of how this project works, and the SImulationAnalysis Notebook for our simulation displaying the capabilities of the CV.

Using Transbot Yahboom OS (https://drive.google.com/drive/folders/1AYY0aqfEIAUVye5KiGW9kl8dchKNQvg9):

### Install Jupyter Notebook:

sudo apt install jupyter-core

sudo pip install notebook

### Install Python3.8:

sudo apt install python3.8

wget https://bootstrap.pypa.io/get-pip.py

sudo python3.8 get-pip.py

### Add the new python kernel to jupyter notebook:

sudo pip3.8 install notebook

sudo pip3.8 install ipykernel

sudo python3.8 -m ipykernel install --name py3.8 --display-name "Python 3.8"

### Install Dependencies:
(If on the VM, may want to change to Bridged Adapter for network so it will download faster)

sudo pip3.8 install --ignore-installed ultralytics
