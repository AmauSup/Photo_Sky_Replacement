#Clone directory:
git clone https://github.com/m-che/photo-sky-replacement.git

#Go to app directory and setup environement from terminal:
python -m venv myenv
source myenv/bin/activate

#Install requirements
pip install -r requirements.txt

#Change directory and download the pretrained sky matting model:
cd checkpoints_G_coord_resnet50
wget http://casti.freeboxos.fr:57748/share/4ttqGqXYQLLfjP2T/best_ckpt.pt

#Back to main directory:
cd ..

#Launch app
python3 demo_server.py

#Open demo at http://0.0.0.0:8001/
#Clone directory:
git clone https://github.com/m-che/photo-sky-replacement.git

#Go to app directory and setup environement from terminal:
python -m venv myenv
source myenv/bin/activate

#Install requirements
pip install -r requirements.txt

#Change directory and download the pretrained sky matting model:
cd checkpoints_G_coord_resnet50
wget http://casti.freeboxos.fr:57748/share/4ttqGqXYQLLfjP2T/best_ckpt.pt

#Back to main directory:
cd ..

#Launch app
python3 demo_server.py

#Open demo at http://0.0.0.0:8001/