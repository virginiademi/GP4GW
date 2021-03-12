# Create new conda environment and install kernel
conda create -n gp4gw python=3.7 ipykernel
conda activate gp4gw
python -m ipykernel install --name gp4gw --user

# Install gp4gw package
pip install -r requirements.txt
python setup.py install
