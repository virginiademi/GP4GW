# Create new conda environment and install kernel
```python
conda create -n gp4gw python=3.7 ipykernel
conda activate gp4gw
python -m ipykernel install --name gp4gw --user
```
# Install gp4gw package
```bash
git clone https://github.com/virginiademi/GP4GW.git
pip install -r requirements.txt
python setup.py install
```
