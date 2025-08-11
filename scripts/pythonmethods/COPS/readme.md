# Important
- Python 12
- Cuda >= 12.4
- better to use an env and intall the following

### PyTorch and PyTorch3D
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.6.0cu124
```
### Only for Cops, torch_geometric and dependencies
```
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```
