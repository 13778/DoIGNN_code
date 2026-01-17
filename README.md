# <div align="center"> Domain-Informed Graph Neural Networks for Climate Factor Forecasting to Support Sustainable Crop Management </div>

## Requirements
The DoIGNN is built based on [BasicTS](https://github.com/zezhishao/BasicTS) with `torch==1.10.0+cu111` and `easy-torch==1.2.10`. `requirements.txt` shows other requred dependencies.
## Steps to train DoIGNN
1. Run `homozone_graph.ipynb` to build ACHZ-guided weighted adjacency.
2. Run `Decomposition/run_CMADS` to perform spatiotemporal decomposition to yield station-wise global representation.
3. Run `doignn/run.py` by certain config file to perform training. After obtaining trained autoencoder, move the best checkpoints to `encoder_ckpt`
4. Run `doignn/run.py` by certain config file to perform graph generation and prediction.
