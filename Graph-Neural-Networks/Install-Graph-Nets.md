# Install Graph Nets Library

[Graph Nets](https://github.com/deepmind/graph_nets) is DeepMind's library for building graph networks in Tensorflow and Sonnet. 

## Install Graph Nets Library via pip    
The Graph Nets library can be installed from [pip](https://github.com/deepmind/graph_nets/#Installation).  
To install the Graph Nets library for CPU, run:  

$ pip install graph_nets "tensorflow>=1.15,<2" tensorflow_probability  

To install the Graph Nets library for GPU, run:

$ pip install graph_nets "tensorflow_gpu>=1.15,<2" tensorflow_probability  

## Install Graph Nets Library via Conda  
First, make a conda environment, e.g. GN, as follows:  

$ conda create -n GN  

Then, activate your environment:  

$ conda activate GN  

Next, install the requirement packages using conda:  

$ conda install python=3.6 tensorflow=1.15 tensorflow-probability=0.8.0 jupyterlab matplotlib  

Next, we need to install  dm-sonnet and graph_nets  packages using pip:  

$ pip install dm-sonnet graph_nets   

Finally, assign a name to your kernel (e.g. GN):  

$ python -m ipykernel install --user --name GN --display-name "GN"    
