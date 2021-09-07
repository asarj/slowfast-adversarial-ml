# slowfast-adversarial-ml

This repo contains all work relating to current research on Facebook's SlowFast models, specifically the X3D models and evaluating them on adversarial strength. We use the [Kinetics-400](https://deepmind.com/research/open-source/kinetics) dataset to evaluate an X3D model against the benchmarks published in the [original paper](https://arxiv.org/pdf/2004.04730.pdf), and then we aim to attack this model using tools from the [ART toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/examples/get_started_pytorch.py) to assess its adversarial strength

# Table of Contents
- [Setup](#Setup)
- [Execution](#Execution)

## Setup
Before doing anything, run the following
```bash
sudo apt-get update -y && sudo apt-get install vim xclip ffmpeg -y
alias pbcopy='xclip -selection clipboard'
alias pbpaste='xclip -selection clipboard -o'
python3 -m pip install virtualenv
```
Then, clone this repo:
```bash
git clone git@github.com:asarj/slowfast-adversarial-ml.git
cd slowfast-adversarial-ml
```

Note that all work has been done in a Zeblok AI-Workstation outfitted with 2 GPUs, 4 vCPUS, and 100 GB of memory, results may vary depending on your setup

### Obtaining the Kinetics-400 dataset
1. Open a terminal and make a directory to store the kinetics dataset
```bash
mkdir kinetics-400-dataset-files
cd kinetics-400-dataset-files
```
2. Download the dataset from. [here](https://deepmind.com/research/open-source/kinetics) or use the following command to download it
```bash
wget "https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz"
```
3. Extract the files
```bash
tar -xvf "kinetics400.tar.gz"
```
4. Since we are only interested in evaluating models, we will download from the `test.csv` file. Note that we will not download all files, as it is 450 GB total (and Zeblok only gives us 100 GB to work with). To do this, follow the corresponding commands
```bash
cd ../
git clone https://github.com/activitynet/ActivityNet.git
cd ActivityNet/Crawler/Kinetics
virtualenv venv
source venv/bin/activate
python3 -m pip install joblib mkl menpo numpy pandas pytz readline setuptools six tk wheel decorator olefile youtube-dl
mkdir ../../../kinetics-400-dataset-files/test
python3 download.py ../../../kinetics-400-dataset-files/kinetics400/test.csv ../../../kinetics-400-dataset-files/test/
```
We can use the following command in a new terminal to monitor how many videos are downloaded to the `test` directory. We only wish to download 1500-2200 videos, so we would stop the `download.py` script (ctrl + c) when the number of files in the `test` directory reaches this point
```bash
cd slowfast-adversarial-ml 
watch "find ./kinetics-400-dataset-files/test/ -type f | wc -l"
```
In our evaluation, we downloaded 2500 videos.

Until the desired video count is reached, we can skip ahead to setting up Facebook SlowFast in the meanwhile

### Setting Up Facebook SlowFast
To install SlowFast, follow the instructions (most of which are taken from [here](https://github.com/facebookresearch/SlowFast/blob/master/INSTALL.md))
1. Install dependencies
```bash
git clone https://github.com/facebookresearch/slowfast
cd slowfast/
virtualenv slowfast_venv
source slowfast_venv/bin/activate
pip install -U torch torchvision cython
pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo
export PYTHONPATH=path/to/slowfast-adversarial-ml/slowfast:$PYTHONPATH
```

2. Build SlowFast
```bash
python3 setup.py build develop
```

## Execution
Our research involves working primarily with SlowFast's X3D models. The X3D config files can be found [here](https://github.com/facebookresearch/SlowFast/tree/master/configs/Kinetics) and the X3D model defintion can be found [here](https://github.com/facebookresearch/SlowFast/blob/e2894034797b9d77625e36f39150380d4d26c878/slowfast/models/video_model_builder.py#L617)

To evaluate this code on the test set, pick an X3D model to use and run the following script in the terminal from the root of the `slowfast-adversarial-ml/slowfast` directory
```bash
python3 tools/run_net.py \ 
        --cfg configs/Kinetics/X3D_{model_size}.yaml \ 
        DATA.PATH_TO_DATA_DIR "../kinetics-400-dataset-files/test/" \ 
        DATA.PATH_LABEL_SEPARATOR " " \ 
        DATA.DECODING_BACKEND "pyav" \ 
        TRAIN.ENABLE False \ 
        DATA_LOADER.NUM_WORKERS 2 \ 
        NUM_GPUS 2 \ 
        TEST.BATCH_SIZE 8 \ 
        |& tee "/home/jovyan/asarjoo/slowfast-adversarial-ml/output_logs/[filename].txt"
```

In our case, we used the `X3D_M` model definition, so our script was
```bash
python3 tools/run_net.py \ 
        --cfg configs/Kinetics/X3D_M.yaml \ 
        DATA.PATH_TO_DATA_DIR "../kinetics-400-dataset-files/test/" \ 
        DATA.PATH_LABEL_SEPARATOR " " \ 
        DATA.DECODING_BACKEND "pyav" \ 
        TRAIN.ENABLE False \ 
        DATA_LOADER.NUM_WORKERS 2 \ 
        NUM_GPUS 2 \ 
        TEST.BATCH_SIZE 8 \ 
        |& tee "/home/jovyan/asarjoo/slowfast-adversarial-ml/output_logs/test_results_X3DM_trial1.txt"
```