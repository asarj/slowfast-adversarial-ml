# slowfast-adversarial-ml

This repo contains all work relating to current research on Facebook's SlowFast models, specifically the X3D models and evaluating them on adversarial strength. We use the [Kinetics-400](https://deepmind.com/research/open-source/kinetics) dataset to evaluate an X3D model against the benchmarks published in the [original paper](https://arxiv.org/pdf/2004.04730.pdf), and then we aim to attack this model using tools from the [ART toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/examples/get_started_pytorch.py) to assess its adversarial strength

# Table of Contents
- [Setup](#Setup)
- [Execution](#Execution)

## Setup
Before doing anything, run the following
```bash
sudo apt-get update
sudo apt-get install vim xclip ffmpeg -y
alias pbcopy='xclip -selection clipboard'
alias pbpaste='xclip -selection clipboard -o'
python3 -m pip install virtualenv
```
Then, clone this repo:
```bash
git clone git@github.com:asarj/slowfast-adversarial-ml.git
cd slowfast-adversarial-ml
```
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
We can use the following command in a new terminal to monitor how many videos are downloaded to the `test` directory. We only wish to download 1500-2000 videos, so we would stop the `download.py` script (ctrl + c) when the number of files in the `test` directory reaches this point
```bash
cd slowfast-adversarial-ml 
watch "find ./kinetics-400-dataset-files/test/ -type f | wc -l"
```

Until the desired video count is reached, we can skip ahead to setting up Facebook SlowFast in the meanwhile

### Setting Up Facebook SlowFast