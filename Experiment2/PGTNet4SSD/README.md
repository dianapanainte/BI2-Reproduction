# Supplementary Github repository for experiment 2 in our paper: On the Use of Steady-State Detection for Process Mining: Achieving More Accurate Insights
In this repository, we explain how we train three different models: Dummy, DALSTM, and PGTNet for remaining time prediction based on steady-state detection results. This repository is a forked version of [PGTNet repository](https://github.com/keyvan-amiri/PGTNet) with minor changes. For more details, you can refer to the original repository.

We applied all three models on 9 publicly available event logs. Each time we train the models first for all traces in the event log, and then for only traces that are included in steady-state periods. 

**<a name="part1">1. Clone repositories:</a>**
Clone the [GPS Graph Transformer repository](https://github.com/rampasek/GraphGPS) using the following command:
```
git clone https://github.com/rampasek/GraphGPS
```
This repository is called **GPS repository** in the remaining of this README file. Now, Navigate to the root directory of **GPS repository**, and clone the current repository (i.e., the **PGTNet repository**). By doing so, the **PGTNet repository** will be placed in the root directory of **GPS repository** meaning that the latter is the parent directory of the former.
```
cd GraphGPS
git clone https://github.com/Keyvan-Amiri-Elyasi/PGTNet4SSD
```

**<a name="part2">2. Set up a Python environement to work with GPS Graph Transformers:</a>**

To install and set up an environment on a Linux system, go to the root directory of **PGTNet repository** and run the following commands:

```bash
conda create -n SSD python=3.11
conda activate SSD
pip install -r requirements.txt
conda clean --all
```

**<a name="part3">3. Converting an event log into a graph dataset:</a>**

In order to convert an event log into its corresponding graph dataset, you need to copy **.xes** files into [raw_dataset](https://github.com/Keyvan-Amiri-Elyasi/PGTNet4SSD/tree/main/raw_dataset) directory and then run the following script:
```
python GTconvertor.py conversion_configs bpic15m1.yaml --overwrite true --ssd
```
The first argument (i.e., conversion_configs) is a name of directory in which all required configuration files are located. The second argument (i.e., bpic15m1.yaml) is the name of configuration file that defines parameters used for converting the event log into its corresponding graph dataset. All conversion configuration files used in our experiment are collected [here](https://github.com/Keyvan-Amiri-Elyasi/PGTNet4SSD/tree/main/conversion_configs). If the third argument (overwrite) is false, and you have already converted the event log into its corresponding graph dataset the script simply skip repeating the task. The last argument, specifies whether the whole event log should be converted to a graph dataset or only steady state traces are converted. In the latter case, a dictionary that divides traces into steady-state and non-steady-state is required (all dictionaries used in our experiments are collected in **raw_dataset** directory).

In order to convert all event logs into thier corresponding graph datasets you can run the following script:
```
bash CONVERT.sh
```
Once training and evaluation is done, you can repeat the conversion process with only steady-state traces running the following script:
```
bash CONVERT_SSD.sh
```

**<a name="part4">4. Training and evaluation of PGTNet models for remaining time prediction:</a>**

To train and evaluate PGTNet, we employ the implementation of [GraphGPS: General Powerful Scalable Graph Transformers](https://github.com/rampasek/GraphGPS). However, in order to use it for remaining time prediction of business process instances, you need to adjust some part of the original implementation. This can be achieved by running the following command:
```
python file_transfer.py
```
This script copies 5 important python scripts which take care of all necessary adjustments to the original implementation of GPS Graph Transformer recipe. Training is done using the relevant .yml configuration file which specifies all hyperparameters and training parameters. All configuration files required to train PGTNet based on the event logs used in our experiments are collected [here](https://github.com/Keyvan-Amiri-Elyasi/PGTNet4SSD/tree/main/training_configs). Similarly, all **inference configuration files** that are used in our experiments are collected [here](https://github.com/keyvan-amiri/PGTNet/tree/main/evaluation_configs). The **file_transfer.py** script also copy all required configuration files for training and evaluation of PGTNet to the relevant folder (i.e., configs/GPS) in **GPS repository**.

For training and evaluation of PGTNet, you need to navigate to the root directory of **GPS repository** and run **main.py** script. To train and evaluate PGTNet for all event logs, you need to run the following script: 
```
cd ..
bash PGTNet.sh
```
In our experiment, we executed this script first for all traces in each event logs, and then executed the same script only for steady-state traces.
Training results are saved in a seperate folder which is located in the **results** folder in the root directory of **GPS repository**.

**<a name="part5">5. DALSTM model:</a>**

[DALSTM](https://ieeexplore.ieee.org/abstract/document/8285184): An LSTM-based approach that was recently shown to have superior results among LSTMs used for remaining time prediction. To implement this baseline, we used the [**pmdlcompararator**](https://gitlab.citius.usc.es/efren.rama/pmdlcompararator) gitlab repository of a recently published [benchamrk](https://ieeexplore.ieee.org/abstract/document/9667311).

In order to train and evaluate DALSTM models, you need to navigate to [this](https://github.com/Keyvan-Amiri-Elyasi/PGTNet4SSD/tree/main/baselines/dalstm) directory and run the following script:
```
bash DALSTM.sh
```

**<a name="part6">6. DUMMY model:</a>**

A simple baseline that predicts the average remaining time of all training prefixes with the same length k as a given prefix. Training and inference with DUMMY model is done using a Jupyter Notebook that is provided in [this](https://github.com/Keyvan-Amiri-Elyasi/PGTNet4SSD/tree/main/baselines/dummy) folder.
