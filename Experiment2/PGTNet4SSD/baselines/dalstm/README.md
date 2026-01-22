To replicate our results in our paper: [On the Use of Steady-State Detection for Process Mining: Achieving More Accurate Insights](https://link.springer.com/chapter/10.1007/978-3-031-94569-4_12) published in International Conference on Advanced Information Systems Engineering you can run [this python script](https://github.com/Keyvan-Amiri-Elyasi/PGTNet4SSD/blob/main/baselines/dalstm/DALSTM.py). To run the script for all datasets, you can use [this shell script](https://github.com/Keyvan-Amiri-Elyasi/PGTNet4SSD/blob/main/baselines/dalstm/DALSTM.sh).

If you aim to replicated our extended experiments for our paper in Information System Journal, you need to run the pre-processing and training separately. For preprocessing, you should run [this python script](https://github.com/Keyvan-Amiri-Elyasi/PGTNet4SSD/blob/main/baselines/dalstm/DALSTM_process.py) as shown in the following example for BPIC15_1 event log:

```
python DALSTM_process.py --dataset BPIC15_1
```

This will take care of pre-processing steps for all bucketing strategies included in our experiments. Once pre-processing is finished, you should run [this python script](https://github.com/Keyvan-Amiri-Elyasi/PGTNet4SSD/blob/main/baselines/dalstm/DALSTM_train_evaluate.py) as shown in the following example for the same event log:

```
python DALSTM_train_evaluate.py --dataset BPIC15_1 --bucketing SSD_B
```

The bucketing argument specifies the bucketing strategy (in this example based on steady state detection). 

All data attributes that are used by DALSTM model are included `preprocessing_config.yaml` file. Note that only categorical attributes in event level can be used by DALSTM model. We used the same attributes that are used by PGTNet to have a fair comparison. 
