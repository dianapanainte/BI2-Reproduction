#!/bin/bash
#SBATCH --job-name=conversion
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --partition=cpu
#SBATCH --chdir=/ceph/kamiriel/GraphGPS4SSD/PGTNet4SSD/transformation
export DIRECTORY=conversion_configs
# export CONFIG=helpdesk.yaml
# python GTconvertor.py ${DIRECTORY} ${CONFIG} --overwrite 
# export CONFIG=sepsis.yaml
# python GTconvertor.py ${DIRECTORY} ${CONFIG} --overwrite
export CONFIG=bpic15m1.yaml
python GTconvertor.py ${DIRECTORY} ${CONFIG} --overwrite
# export CONFIG=bpic15m2.yaml
# python GTconvertor.py ${DIRECTORY} ${CONFIG} --overwrite
# export CONFIG=bpic15m3.yaml
# python GTconvertor.py ${DIRECTORY} ${CONFIG} --overwrite
# export CONFIG=bpic15m4.yaml
# python GTconvertor.py ${DIRECTORY} ${CONFIG} --overwrite
# export CONFIG=bpic15m5.yaml
# python GTconvertor.py ${DIRECTORY} ${CONFIG} --overwrite
# export CONFIG=bpic12.yaml
# python GTconvertor.py ${DIRECTORY} ${CONFIG} --overwrite
# export CONFIG=hospital.yaml
# python GTconvertor.py ${DIRECTORY} ${CONFIG} --overwrite