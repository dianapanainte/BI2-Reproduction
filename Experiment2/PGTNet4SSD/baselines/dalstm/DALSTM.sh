export SEEDS=42
export DEVICE_ID=3
# export DATASET=HelpDesk
# python DALSTM.py --dataset ${DATASET} --seed ${SEEDS} --device ${DEVICE_ID} --ssd
# export DATASET=Sepsis
# python DALSTM.py --dataset ${DATASET} --seed ${SEEDS} --device ${DEVICE_ID} --ssd
# export DATASET=BPIC15_2
# python DALSTM.py --dataset ${DATASET} --seed ${SEEDS} --device ${DEVICE_ID} --ssd
# export DATASET=BPIC15_3
# python DALSTM.py --dataset ${DATASET} --seed ${SEEDS} --device ${DEVICE_ID} --ssd
# export DATASET=BPIC15_4
# python DALSTM.py --dataset ${DATASET} --seed ${SEEDS} --device ${DEVICE_ID} --ssd
# export DATASET=BPIC15_5
# python DALSTM.py --dataset ${DATASET} --seed ${SEEDS} --device ${DEVICE_ID} --ssd
# export DATASET=BPI_Challenge_2012
# python DALSTM.py --dataset ${DATASET} --seed ${SEEDS} --device ${DEVICE_ID} --ssd
export DATASET=Hospital
python DALSTM.py --dataset ${DATASET} --seed ${SEEDS} --device ${DEVICE_ID} --ssd
# export DATASET=BPIC15_1
# python DALSTM.py --dataset ${DATASET} --seed ${SEEDS} --device ${DEVICE_ID} --ssd