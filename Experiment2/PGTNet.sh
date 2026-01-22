export SEEDS=42
export CONFIG=configs/GPS/helpdesk-GPSwGraphormer-ckptbest.yaml
python main.py --cfg ${CONFIG} seed ${SEEDS}
export CONFIG=configs/GPS/helpdesk-GPSwGraphormer-ckptbest-eventinference.yaml
python main.py --cfg ${CONFIG} seed ${SEEDS}
export CONFIG=configs/GPS/sepsis-GPS+SNMLP-ckptbest.yaml
python main.py --cfg ${CONFIG} seed ${SEEDS}
export CONFIG=configs/GPS/sepsis-GPS+SNMLP-ckptbest-eventinference.yaml
python main.py --cfg ${CONFIG} seed ${SEEDS}
export CONFIG=configs/GPS/bpic2015m1-GPS+LapPE+RWSE-ckptbest.yaml
python main.py --cfg ${CONFIG} seed ${SEEDS}
export CONFIG=configs/GPS/bpic2015m1-GPS+LapPE+RWSE-ckptbest-eventinference.yaml
python main.py --cfg ${CONFIG} seed ${SEEDS}
export CONFIG=configs/GPS/bpic2015m2-GPS+LapPE+RWSE-ckptbest.yaml
python main.py --cfg ${CONFIG} seed ${SEEDS}
export CONFIG=configs/GPS/bpic2015m2-GPS+LapPE+RWSE-ckptbest-eventinference.yaml
python main.py --cfg ${CONFIG} seed ${SEEDS}
export CONFIG=configs/GPS/bpic2015m3-GPSwGraphormer-ckptbest.yaml
python main.py --cfg ${CONFIG} seed ${SEEDS}
export CONFIG=configs/GPS/bpic2015m3-GPSwGraphormer-ckptbest-eventinference.yaml
python main.py --cfg ${CONFIG} seed ${SEEDS}
export CONFIG=configs/GPS/bpic2015m4-GPS+LapPE+RWSE-ckptbest.yaml
python main.py --cfg ${CONFIG} seed ${SEEDS}
export CONFIG=configs/GPS/bpic2015m4-GPS+LapPE+RWSE-ckptbest-eventinference.yaml
python main.py --cfg ${CONFIG} seed ${SEEDS}
export CONFIG=configs/GPS/bpic2015m5-GPS+LapPE+RWSE-ckptbest.yaml
python main.py --cfg ${CONFIG} seed ${SEEDS}
export CONFIG=configs/GPS/bpic2015m5-GPS+LapPE+RWSE-ckptbest-eventinference.yaml
python main.py --cfg ${CONFIG} seed ${SEEDS}
export CONFIG=configs/GPS/bpic2012-GPS+LapPE+RWSE-ckptbest.yaml
python main.py --cfg ${CONFIG} seed ${SEEDS}
export CONFIG=configs/GPS/bpic2012-GPS+LapPE+RWSE-ckptbest-eventinference.yaml
python main.py --cfg ${CONFIG} seed ${SEEDS}
export CONFIG=configs/GPS/hospital-GPS+LapPE+RWSE-ckptbest.yaml
python main.py --cfg ${CONFIG} seed ${SEEDS}
export CONFIG=configs/GPS/hospital-GPS+LapPE+RWSE-ckptbest-eventinference.yaml
python main.py --cfg ${CONFIG} seed ${SEEDS}