DATASET=cifar10 # cifar100, imageNet
MODE=average # none, median
LAYER=1
CONFIG=v11

python main.py --dataset ${DATASET} --arch WRN-28-10 --mode ${MODE} --pos ${LAYER} --config config/${MODE}/${CONFIG}.yaml
