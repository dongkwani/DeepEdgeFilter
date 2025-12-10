DATASET=cifar10 # cifar100
MODE=average
LAYER=11
CONFIG=vit_v7

python main.py --dataset ${DATASET} --arch ViT-B32 --mode ${MODE} --pos ${LAYER} --config config/${MODE}/${CONFIG}.yaml