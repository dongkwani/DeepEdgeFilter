MODE=average
LAYER=11
CONFIG=vit_v7

python main.py --dataset cifar10 --arch ViT-B32 --mode ${MODE} --pos ${LAYER} --config config/${MODE}/${CONFIG}.yaml
