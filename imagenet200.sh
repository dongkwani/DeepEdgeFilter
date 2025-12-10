ARCH=ResNet18   # Vit-B32
MODE=average
LAYER=2
CONFIG=v11

python main.py --dataset imageNet200 --arch ${ARCH} --mode ${MODE} --pos ${LAYER} --config config/${MODE}/${CONFIG}.yaml --bsz 64
