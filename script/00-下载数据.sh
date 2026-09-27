DATE=$(date +%Y%m%d)
BASE=/data/wangzhuo/01-code/02-project/dinov3_finetune/data

mkdir -p $BASE/$DATE

ossutil cp \
oss://roki-ai-ckb-test/wondron/008-dino-class/260725/original.zip \
$BASE/$DATE/class.zip