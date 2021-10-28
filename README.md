# Automatic-Dental-Plaque-Segmentation-based-on-Local-to-global-Features-Fused-Self-attention-Network
PyTorch implementation of paper "Automatic-Dental-Plaque-Segmentation-based-on-Local-to-global-Features-Fused-Self-attention-Network"

![image](images/1618989831(1).jpg)

### Requirements
Please, install the following packages
- numpy
- pytorch-1.8.1
- torchvision-0.9.1
- tqdm

### Teeth Dataset
### Train anet
```sh
cd anet/git_ocnet/scripts/
CUDA_VISIBLE_DEVICES=0 python train.py --model ocnet \
    --backbone resnet101 --dataset teeth \
    --lr 0.0001 --epochs 100 --batch-size 2
```

### Evaluate anet
```sh
cd anet/git_ocnet/scripts/
python eval.py --model ocnet --backbone resnet152 --dataset teeth
```
## Reference
This project is based on https://github.com/Tramac/awesome-semantic-segmentation-pytorch.git

## Tips
If you have any questions about our work, please do not hesitate to contact us by emails.
