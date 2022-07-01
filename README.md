# FLAT

Code for the paper [Adversarial Training for Improving Model Robustness? Look at Both Prediction and Interpretation](https://www.aaai.org/AAAI22Papers/AAAI-2735.ChenH.pdf)

### Data
Download the [data](https://drive.google.com/drive/folders/1J18AsUKuBYFtHmV0b1pfyd93G_lb2eLQ?usp=sharing) and put it in the same folder with the code.

### Preparation
- Install the packages and toolkits in `requirements.txt`
- `cd` into `CNN_LSTM` and `BERT_DeBERTa` for running experiments for CNN/LSTM and BERT/DeBERTa respectively

### Training base models

**Training CNN/LSTM base models**

For IMDB, set `--max_seq_length 250`. Fine-tune hyperparameters (e.g. learning rate, the number of hidden units) on each dataset.
```
python train.py train --gpu_id 2 --model cnn/lstm --dataset sst2/imdb/ag/trec --task base --batch-size 64 --epochs 10 --learning-rate 0.01 --max_seq_length 50
```

**Training BERT/DeBERTa base models**

Fine-tune hyperparameters (e.g. learning rate, weight decay) on each dataset.
```
python train.py train --gpu_id 2 --model bert/deberta --dataset sst2/imdb/ag/trec --task base --epochs 10 --learning-rate 1e-5
```

### Adversarial training

**Adversarial training for CNN/LSTM**

For IMDB, set `--max_seq_length 250`. Fine-tune hyperparameters (e.g. learning rate, the number of hidden units) on each dataset.
```
python train.py train --attack textfooler/pwws --gpu_id 2 --model cnn/lstm --dataset sst2/imdb/ag/trec --task adv --batch-size 64 --epochs 30 --learning-rate 0.01 --max_seq_length 50 --num-clean-epochs 10
```

**Adversarial training for BERT/DeBERTa**

Fine-tune hyperparameters (e.g. learning rate, weight decay) on each dataset.
```
python train.py train --attack textfooler --gpu_id 0 --model bert --dataset trec --task adv --epochs 39 --learning-rate 3e-5 --low_freq 0 --max_seq_length 15 --num-clean-epochs 10
```


### Acknowledgments
The code was built on [TextAttack](https://github.com/QData/TextAttack) and [Hugging Face/Transformers](https://github.com/huggingface/transformers)

### Reference:
If you find this repository helpful, please cite our paper:
```bibtex
@inproceedings{chen2022adversarial,
    title={Adversarial Training for Improving Model Robustness? Look at Both Prediction and Interpretation},
    author={Chen, Hanjie and Ji, Yangfeng},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    year={2022}
}
```
