# FLAT

Code for the paper [Adversarial Training for Improving Model Robustness? Look at Both Prediction and Interpretation](https://www.aaai.org/AAAI22Papers/AAAI-2735.ChenH.pdf)

### Data
Download the [data](https://drive.google.com/drive/folders/1J18AsUKuBYFtHmV0b1pfyd93G_lb2eLQ?usp=sharing) and put it in the same folder with the code.

### Preparation
- Install the packages and toolkits in `requirements.txt`
- `cd` into `CNN_LSTM` and `BERT_DeBERTa` for running experiments for CNN/LSTM and BERT/DeBERTa respectively

### Training

**Training CNN/LSTM base models**
For IMDB, set `--max_seq_length 250`. Fine-tune hyperparameters (e.g. learning rate, the number of hidden units) on each dataset.
```
python train.py train --gpu_id 2 --model cnn/lstm --dataset sst2/imdb/ag/trec --task base --batch-size 64 --epochs 10 --learning-rate 0.01 --max_seq_length 50
```

**Training BERT/DeBERTa base models**
Fine-tune hyperparameters (e.g. learning rate, weight decay) on each dataset.
```
python train_main.py train --gpu_id 2 --model bert/deberta --dataset sst2 --task base --epochs 10 --learning-rate 1e-5
```

Note that the code is for the e-SNLI dataset. For the BERT model on other datasets, set `--task_name` with the data name `quora/qqp/mrpc`. For the MRPC dataset, set `--max_seq_length` as `100`. 

For the DAttn model on other datasets, utilize the corresponding `DataLoader` and `Sampler` by revising lines `1, 2, 6` in `load_data.py`. Set `--data_path` as `train.tsv`. Set the output dimension of the final linear layer of the DAttn model as `2` (line 59 in `deatten_model.py`).

### Explain models on test data via GMASK:
Explain the well-trained model by running
```
python explain.py
```
For each test example, we save the words and their indexes in the order of importance as the explanation.

### Acknowledgments
The code was built on
- https://github.com/huggingface/transformers
- https://github.com/asappresearch/rationale-alignment
- https://github.com/libowen2121/SNLI-decomposable-attention

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
