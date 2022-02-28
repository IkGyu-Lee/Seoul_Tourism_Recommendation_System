### Seoul Tourism Recommendation System using MF & NCF(GMF, MLP, NeuMF without pretraining, NeuMF with pretraining)

>This repository contains a demo based on pretrained models.
> 
>Train Dataset is private.

### Directory Tree
```python
├── README.md
├── dataset
│   ├── congestion.pkl
│   ├── destination_id_name_genre_coordinate.pkl
│   └── seoul_gu_dong_coordinate.pkl
├── demo.py
├── main.py
├── data_utils.py
├── evaluate.py
├── parser.py
├── model_visitor
│   ├── Create_userId.py
│   ├── GMF.py
│   ├── MLP.py
│   └── NeuMF.py
├── model_congestion
│   ├── GMF.py
│   └── MF.py
├── pretrain_model
│   ├── GMF.z01 ... GMF.zip
│   ├── MLP.z01 ... MLP.zip
│   └── NeuMF0.z01 ... NeuMF0.zip
├── create_congestion.py
└── csv_to_pickle.py
```

### Dependency
```bash
Python        >= 3.7
tokenizers    >= 0.9.4
torch         >= 1.10.2
konlpy        >= 0.6.0
pandas        >= 1.3.5
numpy         >= 1.21.5
```

### Development Environment
- OS: ubuntu
- IDE: vim
- GPU: NVIDIA RTX A6000

### Unzip Pretrained Model
```bash
cd pretrain_model
cat GMF.* > pretrain_GMF.zip
unzip pretrain_GMF.zip
cat MLP.* > pretrain_MLP.zip
unzip pretrain_MLP.zip
cat NeuMF0.* > pretrain_NeuMF0.zip
unzip pretrain_NeuMF0.zip
```

### Change Model in demo.py
```python
def define_args():
    use_pretrain = False
    model_name = 'NeuMF'  # Choice GMF, MLP, NeuMF
    epochs = 10           # Choice 20,  10,  10
    num_factors = 36      # Choice 36,  24,  36
    return use_pretrain, model_name, epochs, num_factors
```

### Quick Start
```bash
python demo.py
```

### Demo Video
<img width="100%" src="demo_video/demo_video.gif"/>