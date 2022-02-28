### Seoul Tourism Recommendation System using MF & NCF(GMF, MLP, NeuMF without pretraining, NeuMF with pretraining)

>This repository contains a demo based on pretrained models.
> 
>Train Dataset is private.

### Quick Start
```bash
cd pretrain_model
cat GMF.* > pretrain_GMF.zip
unzip pretrain_GMF.zip
cat MLP.* > pretrain_MLP.zip
unzip pretrain_MLP.zip
cat NeuMF0.* > pretrain_NeuMF0.zip
unzip pretrain_NeuMF0.zip
python demo.py
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

### Demo Video
<img width="100%" src="demo_video/demo_video.gif"/>