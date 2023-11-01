# Code usage method (taking Quanzhou as an example)

## train
```
cd gpt_flow
### initial model
python flowgpt.py --trainer.max_iters=101
### train model
python train_all.py
### finetune model
python flowgpt_finetune.py
```

## validate
```
cd gpt_flow
python generate_flow_qz.py
python generate_flow_qz_finetune.py
```

## calculate RMSE, MAE
```
cd compare
```
