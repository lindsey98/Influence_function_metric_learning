
Debugging and Explaining Metric Learning Approach: An Influence Function Perspective
==============================================================================

## Requirements
```
pip install -r requirements.txt
```

## Download datasets from
[Caltech_birds2011](http://www.vision.caltech.edu/datasets/cub_200_2011/)\[1\]

[Cars196](https://ai.stanford.edu/%7Ejkrause/cars/car_dataset.html)\[2\]

[In-shop Clothes Retrieval](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html)\[3\]

Put them under mnt/datasets/

## Training details
- We follow the train-test split provided by the original datasets
- We use the same hyperparameters specified in [Proxy-NCA++](https://github.com/euwern/proxynca_pp), except for In-Shop we reduce the batch size to 32 due to the limit of our GPU resources.

## Project Structure
- Training the original models
  - Training the DML models with Proxy-NCA++ loss or with SoftTriple loss
```
python train.py --dataset [cub|cars|inshop] \
--loss-type ProxyNCA_prob_orig \
--seed [0|1|2|3|4] \
--config [config/cub_ProxyNCA_prob_orig.json|config/cars_ProxyNCA_prob_orig.json|config/inshop_ProxyNCA_prob_orig.json]
```
```
python train.py --dataset [cub|cars|inshop] \
--loss-type SoftTriple \
--seed [0|1|2|3|4] \
--config [config/cub_SoftTriple.json|config/cars_SoftTriple.json|config/inshop_SoftTriple.json]
```

- Training the models with mislabelled data
  - Training the DML models with Proxy-NCA++ loss or with SoftTriple loss
```
python train_noisydata.py --dataset [cub_noisy|cars_noisy|inshop_noisy] \
--loss-type ProxyNCA_prob_orig_noisy_0.1 \
--seed [0|1|2|3|4] \
--mislabel_percentage 0.1 \
--config [config/cub_ProxyNCA_prob_orig.json|config/cars_ProxyNCA_prob_orig.json|config/inshop_ProxyNCA_prob_orig.json]
```
```
python train_noisydata.py --dataset [cub_noisy|cars_noisy|inshop_noisy] \
--loss-type SoftTriple_noisy_0.1 \
--seed [0|1|2|3|4] \
--mislabel_percentage 0.1 \
--config [config/cub_SoftTriple.json|config/cars_SoftTriple.json|config/inshop_SoftTriple.json]
```

- DML training experiment (Table 1): comparing <img src="https://render.githubusercontent.com/render/math?math=\Delta d(p)"> or <img src="https://render.githubusercontent.com/render/math?math=\Delta d(G_p)">
  
  &nbsp;&nbsp;&nbsp;See experiments/EIF_group_confusion.py, experiments/IF_group_confusion.py, experiments/EIF_pair_confusion.py, experiments/IF_pair_confusion.py

- Mislabelled detection experiment

  &nbsp;&nbsp;&nbsp;See experiments/EIFvsIF_mislabel_evaluation.py

- Field study 

  &nbsp;&nbsp;&nbsp;See experiments/sample_recommendation_evaluation.py

- Implementation of EIF
  &nbsp;&nbsp;&nbsp; See Influence_function/influence_function.py

## Results
- All trained models: https://drive.google.com/drive/folders/1yKR8BWPxM5MtUwjHzU7k-Mbzse73Ij_t?usp=sharing
- For the detailed statistics of Table 1, please see https://docs.google.com/spreadsheets/d/1f4OXVLO2Mu2CHrBVm72a2ztTHx5nNG92dczTNNw7io4/edit?usp=sharing

## References
[1] Wah, C., Branson, S., Welinder, P., Perona, P., & Belongie, S. (2011). The caltech-ucsd birds-200-2011 dataset.

[2] Krause, J., Stark, M., Deng, J., & Fei-Fei, L. (2013). 3d object representations for fine-grained categorization. In Proceedings of the IEEE international conference on computer vision workshops (pp. 554-561).

[3] Liu, Z., Luo, P., Qiu, S., Wang, X., & Tang, X. (2016). Deepfashion: Powering robust clothes recognition and retrieval with rich annotations. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1096-1104).
