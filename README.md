
Debugging and Explaining Metric Learning Approach: An Influence Function Perspective
==============================================================================

## Requirements
```
pip install -r requirements.txt
```

## Download datasets from
[Caltech_birds2011](http://www.vision.caltech.edu/datasets/cub_200_2011/)\[1\]

[Cars196](https://ai.stanford.edu/%7Ejkrause/cars/car_dataset.html)\[2\]

[Stanford Online Product](https://cvgl.stanford.edu/projects/lifted_struct/)\[3\]


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

- Implementation of EIF and IF

  &nbsp;&nbsp;&nbsp;See Influence_function/

- Plot the figure for the generalization errors

  &nbsp;&nbsp;&nbsp;See experiments/intro_fig.py

- Runtime evaluation

  &nbsp;&nbsp;&nbsp;See experiments/runtime_evaluation.py

- Table 1: comparing <img src="https://render.githubusercontent.com/render/math?math=\triangle d(p)"> or <img src="https://render.githubusercontent.com/render/math?math=\triangle d(G_p)">
  
  &nbsp;&nbsp;&nbsp;See experiments/EIF_group_confusion.py, experiments/IF_group_confusion.py, experiments/EIF_pair_confusion.py, experiments/IF_pair_confusion.py

- Mislabelled detection experiment

  &nbsp;&nbsp;&nbsp;See experiments/EIFvsIF_mislabel_evaluation.py

- Sample relabelling recommendation evaluation

  &nbsp;&nbsp;&nbsp;See experiments/sample_recommendation_evaluation.py

## Results
- All trained models: https://drive.google.com/drive/folders/1yKR8BWPxM5MtUwjHzU7k-Mbzse73Ij_t?usp=sharing
- For the detailed statistics of Table 1, please see https://docs.google.com/spreadsheets/d/1f4OXVLO2Mu2CHrBVm72a2ztTHx5nNG92dczTNNw7io4/edit?usp=sharing

## References
[1] Wah, C., Branson, S., Welinder, P., Perona, P., & Belongie, S. (2011). The caltech-ucsd birds-200-2011 dataset.

[2] Krause, J., Stark, M., Deng, J., & Fei-Fei, L. (2013). 3d object representations for fine-grained categorization. In Proceedings of the IEEE international conference on computer vision workshops (pp. 554-561).

[3] Oh Song, H., Xiang, Y., Jegelka, S., & Savarese, S. (2016). Deep metric learning via lifted structured feature embedding. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4004-4012).
