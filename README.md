
Debugging and Explaining Metric Learning Approach: An Influence Function Perspective
==============================================================================

## Requirements
```
pip install -r requirements.txt
```

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
python train.py --dataset [cub_noisy|cars_noisy|inshop_noisy] \
--loss-type ProxyNCA_prob_orig_noisy_0.1 \
--seed [0|1|2|3|4] \
--mislabel_percentage 0.1 \
--config [config/cub_ProxyNCA_prob_orig.json|config/cars_ProxyNCA_prob_orig.json|config/inshop_ProxyNCA_prob_orig.json]
```
```
python train.py --dataset [cub_noisy|cars_noisy|inshop_noisy] \
--loss-type SoftTriple_noisy_0.1 \
--seed [0|1|2|3|4] \
--mislabel_percentage 0.1 \
--config [config/cub_SoftTriple.json|config/cars_SoftTriple.json|config/inshop_SoftTriple.json]
```

- Implementation of EIF and IF

  See Influence_function/

- Plot the figure for the generalization errors

  See experiments/intro_fig.py

- Runtime evaluation

  See experiments/runtime_evaluation.py

- Table 1: comparing <img src="https://render.githubusercontent.com/render/math?math=\triangle d(p)"> or <img src="https://render.githubusercontent.com/render/math?math=\triangle d(G_p)">
  
  See experiments/EIF_group_confusion.py, experiments/IF_group_confusion.py, experiments/EIF_pair_confusion.py, experiments/IF_pair_confusion.py

- Mislabelled detection experiment

  See experiments/EIFvsIF_mislabel_evaluation.py

- Sample relabelling recommendation evaluation

  See experiments/sample_recommendation_evaluation.py