
Influence function for metric learning
==============================================================================

## Project Structure
```
Influence_function/
scripts/
explaination/
config/: Training config files
dataset/: Definition of dataloader
evaluation/: Performance evaluation scripts
networks.py: Network structure, i.e. ResNet50, BN-Inception
loss.py: Implementation of 3 losses, i.e. ProxyNCA++, ProxyAnchor, SoftTriple
train.py: Normal training scripts
train_sample_relabel.py: Fine-tuning scripts with relabelled data
train_sample_reweight.py: Fine-tuning scripts with reweighted data
```