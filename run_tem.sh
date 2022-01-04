python train_sample_reweight.py --dataset cub \
--loss-type ProxyNCA_pfix_confusion_55_4204 \
--helpful Confuse_pair_influential_data/helpful_indices_55_4204.npy \
--harmful Confuse_pair_influential_data/harmful_indices_55_4204.npy \
--model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 10 --harmful_weight -10 \
--seed 4 --config config/cub_reweight.json