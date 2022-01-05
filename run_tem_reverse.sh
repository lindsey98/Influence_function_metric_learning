python train_sample_reweight.py --dataset cub \
--loss-type ProxyNCA_pfix_confusion_638_595 \
--helpful Confuse_pair_influential_data/helpful_indices_638_595.npy \
--harmful Confuse_pair_influential_data/harmful_indices_638_595.npy \
--model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight -10 --harmful_weight 10 \
--seed 4 --config config/cub_reweight.json

python train_sample_reweight.py --dataset cub \
--loss-type ProxyNCA_pfix_confusion_638_595_20samples \
--helpful Confuse_pair_influential_data/20helpful_indices_638_595.npy \
--harmful Confuse_pair_influential_data/20harmful_indices_638_595.npy \
--model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight -10 --harmful_weight 10 \
--seed 4 --config config/cub_reweight.json

python train_sample_reweight.py --dataset cub \
--loss-type ProxyNCA_pfix_confusion_638_595_50samples \
--helpful Confuse_pair_influential_data/50helpful_indices_638_595.npy \
--harmful Confuse_pair_influential_data/50harmful_indices_638_595.npy \
--model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight -10 --harmful_weight 10 \
--seed 4 --config config/cub_reweight.json

python train_sample_reweight.py --dataset cub \
--loss-type ProxyNCA_pfix_confusion_638_595_Allsamples \
--helpful Confuse_pair_influential_data/Allhelpful_indices_638_595.npy \
--harmful Confuse_pair_influential_data/Allharmful_indices_638_595.npy \
--model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight -10 --harmful_weight 10 \
--seed 4 --config config/cub_reweight.json