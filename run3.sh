python train_sample_reweight.py --loss-type ProxyNCA_pfix_confusion_143_145 \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_confusion_testcls0.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_confusion_testcls0.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 1 --harmful_weight -1
python train_sample_reweight.py --loss-type ProxyNCA_pfix_confusion_111_110 \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_confusion_testcls8.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_confusion_testcls8.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 1 --harmful_weight -1
python train_sample_reweight.py --loss-type ProxyNCA_pfix_confusion_144_142 \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_confusion_testcls9.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_confusion_testcls9.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 1 --harmful_weight -1

