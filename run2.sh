python train_sample_reweight.py --loss-type ProxyNCA_pfix_confusion_124_threshold50_40epochs \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_testcls7_threshold50.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_testcls7_threshold50.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 2 --harmful_weight 0
python train_sample_reweight.py --loss-type ProxyNCA_pfix_confusion_111_threshold50_40epochs \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_testcls8_threshold50.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_testcls8_threshold50.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 2 --harmful_weight 0
python train_sample_reweight.py --loss-type ProxyNCA_pfix_confusion_144_threshold50_40epochs \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_testcls9_threshold50.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_testcls9_threshold50.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 2 --harmful_weight 0