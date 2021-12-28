python train_sample_reweight.py --loss-type ProxyNCA_pfix_intravar_178_threshold10 \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_intravar_threshold10_testcls1.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_intravar_threshold10_testcls1.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 1 --harmful_weight -1
python train_sample_reweight.py --loss-type ProxyNCA_pfix_intravar_117_threshold10 \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_intravar_threshold10_testcls2.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_intravar_threshold10_testcls2.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 1 --harmful_weight -1
python train_sample_reweight.py --loss-type ProxyNCA_pfix_intravar_172_threshold10 \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_intravar_threshold10_testcls3.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_intravar_threshold10_testcls3.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 1 --harmful_weight -1
python train_sample_reweight.py --loss-type ProxyNCA_pfix_intravar_116_threshold10 \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_intravar_threshold10_testcls4.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_intravar_threshold10_testcls4.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 1 --harmful_weight -1
python train_sample_reweight.py --loss-type ProxyNCA_pfix_intravar_196_threshold10 \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_intravar_threshold10_testcls5.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_intravar_threshold10_testcls5.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 1 --harmful_weight -1
python train_sample_reweight.py --loss-type ProxyNCA_pfix_intravar_130_threshold10 \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_intravar_threshold10_testcls6.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_intravar_threshold10_testcls6.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 1 --harmful_weight -1
python train_sample_reweight.py --loss-type ProxyNCA_pfix_intravar_124_threshold10 \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_intravar_threshold10_testcls7.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_intravar_threshold10_testcls7.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 1 --harmful_weight -1

