python train_sample_reweight.py --loss-type ProxyNCA_pfix_intravar_178 \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_intravar_testcls1.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_intravar_testcls1.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 1 --harmful_weight -1
python train_sample_reweight.py --loss-type ProxyNCA_pfix_intravar_117 \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_intravar_testcls2.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_intravar_testcls2.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 1 --harmful_weight -1
python train_sample_reweight.py --loss-type ProxyNCA_pfix_intravar_172 \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_intravar_testcls3.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_intravar_testcls3.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 1 --harmful_weight -1
python train_sample_reweight.py --loss-type ProxyNCA_pfix_intravar_116 \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_intravar_testcls4.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_intravar_testcls4.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 1 --harmful_weight -1
python train_sample_reweight.py --loss-type ProxyNCA_pfix_intravar_196 \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_intravar_testcls5.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_intravar_testcls5.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 1 --harmful_weight -1
python train_sample_reweight.py --loss-type ProxyNCA_pfix_intravar_130 \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_intravar_testcls6.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_intravar_testcls6.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 1 --harmful_weight -1
python train_sample_reweight.py --loss-type ProxyNCA_pfix_intravar_124 \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_intravar_testcls7.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_intravar_testcls7.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 1 --harmful_weight -1

