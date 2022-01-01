python train_sample_reweight.py --dataset cub \
--loss-type ProxyNCA_pfix_confusion_143_threshold50_reverse \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_testcls0_threshold50.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_testcls0_threshold50.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 0 --harmful_weight 2 \
--seed 4 --config config/cub_reweight.json

python train_sample_reweight.py --dataset cub \
--loss-type ProxyNCA_pfix_confusion_117_threshold50_reverse \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_testcls1_threshold50.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_testcls1_threshold50.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 0 --harmful_weight 2 \
--seed 4 --config config/cub_reweight.json

python train_sample_reweight.py --dataset cub \
--loss-type ProxyNCA_pfix_confusion_178_threshold50_reverse \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_testcls2_threshold50.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_testcls2_threshold50.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 0 --harmful_weight 2 \
--seed 4 --config config/cub_reweight.json

python train_sample_reweight.py --dataset cub \
--loss-type ProxyNCA_pfix_confusion_111_threshold50_reverse \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_testcls3_threshold50.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_testcls3_threshold50.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 0 --harmful_weight 2 \
--seed 4 --config config/cub_reweight.json

python train_sample_reweight.py --dataset cub \
--loss-type ProxyNCA_pfix_confusion_126_threshold50_reverse \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_testcls4_threshold50.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_testcls4_threshold50.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 0 --harmful_weight 2 \
--seed 4 --config config/cub_reweight.json

python train_sample_reweight.py --dataset cub \
--loss-type ProxyNCA_pfix_confusion_128_threshold50_reverse \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_testcls5_threshold50.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_testcls5_threshold50.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 0 --harmful_weight 2 \
--seed 4 --config config/cub_reweight.json

python train_sample_reweight.py --dataset cub \
--loss-type ProxyNCA_pfix_confusion_124_threshold50_reverse \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_testcls6_threshold50.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_testcls6_threshold50.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 0 --harmful_weight 2 \
--seed 4 --config config/cub_reweight.json

python train_sample_reweight.py --dataset cub \
--loss-type ProxyNCA_pfix_confusion_196_threshold50_reverse \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_testcls7_threshold50.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_testcls7_threshold50.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 0 --harmful_weight 2 \
--seed 4 --config config/cub_reweight.json

python train_sample_reweight.py --dataset cub \
--loss-type ProxyNCA_pfix_confusion_142_threshold50_reverse \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_testcls8_threshold50.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_testcls8_threshold50.npy --model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 0 --harmful_weight 2 \
--seed 4 --config config/cub_reweight.json

python train_sample_reweight.py --dataset cub \
--loss-type ProxyNCA_pfix_confusion_195_threshold50_reverse \
--helpful Influential_data/cub_ProxyNCA_pfix_helpful_testcls9_threshold50.npy \
--harmful Influential_data/cub_ProxyNCA_pfix_harmful_testcls9_threshold50.npy \
--model_dir models/dvi_data_cub_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 0 --harmful_weight 2 \
--seed 4 --config config/cub_reweight.json

