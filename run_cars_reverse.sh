python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_pfix_confusion_103_threshold50 \
--helpful Influential_data/cars_ProxyNCA_pfix_helpful_testcls0_threshold50.npy \
--harmful Influential_data/cars_ProxyNCA_pfix_harmful_testcls0_threshold50.npy --model_dir models/dvi_data_cars_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 0 --harmful_weight 2 \
--seed 4 --config config/cars_reweight.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_pfix_confusion_111_threshold50 \
--helpful Influential_data/cars_ProxyNCA_pfix_helpful_testcls1_threshold50.npy \
--harmful Influential_data/cars_ProxyNCA_pfix_harmful_testcls1_threshold50.npy --model_dir models/dvi_data_cars_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 0 --harmful_weight 2 \
--seed 4 --config config/cars_reweight.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_pfix_confusion_139_threshold50 \
--helpful Influential_data/cars_ProxyNCA_pfix_helpful_testcls2_threshold50.npy \
--harmful Influential_data/cars_ProxyNCA_pfix_harmful_testcls2_threshold50.npy --model_dir models/dvi_data_cars_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 0 --harmful_weight 2 \
--seed 4 --config config/cars_reweight.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_pfix_confusion_178_threshold50 \
--helpful Influential_data/cars_ProxyNCA_pfix_helpful_testcls3_threshold50.npy \
--harmful Influential_data/cars_ProxyNCA_pfix_harmful_testcls3_threshold50.npy --model_dir models/dvi_data_cars_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 0 --harmful_weight 2 \
--seed 4 --config config/cars_reweight.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_pfix_confusion_182_threshold50 \
--helpful Influential_data/cars_ProxyNCA_pfix_helpful_testcls4_threshold50.npy \
--harmful Influential_data/cars_ProxyNCA_pfix_harmful_testcls4_threshold50.npy --model_dir models/dvi_data_cars_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 0 --harmful_weight 2 \
--seed 4 --config config/cars_reweight.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_pfix_confusion_128_threshold50 \
--helpful Influential_data/cars_ProxyNCA_pfix_helpful_testcls5_threshold50.npy \
--harmful Influential_data/cars_ProxyNCA_pfix_harmful_testcls5_threshold50.npy --model_dir models/dvi_data_cars_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 0 --harmful_weight 2 \
--seed 4 --config config/cars_reweight.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_pfix_confusion_163_threshold50 \
--helpful Influential_data/cars_ProxyNCA_pfix_helpful_testcls6_threshold50.npy \
--harmful Influential_data/cars_ProxyNCA_pfix_harmful_testcls6_threshold50.npy --model_dir models/dvi_data_cars_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 0 --harmful_weight 2 \
--seed 4 --config config/cars_reweight.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_pfix_confusion_106_threshold50 \
--helpful Influential_data/cars_ProxyNCA_pfix_helpful_testcls7_threshold50.npy \
--harmful Influential_data/cars_ProxyNCA_pfix_harmful_testcls7_threshold50.npy --model_dir models/dvi_data_cars_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 0 --harmful_weight 2 \
--seed 4 --config config/cars_reweight.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_pfix_confusion_143_threshold50 \
--helpful Influential_data/cars_ProxyNCA_pfix_helpful_testcls8_threshold50.npy \
--harmful Influential_data/cars_ProxyNCA_pfix_harmful_testcls8_threshold50.npy --model_dir models/dvi_data_cars_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 0 --harmful_weight 2 \
--seed 4 --config config/cars_reweight.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_pfix_confusion_166_threshold50 \
--helpful Influential_data/cars_ProxyNCA_pfix_helpful_testcls9_threshold50.npy \
--harmful Influential_data/cars_ProxyNCA_pfix_harmful_testcls9_threshold50.npy \
--model_dir models/dvi_data_cars_4_lossProxyNCA_pfix/ResNet_512_Model \
--helpful_weight 0 --harmful_weight 2 \
--seed 4 --config config/cars_reweight.json

