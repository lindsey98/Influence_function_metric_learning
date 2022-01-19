python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_prob_orig_confusion_103_threshold50 \
--helpful Influential_data/cars_ProxyNCA_prob_orig_helpful_testcls0_threshold50.npy \
--harmful Influential_data/cars_ProxyNCA_prob_orig_harmful_testcls0_threshold50.npy \
--model_dir models/dvi_data_cars_3_lossProxyNCA_prob_orig/ResNet_512_Model \
--helpful_weight 2 --harmful_weight 0 \
--seed 3 --config config/cars_reweight.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_prob_orig_confusion_163_threshold50 \
--helpful Influential_data/cars_ProxyNCA_prob_orig_helpful_testcls1_threshold50.npy \
--harmful Influential_data/cars_ProxyNCA_prob_orig_harmful_testcls1_threshold50.npy --model_dir models/dvi_data_cars_3_lossProxyNCA_prob_orig/ResNet_512_Model \
--helpful_weight 2 --harmful_weight 0 \
--seed 3 --config config/cars_reweight.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_prob_orig_confusion_183_threshold50 \
--helpful Influential_data/cars_ProxyNCA_prob_orig_helpful_testcls2_threshold50.npy \
--harmful Influential_data/cars_ProxyNCA_prob_orig_harmful_testcls2_threshold50.npy --model_dir models/dvi_data_cars_3_lossProxyNCA_prob_orig/ResNet_512_Model \
--helpful_weight 2 --harmful_weight 0 \
--seed 3 --config config/cars_reweight.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_prob_orig_confusion_179_threshold50 \
--helpful Influential_data/cars_ProxyNCA_prob_orig_helpful_testcls3_threshold50.npy \
--harmful Influential_data/cars_ProxyNCA_prob_orig_harmful_testcls3_threshold50.npy --model_dir models/dvi_data_cars_3_lossProxyNCA_prob_orig/ResNet_512_Model \
--helpful_weight 2 --harmful_weight 0 \
--seed 3 --config config/cars_reweight.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_prob_orig_confusion_187_threshold50 \
--helpful Influential_data/cars_ProxyNCA_prob_orig_helpful_testcls4_threshold50.npy \
--harmful Influential_data/cars_ProxyNCA_prob_orig_harmful_testcls4_threshold50.npy --model_dir models/dvi_data_cars_3_lossProxyNCA_prob_orig/ResNet_512_Model \
--helpful_weight 2 --harmful_weight 0 \
--seed 3 --config config/cars_reweight.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_prob_orig_confusion_139_threshold50 \
--helpful Influential_data/cars_ProxyNCA_prob_orig_helpful_testcls5_threshold50.npy \
--harmful Influential_data/cars_ProxyNCA_prob_orig_harmful_testcls5_threshold50.npy --model_dir models/dvi_data_cars_3_lossProxyNCA_prob_orig/ResNet_512_Model \
--helpful_weight 2 --harmful_weight 0 \
--seed 3 --config config/cars_reweight.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_prob_orig_confusion_166_threshold50 \
--helpful Influential_data/cars_ProxyNCA_prob_orig_helpful_testcls6_threshold50.npy \
--harmful Influential_data/cars_ProxyNCA_prob_orig_harmful_testcls6_threshold50.npy --model_dir models/dvi_data_cars_3_lossProxyNCA_prob_orig/ResNet_512_Model \
--helpful_weight 2 --harmful_weight 0 \
--seed 3 --config config/cars_reweight.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_prob_orig_confusion_176_threshold50 \
--helpful Influential_data/cars_ProxyNCA_prob_orig_helpful_testcls7_threshold50.npy \
--harmful Influential_data/cars_ProxyNCA_prob_orig_harmful_testcls7_threshold50.npy --model_dir models/dvi_data_cars_3_lossProxyNCA_prob_orig/ResNet_512_Model \
--helpful_weight 2 --harmful_weight 0 \
--seed 3 --config config/cars_reweight.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_prob_orig_confusion_186_threshold50 \
--helpful Influential_data/cars_ProxyNCA_prob_orig_helpful_testcls8_threshold50.npy \
--harmful Influential_data/cars_ProxyNCA_prob_orig_harmful_testcls8_threshold50.npy --model_dir models/dvi_data_cars_3_lossProxyNCA_prob_orig/ResNet_512_Model \
--helpful_weight 2 --harmful_weight 0 \
--seed 3 --config config/cars_reweight.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_prob_orig_confusion_177_threshold50 \
--helpful Influential_data/cars_ProxyNCA_prob_orig_helpful_testcls9_threshold50.npy \
--harmful Influential_data/cars_ProxyNCA_prob_orig_harmful_testcls9_threshold50.npy \
--model_dir models/dvi_data_cars_3_lossProxyNCA_prob_orig/ResNet_512_Model \
--helpful_weight 2 --harmful_weight 0 \
--seed 3 --config config/cars_reweight.json

