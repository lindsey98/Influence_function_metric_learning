python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_prob_orig_confusion_103 \
--helpful Influential_data/cars_ProxyNCA_prob_orig_helpful_testcls0.npy \
--harmful Influential_data/cars_ProxyNCA_prob_orig_harmful_testcls0.npy \
--model_dir models/dvi_data_cars_3_lossProxyNCA_prob_orig/ResNet_512_Model \
--helpful_weight 2 --harmful_weight 0 \
--seed 3 --config config/cars_reweight_ProxyNCA_prob_orig.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_prob_orig_confusion_163 \
--helpful Influential_data/cars_ProxyNCA_prob_orig_helpful_testcls1.npy \
--harmful Influential_data/cars_ProxyNCA_prob_orig_harmful_testcls1.npy \
--model_dir models/dvi_data_cars_3_lossProxyNCA_prob_orig/ResNet_512_Model \
--helpful_weight 2 --harmful_weight 0 \
--seed 3 --config config/cars_reweight_ProxyNCA_prob_orig.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_prob_orig_confusion_183 \
--helpful Influential_data/cars_ProxyNCA_prob_orig_helpful_testcls2.npy \
--harmful Influential_data/cars_ProxyNCA_prob_orig_harmful_testcls2.npy \
--model_dir models/dvi_data_cars_3_lossProxyNCA_prob_orig/ResNet_512_Model \
--helpful_weight 2 --harmful_weight 0 \
--seed 3 --config config/cars_reweight_ProxyNCA_prob_orig.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_prob_orig_confusion_179 \
--helpful Influential_data/cars_ProxyNCA_prob_orig_helpful_testcls3.npy \
--harmful Influential_data/cars_ProxyNCA_prob_orig_harmful_testcls3.npy \
--model_dir models/dvi_data_cars_3_lossProxyNCA_prob_orig/ResNet_512_Model \
--helpful_weight 2 --harmful_weight 0 \
--seed 3 --config config/cars_reweight_ProxyNCA_prob_orig.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_prob_orig_confusion_187 \
--helpful Influential_data/cars_ProxyNCA_prob_orig_helpful_testcls4.npy \
--harmful Influential_data/cars_ProxyNCA_prob_orig_harmful_testcls4.npy --model_dir models/dvi_data_cars_3_lossProxyNCA_prob_orig/ResNet_512_Model \
--helpful_weight 2 --harmful_weight 0 \
--seed 3 --config config/cars_reweight_ProxyNCA_prob_orig.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_prob_orig_confusion_139 \
--helpful Influential_data/cars_ProxyNCA_prob_orig_helpful_testcls5.npy \
--harmful Influential_data/cars_ProxyNCA_prob_orig_harmful_testcls5.npy \
--model_dir models/dvi_data_cars_3_lossProxyNCA_prob_orig/ResNet_512_Model \
--helpful_weight 2 --harmful_weight 0 \
--seed 3 --config config/cars_reweight_ProxyNCA_prob_orig.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_prob_orig_confusion_166 \
--helpful Influential_data/cars_ProxyNCA_prob_orig_helpful_testcls6.npy \
--harmful Influential_data/cars_ProxyNCA_prob_orig_harmful_testcls6.npy \
--model_dir models/dvi_data_cars_3_lossProxyNCA_prob_orig/ResNet_512_Model \
--helpful_weight 2 --harmful_weight 0 \
--seed 3 --config config/cars_reweight_ProxyNCA_prob_orig.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_prob_orig_confusion_176 \
--helpful Influential_data/cars_ProxyNCA_prob_orig_helpful_testcls7.npy \
--harmful Influential_data/cars_ProxyNCA_prob_orig_harmful_testcls7.npy \
--model_dir models/dvi_data_cars_3_lossProxyNCA_prob_orig/ResNet_512_Model \
--helpful_weight 2 --harmful_weight 0 \
--seed 3 --config config/cars_reweight_ProxyNCA_prob_orig.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_prob_orig_confusion_186 \
--helpful Influential_data/cars_ProxyNCA_prob_orig_helpful_testcls8.npy \
--harmful Influential_data/cars_ProxyNCA_prob_orig_harmful_testcls8.npy \
--model_dir models/dvi_data_cars_3_lossProxyNCA_prob_orig/ResNet_512_Model \
--helpful_weight 2 --harmful_weight 0 \
--seed 3 --config config/cars_reweight_ProxyNCA_prob_orig.json

python train_sample_reweight.py --dataset cars \
--loss-type ProxyNCA_prob_orig_confusion_177 \
--helpful Influential_data/cars_ProxyNCA_prob_orig_helpful_testcls9.npy \
--harmful Influential_data/cars_ProxyNCA_prob_orig_harmful_testcls9.npy \
--model_dir models/dvi_data_cars_3_lossProxyNCA_prob_orig/ResNet_512_Model \
--helpful_weight 2 --harmful_weight 0 \
--seed 3 --config config/cars_reweight_ProxyNCA_prob_orig.json
