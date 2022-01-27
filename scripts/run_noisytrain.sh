python train_noisydata.py --dataset cub_noisy \
--loss-type ProxyNCA_prob_orig_noisy_0.01 \
--mislabel_percentage 0.01 \
--seed 0 --config config/cub.json &
python train_noisydata.py --dataset cub_noisy \
--loss-type ProxyNCA_prob_orig_noisy_0.05 \
--mislabel_percentage 0.05 \
--seed 0 --config config/cub.json &
python train_noisydata.py --dataset cub_noisy \
--loss-type ProxyNCA_prob_orig_noisy_0.1 \
--mislabel_percentage 0.1 \
--seed 0 --config config/cub.json

python train_noisydata.py --dataset cars_noisy \
--loss-type ProxyNCA_prob_orig_noisy_0.01 \
--mislabel_percentage 0.01 \
--seed 3 --config config/cars.json &
python train_noisydata.py --dataset cars_noisy \
--loss-type ProxyNCA_prob_orig_noisy_0.05 \
--mislabel_percentage 0.05 \
--seed 3 --config config/cars.json &
python train_noisydata.py --dataset cars_noisy \
--loss-type ProxyNCA_prob_orig_noisy_0.1 \
--mislabel_percentage 0.1 \
--seed 3 --config config/cars.json

python train_noisydata.py --dataset inshop_noisy \
--loss-type ProxyNCA_prob_orig_noisy_0.01 \
--mislabel_percentage 0.01 \
--seed 4 --config config/inshop.json

python train_noisydata.py --dataset inshop_noisy \
--loss-type ProxyNCA_prob_orig_noisy_0.05 \
--mislabel_percentage 0.05 \
--seed 4 --config config/inshop.json

python train_noisydata.py --dataset inshop_noisy \
--loss-type ProxyNCA_prob_orig_noisy_0.1 \
--mislabel_percentage 0.1 \
--seed 4 --config config/inshop.json