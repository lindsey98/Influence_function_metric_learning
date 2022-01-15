from Influence_function.influential_sample import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
        dataset_name = 'sop'
        loss_type = 'ProxyNCA_prob_orig'
        config_name = 'sop'
        sz_embedding = 512
        seed = 0
        epoch = 40
        test_crop = False

        # dataset_name = 'sop'
        # loss_type = 'ProxyNCA_pfix_var'
        # config_name = 'sop'
        # sz_embedding = 512
        # seed = 2
        # epoch = 40
        # test_crop = False

        # dataset_name = 'inshop'
        # loss_type = 'ProxyNCA_pfix_var_complicate'
        # config_name = 'inshop'
        # sz_embedding = 512
        # seed = 3
        # epoch = 40
        # test_crop = False

        IS = InfluentialSample(dataset_name, seed, loss_type, config_name, test_crop, sz_embedding, epoch)
        confusion_class_pairs = IS.get_confusion_class_pairs()
