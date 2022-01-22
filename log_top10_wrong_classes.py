from Influence_function.influence_function import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
        sz_embedding = 512
        epoch = 40
        test_crop = False
        dataset_name = 'sop'
        config_name = 'sop'
        loss_type = 'SoftTriple'
        seed = 4

        IS = ScalableIF(dataset_name, seed, loss_type, config_name, test_crop, sz_embedding, epoch)
        confusion_class_pairs = IS.get_confusion_class_pairs()
