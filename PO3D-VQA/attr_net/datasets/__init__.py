from torch.utils.data import DataLoader, ConcatDataset
from .superclevr_object import SuperClevrObjectDataset
from .clevr_object import ClevrObjectDataset




def get_dataset(opt, split):
    print(opt.dataset, opt.multiple_dataset)
    # def __init__(self, img_dir, obj_ann_path, scene_path, type = 'part', split = 'train', 
    # os.path.join('/mnt/data0/xingrui/superclevr_anno/', 'superclevr_anno.json')
    # /mnt/data0/xingrui/superclevr_anno/superclevr_anno.json
    
    # dataset = SuperClevrObjectDataset('/home/xingrui/vqa/NSCL_super/data/superclevr/0k55k/images', 
    #                                 None,
    #                                 '/home/xingrui/vqa/NSCL_super/data/superclevr/0k55k/scenes.json')
    if opt.dataset == 'superclevr':
        if not opt.multiple_dataset:
            print(opt.img_dir)
            print(opt.obj_ann_path)
            ds_original = SuperClevrObjectDataset(opt.img_dir, opt.obj_ann_path, opt.scene_path, opt.type, split, bbox_mode = opt.bbox_mode)

            # if opt.use_aug_ds and split == 'train':
            #     obj_ann_path = "/home/xingrui/vqa/ns-vqa/data/ver_mask/attr_net/superclevr_anno.json"
            #     img_dir = "/mnt/data0/xingrui/ccvl17/ver_mask/images"
            #     scene_path = "/mnt/data0/xingrui/ccvl17/ver_mask/superCLEVR_scenes.json"
            #     ds_aug =SuperClevrObjectDataset(img_dir, obj_ann_path, scene_path, opt.type, split, trim = 0.3, aug_level="hard")
            #     ds = ConcatDataset([ds_original, ds_aug])
            # else:
            #     ds = ds_original
                # ds = ConcatDataset([ds_original])
            ds = ds_original
            print("Length of dataset", len(ds))
            
        elif opt.multiple_dataset:
            dataset_list = ['ver_nopart', 'ver_mask', 'ver_texture']
            dataset_all = []
            for d in dataset_list:
                obj_ann_path = "/home/xingrui/vqa/ns-vqa/data/{}/attr_net/superclevr_anno.json".format(d)
                img_dir = "/mnt/data0/xingrui/ccvl17/{}/images".format(d)
                scene_path = "/mnt/data0/xingrui/ccvl17/{}/superCLEVR_scenes.json".format(d)
                
                ds = SuperClevrObjectDataset(img_dir, obj_ann_path, scene_path, opt.type, split, trim = 0.6)
                print("Length of dataset", len(ds))
                dataset_all.append(ds)
            ds = ConcatDataset(dataset_all)
            print("Length of dataset", len(ds))
    elif opt.dataset == 'clevr':
        ds = ClevrObjectDataset(opt.img_dir, opt.obj_ann_path, opt.scene_path, split, bbox_mode = opt.bbox_mode)
        print("len of data ", split, len( ds))
    else:
        raise ValueError('Invalid datsaet %s' % opt.dataset)
    return ds


def get_dataloader(opt, split):
    ds = get_dataset(opt, split)
    loader = DataLoader(dataset=ds, batch_size=opt.batch_size, shuffle=opt.shuffle_data)
    return loader