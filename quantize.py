import os
import sys
import argparse
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pytorch_nndct.apis import torch_quantizer, dump_xmodel

from common import *

from models.common import DetectMultiBackend
from models.yolo import Model

DIVIDER = '-----------------------------------------'


def quantize(build_dir, quant_mode, batchsize, dataset_dir):
    dset_dir = build_dir + '/dataset'
    float_model = build_dir + '/float_model'
    quant_model = build_dir + '/quant_model'

    # use GPU if available
    if (torch.cuda.device_count() > 0):
        print('You have', torch.cuda.device_count(), 'CUDA devices available')
        for i in range(torch.cuda.device_count()):
            print(' Device', str(i), ': ', torch.cuda.get_device_name(i))
        print('Selecting device 0..')
        device = torch.device('cuda:0')
    else:
        print('No CUDA devices available..selecting CPU')
        device = torch.device('cpu')

    # load trained model
    model = DetectMultiBackend("./best_2.pt", device=device)

    # force to merge BN with CONV for better quantization accuracy
    optimize = 1

    # override batchsize if in dataset mode
    if (quant_mode == 'dataset'):
        batchsize = 1

    rand_in = torch.randn([batchsize, 3, 640, 640])
    quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=quant_model)
    quantized_model = quantizer.quant_model

    # create a Data Loader
    test_transforms = transforms.Compose([transforms.ToTensor(),
                                          transforms.Resize((640, 640)),
                                          transforms.RandomHorizontalFlip(0.5)])
    test_dataset = MyDataset(dataset_dir, test_transforms)
    print(test_dataset.__len__())
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batchsize,
                                              shuffle=False)

    t_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=1 if quant_mode == 'dataset' else 10,
                                           shuffle=False)

    # create a dataloader
    # https://blog.csdn.net/m0_45287781/article/details/127947918
    # from utils.dataloaders import create_dataloader
    # t_loader = create_dataloader(dataset_dir, imgsz=640, batch_size=3, pad=0.5, stride=32,
    #                                workers=8, prefix='')[0]

    print(t_loader)

    # evaluate
    test(quantized_model, device, t_loader)



    # export config
    if quant_mode == 'calib':
        quantizer.export_quant_config()
    if quant_mode == 'dataset':
        quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)

    return


def run_main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--build_dir', type=str, default='build', help='Path to build folder. Default is build')
    ap.add_argument('-q', '--quant_mode', type=str, default='calib', choices=['calib', 'dataset'],
                    help='Quantization mode (calib or dataset). Default is calib')
    ap.add_argument('-b', '--batchsize', type=int, default=50,
                    help='Testing batchsize - must be an integer. Default is 100')
    ap.add_argument('-s', '--source', type=str, default='dataset',
                    help='Data set directory, when quant_mode=calib, it is for calibration, while quant_mode=dataset it is for evaluation')
    args = ap.parse_args()

    print('\n' + DIVIDER)
    print('PyTorch version : ', torch.__version__)
    print(sys.version)
    print(DIVIDER)
    print(' Command line options:')
    print('--build_dir    : ', args.build_dir)
    print('--quant_mode   : ', args.quant_mode)
    print('--batchsize    : ', args.batchsize)
    print('--source    : ', args.source)
    print(DIVIDER)

    quantize(args.build_dir, args.quant_mode, args.batchsize, args.source)

    return


if __name__ == '__main__':
    run_main()
