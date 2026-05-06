import asyncio
from argparse import ArgumentParser

from mmdet.apis import (inference_detector,
                        init_detector)
from mmdet.registry import VISUALIZERS
import mmcv

import sys
sys.path.insert(0, '/home/dsdl-4090/Documents/dsdl_ssd/MMDET_workspace/Backbone')
sys.path.insert(0, '/home/dsdl-4090/Documents/dsdl_ssd/MMDET_workspace/FPN')
import ravit, microvit, microvitv2, parformer, efficient_fpn, rep_fpn

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.3,
        help='bbox score threshold')
    # parser.add_argument(
    #     '--async-test',
    #     action='store_true',
    #     help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device, )
    # test a single image
    image = mmcv.imread(args.img)
    result = inference_detector(model, image)
    # show the results
    # show_result_pyplot(model, args.img, result, score_thr=args.score_thr)
    # model.show_result(
    #     args.img,
    #     result,
    #     score_thr=args.score_thr,
    #     show=False,
    #     out_file=args.img+'_result.jpg')

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    # show the results
    visualizer.add_datasample(
        'result',
        image,
        data_sample=result,
        draw_gt=False,
        wait_time=0,
        out_file='./outputs_img/'+args.img+'_result.png' # optionally, write to output file
    )
    # visualizer.show()

if __name__ == '__main__':
    args = parse_args()
    # if args.async_test:
    #     asyncio.run(async_main(args))
    # else:
    #     main(args)
    main(args)
