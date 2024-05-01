# import torch
# print(torch.__version__)
# import mmcv
# print(mmcv.__version__)
# import mmseg
# print(mmseg.__version__)
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import cv2
import shutil
import glob
from tqdm import tqdm
from uuid import uuid1
from PIL import Image
import numpy as np
from argparse import ArgumentParser

from mmseg.apis import inference_segmentor

import mmcv
from mmcv.runner import load_checkpoint

from mmseg.models import build_segmentor


def init_segmentor(config, checkpoint=None, device='cuda:0'):
    """Initialize a segmentor from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], [0, 11, 123], \
                         [118, 20, 12], [122, 81, 25], [241, 134, 51], [120, 240, 90], [244, 20, 57]]
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def extract_frames_from_video(video_path, dir_path):

    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    for i in tqdm(range(frame_count), desc="Extract Frames"):
        ret, frame = video.read()
        if not ret:
            break
        filename = os.path.join(dir_path, str(i).zfill(4) + '.png')
        cv2.imwrite(filename, frame)

    video.release()

def synthesize_video_from_frames(dir_path):

    uuid = str(uuid1())
    output_video = f"./results/{uuid}.mp4"
    cmd = f'ffmpeg -r 60 -i {dir_path}/%04d.png -b:v 3M {output_video}'
    os.system(cmd)

    return output_video

def video_inference(video_path, model_path):
    palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], \
               [0, 11, 123], [118, 20, 12], [122, 81, 25], [241, 134, 51], [120, 240, 90], [244, 20, 57]]

    input_tmp_path = './input_tmp'
    if os.path.exists(input_tmp_path):
        shutil.rmtree(input_tmp_path)
    os.mkdir(input_tmp_path)

    output_tmp_path = './output_tmp'
    if os.path.exists(output_tmp_path):
        shutil.rmtree(output_tmp_path)
    os.mkdir(output_tmp_path)

    extract_frames_from_video(video_path, input_tmp_path)

    frames = glob.glob(os.path.join(input_tmp_path, "*.png"))
    frames.sort()

    for i, frame in tqdm(enumerate(frames), total=len(frames), desc="Video Synthesis"):
        model = init_segmentor("./config.py", model_path, device='cuda:0')
        seg_map = inference_segmentor(model, frame)[0].astype('uint8')
        seg_img = Image.fromarray(seg_map).convert('P')
        seg_img.putpalette(np.array(palette, dtype=np.uint8))
        seg_img.save(os.path.join(output_tmp_path, str(i).zfill(4) + '.png'))

    output_video = synthesize_video_from_frames(output_tmp_path)

    """shutil.rmtree(input_tmp_path)
    shutil.rmtree(output_tmp_path)"""

    print(f"The segmented video has been saved at {output_video}")

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--input_video", default='./examples/demo.mp4', help="input video path")
    parser.add_argument("--model", default='./checkpoints/demo.pth', help="model path")

    args = parser.parse_args()

    video_inference(args.input_video, args.model)
