import cv2

from tensorpack import *
import numpy as np

# list split utils
from operator import itemgetter
from itertools import *

# import models
from classify_frames.train import Model as Model_classify_frames
from detect_text_area.train import Model as Model_detect_text_area
from segment_lines.train import Model as Model_segment_lines
from recognize_sequences.train import Model as Model_recognize_sequences

# import configs
from classify_frames.cfgs.config import cfg as cfg_classify_frames
from detect_text_area.cfgs.config import cfg as cfg_detect_text_area
from segment_lines.cfgs.config import cfg as cfg_segment_lines
from recognize_sequences.cfgs.config import cfg as cfg_recognize_sequences

# import utils
from recognize_sequences.mapper import Mapper


def predict_classify_frames(video_path, predictor):
    """use lineSeg model to predict the mask from an image

    # Arguments
        img: given image
        predictor: predict functions

    # Returns
        mask: the predict mask of given image

    """
    preds = predictor()
    pass

def predict_detect_text_area(predictor):
    """use lineSeg model to predict the mask from an image

    # Arguments
        img: given image
        predictor: predict functions

    # Returns
        mask: the predict mask of given image

    """
    preds = predictor()
    pass
def predict_segment_lines(predictor):
    """use lineSeg model to predict the mask from an image

    # Arguments
        img: given image
        predictor: predict functions

    # Returns
        mask: the predict mask of given image

    """
    preds = predictor()
    pass
def predict_recognize_sequences(predictor):
    """use lineSeg model to predict the mask from an image

    # Arguments
        img: given image
        predictor: predict functions

    # Returns
        mask: the predict mask of given image

    """
    preds = predictor()
    pass

def extract_frames(cap):
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(total_frame):
        _, frame = cap.read()
        frames.append(frame)

    return frames

def generate_tensors(frames):
    tensors = []
    total_frame = len(frames)
    margin = len(cfg_classify_frames.frame_extract_pattern) // 2
    for frame_idx in range(total_frame):
        if frame_idx - margin < 0 or frame_idx + margin >= total_frame:
            continue
        selected_frames = frames[frame_idx - margin:frame_idx + margin]
        tensor = np.asarray(selected_frames)
        tensor = tensor.swapaxes(0,2)

        tensors.append(tensor)

    return tensors

def predict(video_path):
    '''
    # Load weights
    weights_classify_frames = SaverRestore('models/classify_frames')
    weights_detect_text_area = SaverRestore('models/detect_text_area')
    weights_segment_lines = SaverRestore('models/segment_lines')
    weights_recognize_sequences = SaverRestore('models/recognize_sequences')

    # Build graphs
    model_classify_frames = Model_classify_frames()
    model_detect_text_area = Model_detect_text_area()
    model_segment_lines = Model_segment_lines()
    model_recognize_sequences = Model_recognize_sequences()

    # Build predict configs
    config_classify_frames = PredictConfig(session_init = weights_classify_frames, model = model_classify_frames, input_names = ['feat', 'seqlen'], output_names = ['prediction'])
    config_detect_text_area = PredictConfig(session_init = weights_detect_text_area, model = model_detect_text_area, input_names = ['feat', 'seqlen'], output_names = ['prediction'])
    config_segment_lines = PredictConfig(session_init = weights_segment_lines, model = model_segment_lines, input_names = ['feat', 'seqlen'], output_names = ['prediction'])
    config_recognize_sequences = PredictConfig(session_init = weights_recognize_sequences, model = model_recognize_sequences, input_names = ['feat', 'seqlen'], output_names = ['prediction'])

    # Build predictors
    predictor_classify_frames = OfflinePredictor(config_classify_frames)
    predictor_detect_text_area = OfflinePredictor(config_detect_text_area)
    predictor_segment_lines = OfflinePredictor(config_segment_lines)
    predictor_recognize_sequences = OfflinePredictor(config_recognize_sequences)
    '''

    # predict all effective frames from a video

    # for each sequence of frames, choose the frame of max blurry

    # for each chosen frame, detect the text area and crop

    # for cropped frame, segment line_area and out_line_area

    # video -> (224, 224, c) tensors in cfg.frame_extract_pattern

    #
    cap = cv2.VideoCapture(video_path)
    # extract all frames
    width, height = int(cap.get(3)), int(cap.get(4))
    # output_final = cv2.VideoWriter('demo_output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (width,height))
    frames = extract_frames(cap)
    print(len(frames))
    # get resized gray-level frames
    resized_frames = [cv2.cvtColor(cv2.resize(i, (224, 224)), cv2.COLOR_BGR2GRAY) for i in frames]
    print(len(resized_frames))

    # generate tensors in shape of (224, 224, c)
    tensors_classify_frames = generate_tensors(resized_frames)
    print(len(tensors_classify_frames))
    # batch tensors
    return
    # predict

    # list all preds and split
    preds = []
    frame_idx = [idx for idx, lb in enumerate(preds) if lb == '1']
    frame_idxss = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(frame_idx), lambda x: x[0]-x[1])]


    # output_classify_frames = predict_classify_frames()
    # output_detect_text_area = predict_detect_text_area()
    # output_segment_lines = predict_segment_lines()
    # output_recognize_sequences = predict_recognize_sequences()

    return 

if __name__ == '__main__':
    predict('demo_input.mp4')