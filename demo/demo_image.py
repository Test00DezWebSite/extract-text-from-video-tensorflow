from tensorpack import *
import numpy as np
from scipy import misc
import cv2
import shutil
import os

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian

# import models
from lineSeg.train import Model as lineSeg_Model
from SeqRecog.train import Model as SeqRecog_Model
from lineSeg.cfgs.config import cfg as lineSeg_model_cfg
from SeqRecog.cfgs.config import cfg as SeqRecog_model_cfg
from SeqRecog.mapper import Mapper

import utils

def lineSeg_predict(img, predict_func):
    """use lineSeg model to predict the mask from an image

    # Arguments
        img: given image
        predict_func: predict functions

    # Returns
        mask: the predict mask of given image

    """
    # slice the image
    slices = utils.slice_even(img, (200,200))
    h_cnt, w_cnt, h, w = slices.shape
    
    # get the predictions, with the shape of (h, w, 2)
    # expand 2 dims, axis 0 is the batch dim, axis 3 is the channel dim
    predictions = [predict_func([np.reshape(j, (1, j.shape[0], j.shape[1], 1))]) for i in slices for j in i ]
    predictions = np.reshape(predictions, (h_cnt, w_cnt, h, w, 2))

    # merge predictions
    predictions = utils.merge(predictions,0,0,200,200)
    predictions = predictions[:img.shape[0],:img.shape[1]]

    # CRF
    w, h, c = predictions.shape
    d = dcrf.DenseCRF2D(w, h, c)

    # set unary potential
    predictions = np.transpose(predictions, (2, 0, 1))
    U = unary_from_softmax(predictions)
    d.setUnaryEnergy(U)

    # set pairwise potential
    # This creates the color-independent features and then add them to the CRF
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    iter_num = 5
    result = np.argmax(d.inference(iter_num), axis=0)
    result = np.reshape(result, (img.shape[0],img.shape[1]))

    # return the mask
    result = (1 - result) * 255
    result = np.uint8(result)
    misc.imsave('result.jpg', result)
    return result

def SeqRecog_predict(img, predict_func):
    """According to given image, return a predicted string.

    # Arguments

    # Returns
        text: string of a line
    """
    if img.shape[0] != SeqRecog_model_cfg.input_height:
        if SeqRecog_model_cfg.input_width != None:
            img = cv2.resize(img, (SeqRecog_model_cfg.input_width, SeqRecog_model_cfg.input_height))
        else:
            scale = SeqRecog_model_cfg.input_height / img.shape[0]
            img = cv2.resize(img, None, fx=scale, fy=scale)

    seqlen = img.shape[1]
    img = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
    predictions = predict_func([img, [seqlen]])[0]

    mapper = Mapper()
    result = mapper.decode_output(predictions[0])
    return result

def img2txt(img):
    # build lineSeg predict function
    lineSeg_sess_init = SaverRestore('models/lineSeg')
    lineSeg_model = lineSeg_Model()
    lineSeg_predict_config = PredictConfig(session_init = lineSeg_sess_init,
                                        model = lineSeg_model,
                                        input_names = ['input'],
                                        output_names = ['softmax_output'])
    lineSeg_predict_func = OfflinePredictor(lineSeg_predict_config)

    # build SeqRecog predict function
    SeqRecog_sess_init = SaverRestore('models/SeqRecog')
    SeqRecog_model = SeqRecog_Model()
    SeqRecog_predict_config = PredictConfig(session_init = SeqRecog_sess_init,
                                        model = SeqRecog_model,
                                        input_names = ['feat', 'seqlen'],
                                        output_names = ['prediction'])
    SeqRecog_predict_func = OfflinePredictor(SeqRecog_predict_config)

    
    mask = lineSeg_predict(img, lineSeg_predict_func)
    lines_img = utils.lineSeg(img, mask)
    lines_txt = [SeqRecog_predict(i, SeqRecog_predict_func) for i in lines_img]

    output_dir = 'output'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    misc.imsave(os.path.join(output_dir,'mask.jpg'),mask)
    for i in range(len(lines_img)):
        misc.imsave(os.path.join(output_dir,'line-%d.jpg'%i),lines_img[i])
        f = open(os.path.join(output_dir,'line-%d.txt'%i),'w')
        f.write(lines_txt[i])
        f.close()
    f = open(os.path.join(output_dir,'all.txt'),'w')
    f.write('\n'.join(lines_txt))
    f.close()
    # merge all lines to one paragraph
    return '\n'.join(lines_txt)

if __name__ == '__main__':
    img = misc.imread('../1.jpg', mode = 'L')
    text = img2txt(img)
