import os
import numpy as np
import pdb
from scipy import misc
import argparse
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian

from tensorpack import *

from train import Model
from cfgs.config import cfg
from img_helper import imgSegmentation, imgMerge

def newPredict_one(img_path, predict_func, output_path, crf):
    img_name = img_path.split('/')
    img_name = img_name[:img_path.index('.jpg')]
    img = misc.imread(img_path, mode = 'L')
    imgs = imgSegmentation(img)
    h,w,_,__ = imgs.shape
    prediction_all = []
    os.mkdir('%s/%s/'%(output_path, img_name))
    for i in range(h):
        prediction_line = []
        for j in range(w):
            prediction = predict_one(imgs[i][j], predict_func, output_path, crf)
            misc.imsave('%s/%s/%d-%d.jpg'%(output_path, img_name,i,j),prediction)
            prediction_line.append(prediction)
        prediction_all.append(prediction_line)
    prediction = imgMerge(np.array(prediction_all))
    misc.imsave('%s/%s/总.jpg'%(output_path, img_name),prediction)

def predict_one(img, predict_func, output_path, crf):
    img = np.expand_dims(img, axis=3)
    batch_img = np.expand_dims(img, axis=0)
    predictions = predict_func([batch_img])[0]

    predictions = np.reshape(predictions, (img.shape[0], img.shape[1], cfg.class_num))

    # CRF
    if crf is True:
        d = dcrf.DenseCRF2D(w, h, cfg.class_num)

        # set unary potential
        predictions = np.transpose(predictions, (2, 0, 1))
        U = unary_from_softmax(predictions)
        d.setUnaryEnergy(U)

        # set pairwise potential
        # This creates the color-independent features and then add them to the CRF
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)
        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(8, 8), srgb=(13, 13, 13), rgbim=img,
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)

        iter_num = 5
        result = np.argmax(d.inference(iter_num), axis=0)
        result = np.reshape(result, (img.shape[0], img.shape[1]))
    else:
        result = np.argmax(predictions, axis=2)
    result = (1 - result) * 255
    mask = np.zeros(img.shape)
    mask[:,:,0] = result
    output = img * 0.7 + mask * 0.3
    output = np.resize(output, [output.shape[0], output.shape[1]])
    return output
    

def predict(args):
    sess_init = SaverRestore(args.model)
    model = Model()
    predict_config = PredictConfig(session_init=sess_init,
                                   model=model,
                                   input_names=["input"],
                                   output_names=["softmax_output"])

    predict_func = OfflinePredictor(predict_config)

    if os.path.isfile(args.input):
        # input is a file
        newPredict_one(args.input, predict_func, args.output or "output", args.crf)

    if os.path.isdir(args.input):
        # input is a directory
        output_dir = args.output or "output"
        if os.path.isdir(output_dir) == False:
            os.makedirs(output_dir)
        for (dirpath, dirnames, filenames) in os.walk(args.input):
            logger.info("Number of images to predict is " + str(len(filenames)) + ".")
            for file_idx, filename in enumerate(filenames):
                if file_idx % 10 == 0 and file_idx > 0:
                    logger.info(str(file_idx) + "/" + str(len(filenames)))
                filepath = os.path.join(args.input, filename)
                newPredict_one(filepath, predict_func, output_dir, args.crf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='path to the model file', required=True)
    parser.add_argument('--input', help='path to the input image', required=True)
    parser.add_argument('--output', help='path to the output image/dir')
    parser.add_argument('--crf', action='store_true')


    args = parser.parse_args()
    predict(args)
