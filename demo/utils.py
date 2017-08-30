from scipy import misc
import cv2
import numpy as np
import os

def sort(imgs):
    """sort images

    # Arguments
        imgs: `[img, x, y, w, h]`

    # Returns
        res_imgs: sorted images
    """
    imgs.sort(key=lambda x:x[2])
    for i in range(len(imgs)-1):
        # find overlaps, sort by x
        j = 0
        while i+j < len(imgs)-1 and imgs[i+j][2] + imgs[i+j][4] > imgs[i+j+1][2]:
            j += 1
        if j > 0:
            imgs[i:i+j+1] = sorted(imgs[i:i+j+1], key=lambda x:x[1])
    res_imgs = [i[0] for i in imgs]

    return res_imgs

def lineSeg(img, mask):
    # erode the mask
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.erode(mask,kernel,iterations = 1)
    # find all lists of contours
    im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # for each list of contours, align all midpoints of columns
    all_pieces = []
    for i in range(len(contours)):
        # find the largest rectangle, which is the final output
        x,y,w,h = cv2.boundingRect(contours[i])
        if y > img.shape[0] or x > img.shape[1]:
            continue
        if w < 20 or h < 10:
            continue
        h = min(h, img.shape[0]-y)
        w = min(w, img.shape[1]-x)

        # crop images
        output_img = img[y:y+h, x:x+w]
        output_mask = mask[y:y+h, x:x+w]
        # collect all points inside the coutour
        all_points = []
        maxLength = 0
        for ix in range(w):
            # get all points of a column
            line = []
            for iy in range(h):
                if cv2.pointPolygonTest(contours[i],(ix+x,iy+y),False) >= 0:
                    line.append(output_img[iy][ix])
            line = np.array(line)
            # length of pixels
            maxLength = max(maxLength, line.shape[0])
            all_points.append(line)

        res = []
        # pad every column to fixed size
        if len(img.shape) == 3:
            for j in all_points:
                pad_prev = (maxLength - j.shape[0]) // 2
                pad_post = maxLength - pad_prev - j.shape[0]
                pad_prev = np.reshape(np.array([255]*pad_prev*3),(pad_prev,3))
                pad_post = np.reshape(np.array([255]*pad_post*3),(pad_post,3))
                line = np.vstack((pad_prev, j, pad_post))
                res.append(line)
            all_points = np.array(res)
            all_points = np.transpose(all_points,(1,0,2))
        else:
            for j in all_points:
                pad_prev = (maxLength - j.shape[0]) // 2
                pad_post = maxLength - pad_prev - j.shape[0]
                pad_prev = np.array([255]*pad_prev)
                pad_post = np.array([255]*pad_post)
                line = np.concatenate((pad_prev, j, pad_post),axis=0)
                res.append(line)
            all_points = np.array(res)
            all_points = np.transpose(all_points)
        all_pieces.append([all_points,x,y,w,h])

    # save all pieces in order
    imgs = sort(all_pieces)

    return imgs

def slice_even(img, shape):
    #h, w = shape
    if len(img.shape) == 3:
        h, w, _ = img.shape
    else:
        h, w = img.shape
    h_gap = 200
    w_gap = 200
    h_cnt = h / h_gap
    w_cnt = w / w_gap
    # 在高度上padding
    if h_cnt != int(h_cnt):
        padding_matrix = [255] * getPaddingNum(h,h_gap) * w
        padding_matrix = np.array(padding_matrix)
        padding_matrix = np.reshape(padding_matrix,(getPaddingNum(h,h_gap),w))
        img = np.vstack((img,padding_matrix))
        h = img.shape[0]
        h_cnt = int(h_cnt)+1
    # 在宽度上padding
    if w_cnt != int(w_cnt):
        padding_matrix = [255] * h * getPaddingNum(w,w_gap)
        padding_matrix = np.array(padding_matrix)
        padding_matrix = np.reshape(padding_matrix,([h,getPaddingNum(w,w_gap)]))
        img = np.concatenate((img,padding_matrix),1)
        w = img.shape[1]
        w_cnt = int(w_cnt)+1
    i = 0
    imgs = []
    while(i < h):
        h_next = min(i+h_gap, h)
        j = 0
        while(j < w):
            w_next = min(j+w_gap, w)
            imgs.append(img[i:h_next,j:w_next])
            j = w_next
        i = h_next
    imgs = np.reshape(imgs, (h_cnt,w_cnt,h_gap,w_gap))
    return np.array(imgs)

def merge(predictions, x, y, w, h):
    rows = [np.concatenate(i[:,y:y+h,x:x+w],axis = 1) for i in predictions]
    return np.vstack(rows)

def getPaddingNum(total,gap):
    div = total / gap
    if div == int(div):
        return 0
    else:
        return (int(div) + 1) * gap - total

if __name__ == '__main__':
    img = misc.imread('img.jpg', mode = 'L')
    mask = misc.imread('result.jpg', mode = 'L')
    res = lineSeg(img, mask)