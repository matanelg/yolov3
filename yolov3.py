import json
import os
import time
import cv2
import numpy as np
import keras.backend as K
from keras.models import load_model


class YOLO:
    def __init__(self, obj_threshold, nms_threshold):
        self._t1 = obj_threshold
        self._t2 = nms_threshold
        self._yolo = load_model("C:/new_computer/test/yolo.h5")

    def _process_feats(self, out, anchors, mask):
        grid_h, grid_w, num_boxes = map(int, out.shape[1: 4])

        anchors = [anchors[i] for i in mask]
        # Reshape to batch, height, width, num_anchors, box_params.
        anchors_tensor = K.reshape(K.variable(anchors),
                                   [1, 1, len(anchors), 2])
        out = out[0]
        box_xy = K.get_value(K.sigmoid(out[..., :2]))
        box_wh = K.get_value(K.exp(out[..., 2:4]) * anchors_tensor)
        box_confidence = K.get_value(K.sigmoid(out[..., 4]))
        box_confidence = np.expand_dims(box_confidence, axis=-1)
        box_class_probs = K.get_value(K.sigmoid(out[..., 5:]))
        col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)
        box_xy += grid
        box_xy /= (grid_w, grid_h)
        box_wh /= (416, 416)
        box_xy -= (box_wh / 2.)
        boxes = np.concatenate((box_xy, box_wh), axis=-1)
        return boxes, box_confidence, box_class_probs

    def _filter_boxes(self, boxes, box_confidences, box_class_probs):
        box_scores = box_confidences * box_class_probs
        box_classes = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)
        pos = np.where(box_class_scores >= self._t1)
        boxes = boxes[pos]
        classes = box_classes[pos]
        scores = box_class_scores[pos]
        return boxes, classes, scores

    def _nms_boxes(self, boxes, scores):
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        areas = w * h
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
            w1 = np.maximum(0.0, xx2 - xx1 + 1)
            h1 = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w1 * h1
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self._t2)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep

    def _yolo_out(self, outs, shape):
        masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                   [59, 119], [116, 90], [156, 198], [373, 326]]
        boxes, classes, scores = [], [], []
        for out, mask in zip(outs, masks):
            b, c, s = self._process_feats(out, anchors, mask)
            b, c, s = self._filter_boxes(b, c, s)
            boxes.append(b)
            classes.append(c)
            scores.append(s)
        boxes = np.concatenate(boxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)
        # Scale boxes back to original image shape.
        width, height = shape[1], shape[0]
        image_dims = [width, height, width, height]
        boxes = boxes * image_dims
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = self._nms_boxes(b, s)
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])
        if not nclasses and not nscores:
            return None, None, None
        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)
        return boxes, classes, scores

    def predict(self, image, shape):
        outs = self._yolo.predict(image)
        boxes, classes, scores = self._yolo_out(outs, shape)
        return boxes, classes, scores


def process_image(img):
    image = cv2.resize(img, (416, 416),interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)
    return image



def draw(image, box):
    x, y, w, h = box
    top = max(0, np.floor(x + 0.5).astype(int))
    left = max(0, np.floor(y + 0.5).astype(int))
    right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
    bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))
    cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
    return image


yolo = YOLO(0.7, 0.5)

d={}
cap = cv2.VideoCapture("C:/new_computer/test7/West_Side.mp4")
frames_original_video = cap.get(cv2.CAP_PROP_FRAME_COUNT) #301
original_fps = cap.get(cv2.CAP_PROP_FPS) #30
i=0;
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        pimage = process_image(frame)
        boxes, classes, scores = yolo.predict(pimage, frame.shape)
        #print(classes)
        if classes is None:
            d[f'frame{i:03}']=[[0,0,0,0]]
            #print(f"frame {i:03} :   Box = None")
        elif 2 not in classes:
            d[f'frame{i:03}']=[[0,0,0,0]]
            #print(f"frame {i:03} :   Box = None")
        else:
            box_list=[]
            for box,obj in zip(boxes,classes):
                if obj == 2: # car in coco dataset
                    frame = draw(frame,box)
                    box_list.append(list(box))
                    #print(f"frame {i:03} :  {box}")
            d[f'frame{i:03}']=box_list
        cv2.imwrite(f"C:/new_computer/test7/pic{i:03}.jpg", frame) 
        i=i+1
    else:
        break     
cap.release()
cv2.destroyAllWindows()

j=json.dumps(d)
with open('C:/new_computer/test5/d.json','w') as f:
    f.write(j)
    f.close()

# t=json.load(open('C:/new_computer/test5/d.json'))
# print(t)