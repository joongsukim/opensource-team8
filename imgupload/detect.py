import time
from absl import app, flags, logging
import cv2
import numpy as np
import tensorflow as tf
from .yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from .yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from .yolov3_tf2.utils import draw_outputs


def detect(input_jpg, output):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if True:
        yolo = YoloV3Tiny(classes=3)
    else:
        yolo = YoloV3(classes=3)

    yolo.load_weights('./imgupload/weights/yolov3-dog.tf').expect_partial()
    print('weights loaded')

    #label
    class_names = [c.strip() for c in open('./imgupload/data/labels/dogs.names').readlines()]
    print('classes loaded')
    raw_images = []
    images = input_jpg
    img_raw = tf.image.decode_image(
        open(images, 'rb').read(), channels=3)
    raw_images.append(img_raw)
    num = 0
    for raw_img in raw_images:
        num += 1
        img = tf.expand_dims(raw_img, 0)
        img = transform_images(img, 416)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        logging.info('time: {}'.format(t2 - t1))

        print('detections:')
        for i in range(nums[0]):
            print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                        np.array(scores[0][i]),
                                        np.array(boxes[0][i])))
            class_process = class_names[int(classes[0][i])]

        img = cv2.cvtColor(raw_img.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imwrite(output + 'detection' + str(num) + '.jpg', img)
        print('output saved to: {}'.format(output + 'detection' + str(num) + '.jpg'))

        return class_process
