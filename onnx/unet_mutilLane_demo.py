import onnxruntime as ort
import cv2
import numpy as np

color_list = [(100, 149, 237), (0, 0, 255), (173, 255, 47), (240, 255, 255), (0, 100, 0),
              (47, 79, 79), (255, 228, 196), (138, 43, 226), (165, 42, 42), (222, 184, 135)]

lane_Id_type = [7, 5, 3, 1, 2, 4, 6, 8]

'''
line_type = ['无车道线', '单条白色实线', '单条白色虚线', '单条黄色实线', '单条黄色虚线', '双条白色实线', '双条黄色实线',
             '双条黄色虚线', '双条白色黄色实线', '双条白色虚实线', '双条白色实虚线']
'''

line_type = ['No lane markings',
             'Single white solid line',
             'Single white dashed line',
             'Single solid yellow line',
             'Single yellow dashed line',
             'Double solid white lines',
             'Double solid yellow lines',
             'Double yellow dashed lines',
             'Double white yellow solid lines',
             'Double white dashed lines',
             'Double white solid dashed lines']


def precess_image(img_src, resize_w, resize_h):
    image = cv2.resize(img_src, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image /= 255
    return image


def softmax(x, axis):
    x -= np.max(x, axis=axis, keepdims=True)
    value = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return value


def test_image(img_path):
    input_w = 640
    input_h = 480

    orig_img = cv2.imread(img_path)
    img_h, img_w = orig_img.shape[0:2]
    image = precess_image(orig_img, input_w, input_h)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)

    ort_session = ort.InferenceSession('./UnetLane_mutilLane_ZQ.onnx')
    output = (ort_session.run(None, {'data': image}))

    seg_output = softmax(output[0], axis=1)[0]
    cls_output = softmax(output[1], axis=2)[0]

    cls_output = np.argmax(cls_output, axis=1)
    mask = np.zeros(shape=(input_h, input_w, 3))

    lane_id = []
    write_pos = []

    for i in range(mask.shape[0] - 1, 0, -1):
        for j in range(mask.shape[1] - 1, 0, -1):
            max_index = np.argmax(seg_output[:, i, j])
            if max_index not in lane_id:
                lane_id.append(max_index)
                if i > input_h - 20 or j > input_w - 20:
                    write_pos.append([j - 20, i - 20])
                else:
                    write_pos.append([j, i])
            if max_index != 0 and seg_output[max_index, i, j] > 0.5:
                mask[i, j, :] = color_list[max_index]

    mask = cv2.resize(mask, (img_w, img_h))

    for i in range(len(lane_id)):
        if lane_id[i] == 0:
            continue

        lane_type = cls_output[lane_Id_type.index(lane_id[i])]

        px = int(write_pos[i][0] / input_w * img_w)
        py = int(write_pos[i][1] / input_h * img_h)

        cv2.putText(orig_img, str(lane_id[i]), (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(orig_img, str(lane_type), (px, py + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.putText(orig_img, 'lane_id: 7-5-3-1-2-4-6-8', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(orig_img, 'line type:', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
    for i in range(len(line_type)):
        cv2.putText(orig_img, str(i) + ': ' + str(line_type[i]), (10, 80 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 0), 2, cv2.LINE_AA)

    opencv_image = np.clip(np.array(orig_img) + np.array(mask) * 0.4, a_min=0, a_max=255)
    opencv_image = opencv_image.astype("uint8")
    cv2.imwrite('./test_result.jpg', opencv_image)


if __name__ == "__main__":
    print('This is main ...')
    img_path = './test.jpg'
    test_image(img_path)
