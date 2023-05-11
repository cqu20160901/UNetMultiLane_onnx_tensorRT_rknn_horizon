import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from math import exp
from math import sqrt

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

color_list = [(100, 149, 237), (0, 0, 255), (173, 255, 47), (240, 255, 255), (0, 100, 0),
              (47, 79, 79), (255, 228, 196), (138, 43, 226), (165, 42, 42), (222, 184, 135)]

lane_Id_type = [7, 5, 3, 1, 2, 4, 6, 8]

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

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def get_engine_from_bin(engine_file_path):
    print('Reading engine from file {}'.format(engine_file_path))
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def preprocess(img_src, resize_w, resize_h):
    image = cv2.resize(img_src, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image /= 255
    image = image.transpose(2, 0, 1)
    image = image.copy()
    return image


def softmax(x, axis):
    x -= np.max(x, axis=axis, keepdims=True)
    value = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return value


def main():
    engine_file_path = './UnetLane_mutilLane_ZQ.trt'
    input_image_path = 'test.jpg'

    input_w = 640
    input_h = 480

    src_img = cv2.imread(input_image_path)

    img_h, img_w = src_img.shape[:2]
    image = preprocess(src_img, input_w, input_h)

    with get_engine_from_bin(engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)

        inputs[0].host = image
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=1)

        cls_output = softmax(trt_outputs[0].reshape(1, 8, 11),  axis=2)[0]
        seg_output = softmax(trt_outputs[1].reshape(1, 9, input_h, input_w), axis=1)[0]

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

            cv2.putText(src_img, str(lane_id[i]), (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(src_img, str(lane_type), (px, py + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.putText(src_img, 'lane_id: 7-5-3-1-2-4-6-8', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(src_img, 'line type:', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
        for i in range(len(line_type)):
            cv2.putText(src_img, str(i) + ': ' + str(line_type[i]), (10, 80 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 0), 2, cv2.LINE_AA)

        opencv_image = np.clip(np.array(src_img) + np.array(mask) * 0.4, a_min=0, a_max=255)
        opencv_image = opencv_image.astype("uint8")
        cv2.imwrite('./test_result.jpg', opencv_image)


if __name__ == '__main__':
    print('This is main ...')
    main()
