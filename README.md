# UNetMultiLane_onnx_tensorRT_rknn_horizon

UNetMultiLane 多车道线和车道线类型识别部署版本，测试不同平台部署（onnx、tensorRT、RKNN、Horzion），可识别所在的车道和车道线的类型。

训练代码参考[【UNetMultiLane】](https://github.com/cqu20160901/UNetMultiLane)

# 文件夹结构说明

onnx：onnx模型、测试图像、测试结果、测试demo脚本

TensorRT：TensorRT版本模型、测试图像、测试结果、测试demo脚本、onnx模型、onnx2tensorRT脚本(tensorRT-7.2.3.4)

rknn：rknn模型、测试（量化）图像、测试结果、onnx2rknn转换测试脚本

horizon：地平线模型、测试（量化）图像、测试结果、转换测试脚本、测试量化后onnx模型脚本

# 数据说明

基于UNet的分割模型，增加了检测头来识别车道线的类型，基于开源数据集 VIL100。其中数据标注了所在的六个车道的车道线和车道线的类型。

8条车道线（六个车道），对应的顺序是：7,5,3,1,2,4,6,8。其中1,2对应的自车所在的车道，从左往右标记。

车道线的类别（10个类别）：单条白色实线、单条白色虚线、单条黄色实线、单条黄色虚线、双条白色实线、双条黄色实线、双条黄色虚线、双条白色实虚线、双条白色黄色实线、双条白色虚实线。

基于UNet 进行修改，可以识别出"所在车道"和"车道线类型"。

# 分割效果

![image](https://github.com/cqu20160901/UNetMultiLane_onnx_tensorRT_rknn_horizon/blob/main/onnx/test_result.jpg)

![image](https://github.com/cqu20160901/UNetMultiLane_onnx_tensorRT_rknn_horizon/assets/22290931/b265e79a-598e-4b24-9f9a-8bafdc0edd9c)
