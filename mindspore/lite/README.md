[查看中文](./README_CN.md)

## What Is MindSpore Lite

MindSpore lite is a high-performance, lightweight open source reasoning framework that can be used to meet the needs of AI applications on mobile devices. MindSpore Lite focuses on how to deploy AI technology more effectively on devices. It has been integrated into HMS (Huawei Mobile Services) to provide inferences for applications such as image classification, object detection and OCR. MindSpore Lite will promote the development and enrichment of the AI software/hardware application ecosystem.

<img src="../../docs/MindSpore-Lite-architecture.png" alt="MindSpore Lite Architecture" width="600"/>

For more details please check out our [MindSpore Lite Architecture Guide](https://www.mindspore.cn/doc/note/en/master/design/mindspore/architecture_lite.html).

### MindSpore Lite features

1. Cooperative work with MindSpore training
   - Provides training, optimization, and deployment.
   - The unified IR realizes the device-cloud AI application integration.

2. Lightweight
   - Provides model compress, which could help to improve performance as well.
   - Provides the ultra-lightweight reasoning solution MindSpore Micro to meet the deployment requirements in extreme environments such as smart watches and headphones.

3. High-performance
   - The built-in high-performance kernel computing library NNACL supports multiple convolution optimization algorithms such as Slide window, im2col+gemm, winograde, etc.
   - Assembly code to improve performance of kernel operators. Supports CPU, GPU, and NPU.
4. Versatility
   - Supports IOS, Android.
   - Supports Lite OS.
   - Supports mobile device, smart screen, pad, and IOT devices.
   - Supports third party models such as TFLite, CAFFE and ONNX.

## MindSpore Lite AI deployment procedure

1. Model selection and personalized training

   Select a new model or use an existing model for incremental training using labeled data. When designing a model for mobile device, it is necessary to consider the model size, accuracy and calculation amount.

   The MindSpore team provides a series of pre-training models used for image classification, object detection. You can use these pre-trained models in your application.

   The pre-trained model provided by MindSpore: [Image Classification](https://download.mindspore.cn/model_zoo/official/lite/). More models will be provided in the feature.

   MindSpore allows you to retrain pre-trained models to perform other tasks.

2. Model converter and optimization

   If you use MindSpore or a third-party model, you need to use [MindSpore Lite Model Converter Tool](https://www.mindspore.cn/tutorial/lite/en/master/use/converter_tool.html) to convert the model into MindSpore Lite model. The MindSpore Lite model converter tool provides the converter of TensorFlow Lite, Caffe, ONNX to MindSpore Lite model, fusion and quantization could be introduced during convert procedure.

   MindSpore also provides a tool to convert models running on IoT devices .

3. Model deployment

   This stage mainly realizes model deployment, including model management, deployment, operation and maintenance monitoring, etc.

4. Inference

   Load the model and perform inference. [Inference](https://www.mindspore.cn/tutorial/lite/en/master/use/runtime.html) is the process of running input data through the model to get output.

   MindSpore provides pre-trained model that can be deployed on mobile device [example](https://www.mindspore.cn/lite/examples/en).

## MindSpore Lite benchmark test result

Base on MindSpore r1.2, we test a couple of networks on HUAWEI Mate40 (Hisilicon Kirin9000) mobile phone, and get the test results below for your reference.

| NetWork             | Thread Number | Average Run Time(ms) |
| ------------------- | ------------- | -------------------- |
| basic_squeezenet    | 4             | 7.246                |
| inception_v3        | 4             | 36.767               |
| mobilenet_v1_10_224 | 4             | 5.187                |
| mobilenet_v2_10_224 | 4             | 4.153                |
| resnet_v2_50        | 4             | 25.071               |
