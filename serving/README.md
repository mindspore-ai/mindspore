# 基于MindSpore部署预测服务


<!-- TOC -->
- [基于MindSpore部署预测服务](#基于mindspore部署预测服务)
	- [概述](#概述)
	- [启动Serving服务](#启动serving服务)
	- [应用示例](#应用示例)
		- [导出模型](#导出模型)
		- [启动Serving推理服务](#启动serving推理服务)
		- [客户端示例](#客户端示例)


## 概述

MindSpore Serving是一个轻量级、高性能的服务模块，旨在帮助MindSpore开发者在生产环境中高效部署在线预测服务。当用户使用MindSpore完成模型训练后，导出MindSpore模型，即可使用MindSpore Serving创建该模型的预测服务。当前Serving仅支持Ascend 910。


## 启动Serving服务
通过pip安装MindSpore后，Serving可执行程序位于`/{your python path}/lib/python3.7/site-packages/mindspore/ms_serving` 。
启动Serving服务命令如下
```bash
ms_serving [--help] [--model_path <MODEL_PATH>] [--model_name <MODEL_NAME>]
                  [--port <PORT>] [--device_id <DEVICE_ID>]
```
参数含义如下

|参数名|属性|功能描述|参数类型|默认值|取值范围|
|---|---|---|---|---|---|
|`--help`|可选|显示启动命令的帮助信息。|-|-|-|
|`--model_path <MODEL_PATH>`|必选|指定待加载模型的存放路径。|str|空|-|
|`--model_name <MODEL_NAME>`|必选|指定待加载模型的文件名。|str|空|-|
|`--port <PORT>`|可选|指定Serving对外的端口号。|int|5500|1~65535|
|`--device_id <DEVICE_ID>`|可选|指定使用的设备号|int|0|0~7|

 > 执行启动命令前，需将`/{your python path}/lib:/{your python path}/lib/python3.7/site-packages/mindspore/lib`对应的路径加入到环境变量LD_LIBRARY_PATH中 。

## 应用示例
下面以一个简单的网络为例，演示MindSpore Serving如何使用。

### 导出模型
使用[add_model.py](https://gitee.com/mindspore/mindspore/blob/master/serving/example/export_model/add_model.py)，构造一个只有Add算子的网络，并导出MindSpore推理部署模型。

```python
python add_model.py
```
执行脚本，生成add.pb文件，该模型的输入为两个shape为[4]的一维Tensor，输出结果是两个输入Tensor之和。

### 启动Serving推理服务
```bash
ms_serving --model_path={current path} --model_name=add.pb
```
当服务端打印日志`MS Serving Listening on 0.0.0.0:5500`时，表示Serving服务已加载推理模型完毕。

### 客户端示例
执行如下命令，编译一个客户端示例程序，并向Serving服务发送推理请求。
```bash
cd mindspore/serving/example/cpp_client
mkdir build
cmake ..
make
./ms_client --target=localhost:5500
```
显示如下返回值说明Serving服务已正确执行Add网络的推理。
```
Compute [1, 2, 3, 4] + [1, 2, 3, 4]
Add result is [2, 4, 6, 8]
client received: RPC OK
```
 > 编译客户端要求用户本地已安装c++版本的[gRPC](https://gRPC.io)，并将对应路径加入到环境变量`PATH`中。

客户端代码主要包含以下几个部分：

1. 基于MSService::Stub实现Client，并创建Client实例。
    ```
    class MSClient {
     public:
      explicit MSClient(std::shared_ptr<Channel> channel) :  stub_(MSService::NewStub(channel)) {}
     private:
      std::unique_ptr<MSService::Stub> stub_;
    };MSClient client(grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));
    
    MSClient client(grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));
    
    ```
2. 根据网络的实际输入构造请求的入参Request、出参Reply和gRPC的客户端Context。
    ```
    PredictRequest request;
    PredictReply reply;
    ClientContext context;
    
    //construct tensor
    Tensor data;
    
    //set shape
    TensorShape shape;
    shape.add_dims(4);
    *data.mutable_tensor_shape() = shape;
    
    //set type
    data.set_tensor_type(ms_serving::MS_FLOAT32);
    std::vector<float> input_data{1, 2, 3, 4};
    
    //set datas
    data.set_data(input_data.data(), input_data.size());
    
    //add tensor to request
    *request.add_data() = data;
    *request.add_data() = data;
    ```
3. 调用gRPC接口和已经启动的Serving服务通信，并取回返回值。
    ```
    Status status = stub_->Predict(&context, request, &reply);
    ```

完整代码参考[ms_client](https://gitee.com/mindspore/mindspore/blob/master/serving/example/cpp_client/ms_client.cc)。 

