# MindSpore-based Inference Service Deployment


<!-- TOC -->

- [MindSpore-based Inference Service Deployment](#mindspore-based-inference-service-deployment)
    - [Overview](#overview)
    - [Starting Serving](#starting-serving)
    - [Application Example](#application-example)
        - [Exporting Model](#exporting-model)
        - [Starting Serving Inference](#starting-serving-inference)
        - [Client Samples](#client-samples)
            - [Python Client Sample](#python-client-sample)
            - [C++ Client Sample](#cpp-client-sample)

<!-- /TOC -->
<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/advanced_use/serving.md" target="_blank"><img src="../_static/logo_source.png"></a>


## Overview

MindSpore Serving is a lightweight and high-performance service module that helps MindSpore developers efficiently deploy online inference services in the production environment. After completing model training using MindSpore, you can export the MindSpore model and use MindSpore Serving to create an inference service for the model. Currently, only Ascend 910 is supported.


## Starting Serving
After MindSpore is installed using `pip`, the Serving executable program is stored in `/{your python path}/lib/python3.7/site-packages/mindspore/ms_serving`.
Run the following command to start Serving:
```bash
ms_serving [--help] [--model_path <MODEL_PATH>] [--model_name <MODEL_NAME>]
                  [--port <PORT>] [--device_id <DEVICE_ID>]
```
Parameters are described as follows:

|Parameter|Attribute|Function|Parameter Type|Default Value|Value Range|
|---|---|---|---|---|---|
|`--help`|Optional|Displays the help information about the startup command. |-|-|-|
|`--model_path=<MODEL_PATH>`|Mandatory|Path for storing the model to be loaded. |String|Null|-|
|`--model_name=<MODEL_NAME>`|Mandatory|Name of the model file to be loaded. |String|Null|-|
|`--=port <PORT>`|Optional|Specifies the external Serving port number. |Integer|5500|1–65535|
|`--device_id=<DEVICE_ID>`|Optional|Specifies device ID to be used.|Integer|0|0 to 7|

 > Before running the startup command, add the path `/{your python path}/lib:/{your python path}/lib/python3.7/site-packages/mindspore/lib` to the environment variable `LD_LIBRARY_PATH`.

## Application Example
The following uses a simple network as an example to describe how to use MindSpore Serving.

### Exporting Model
Use [add_model.py](https://gitee.com/mindspore/mindspore/blob/master/serving/example/export_model/add_model.py) to build a network with only the Add operator and export the MindSpore inference deployment model.

```python
python add_model.py
```
Execute the script to generate the `tensor_add.mindir` file. The input of the model is two one-dimensional tensors with shape [2,2], and the output is the sum of the two input tensors.

### Starting Serving Inference
```bash
ms_serving --model_path={model directory} --model_name=tensor_add.mindir
```
If the server prints the `MS Serving Listening on 0.0.0.0:5500` log, the Serving has loaded the inference model.

### Client Samples
#### <span name="python-client-sample">Python Client Sample</span>
Obtain [ms_client.py](https://gitee.com/mindspore/mindspore/blob/master/serving/example/python_client/ms_client.py) and start the Python client.
```bash
python ms_client.py
```

If the following information is displayed, the Serving has correctly executed the inference of the Add network.
```
ms client received:
[[2. 2.]
 [2. 2.]]
```

#### <span name="cpp-client-sample">C++ Client Sample</span>
1. Obtain an executable client sample program.

    Download the [MindSpore source code](https://gitee.com/mindspore/mindspore). You can use either of the following methods to compile and obtain the client sample program:
    + When MindSpore is compiled using the source code, the Serving C++ client sample program is generated. You can find the `ms_client` executable program in the `build/mindspore/serving/example/cpp_client` directory.
    + Independent compilation

        Preinstall [gRPC](https://gRPC.io).

        Run the following command in the MindSpore source code path to compile a client sample program:
        ```bash
        cd mindspore/serving/example/cpp_client
        mkdir build && cd build
        cmake -D GRPC_PATH={grpc_install_dir} ..
        make
        ```
        In the preceding command, `{grpc_install_dir}` indicates the gRPC installation path. Replace it with the actual gRPC installation path.

2. Start the client.

    Execute `ms_client` to send an inference request to the Serving.
    ```bash
    ./ms_client --target=localhost:5500
    ```
    If the following information is displayed, the Serving has correctly executed the inference of the Add network.
    ```
    Compute [[1, 2], [3, 4]] + [[1, 2], [3, 4]]
    Add result is 2 4 6 8
    client received: RPC OK
    ```

The client code consists of the following parts:

1. Implement the client based on MSService::Stub and create a client instance.
    ```
    class MSClient {
     public:
      explicit MSClient(std::shared_ptr<Channel> channel) :  stub_(MSService::NewStub(channel)) {}
     private:
      std::unique_ptr<MSService::Stub> stub_;
    };MSClient client(grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));
    
    MSClient client(grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));
    
    ```
2. Build the request input parameter `Request`, output parameter `Reply`, and gRPC client `Context` based on the actual network input.
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
3. Call the gRPC API to communicate with the Serving that has been started, and obtain the return value.
    ```
    Status status = stub_->Predict(&context, request, &reply);
    ```

For details about the complete code, see [ms_client](https://gitee.com/mindspore/mindspore/blob/master/serving/example/cpp_client/ms_client.cc). 
