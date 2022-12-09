/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <cstring>
#include <random>
#include <fstream>
#include <thread>
#include <algorithm>
#include <vector>
#include "./flags.h"
#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/types.h"

void Usage() {
  std::cerr << "Usage: ./runtime_cpp --model_path=[model_path] --device_type=[device_type] --device_id=[device_id]"
               " --batch_size=[batch_size] --config_file=[config_file]"
            << std::endl;
  std::cerr << "Example: ./runtime_cpp ../model/mobilenetv2.mindr GPU 0 1" << std::endl;
  std::cerr << "[device_type], optional, can be GPU, Ascend or CPU, default CPU" << std::endl;
  std::cerr << "[device_id], optional, should be an integer >=0, default 0" << std::endl;
  std::cerr << "[batch_size], optional, should be an positive integer, default 1" << std::endl;
  std::cerr << "[config_file], optional, config file for dynamic input" << std::endl;
}

struct CommandArgs {
  std::string model_path;
  std::string device_type;
  std::string config_file;
  int32_t device_id = 0;
  int32_t batch_size = 1;
};

int Run(const CommandArgs &args);

int main(int argc, const char **argv) {
  DEFINE_string(model_path, "", "model path");
  DEFINE_string(device_type, "CPU", "device type, optional, can be GPU, Ascend or CPU, default CPU");
  DEFINE_string(config_file, "", "config file for dynamic input");
  DEFINE_int32(device_id, 0, "device id, optional, should be an integer >=0, default 0");
  DEFINE_int32(batch_size, 1, "optional, should be an positive integer, default 1");

  if (!mindspore::example::ParseCommandLineFlags(argc, argv)) {
    std::cerr << "Failed to parse command args" << std::endl;
    Usage();
  }
  CommandArgs args;
  args.model_path = FLAGS_model_path;
  if (args.model_path.empty()) {
    std::cerr << "model path " << args.model_path << " is invalid.";
    return -1;
  }
  args.device_type = FLAGS_device_type;
  if (args.device_type != "Ascend" && args.device_type != "GPU" && args.device_type != "CPU") {
    Usage();
    return -1;
  }
  args.device_id = FLAGS_device_id;
  if (args.device_id < 0) {
    Usage();
    return -1;
  }
  args.batch_size = FLAGS_batch_size;
  if (args.batch_size <= 0) {
    Usage();
    return -1;
  }
  args.config_file = FLAGS_model_path;
  Run(args);
  return 0;
}

using MemBuffer = std::vector<uint8_t>;

int SetTensorHostData(std::vector<mindspore::MSTensor> *tensors, std::vector<MemBuffer> *buffers) {
  if (!tensors || !buffers) {
    std::cerr << "Argument tensors or buffers cannot be nullptr" << std::endl;
    return -1;
  }
  if (tensors->size() != buffers->size()) {
    std::cerr << "tensors size " << tensors->size() << " != "
              << " buffers size " << buffers->size() << std::endl;
    return -1;
  }
  for (size_t i = 0; i < tensors->size(); i++) {
    auto &tensor = (*tensors)[i];
    auto &buffer = (*buffers)[i];
    if (tensor.DataSize() != buffer.size()) {
      std::cerr << "Tensor data size " << tensor.DataSize() << " != buffer size " << buffer.size() << std::endl;
      return -1;
    }
    // set tensor data, and the memory should be freed by user
    tensor.SetData(buffer.data(), false);
    tensor.SetDeviceData(nullptr);
  }
  return 0;
}

int CopyTensorHostData(std::vector<mindspore::MSTensor> *tensors, std::vector<MemBuffer> *buffers) {
  for (size_t i = 0; i < tensors->size(); i++) {
    auto &tensor = (*tensors)[i];
    auto &buffer = (*buffers)[i];
    if (tensor.DataSize() != buffer.size()) {
      std::cerr << "Tensor data size " << tensor.DataSize() << " != buffer size " << buffer.size() << std::endl;
      return -1;
    }
    auto dst_mem = tensor.MutableData();
    if (dst_mem == nullptr) {
      std::cerr << "Tensor MutableData return nullptr" << std::endl;
      return -1;
    }
    memcpy(tensor.MutableData(), buffer.data(), buffer.size());
  }
  return 0;
}

std::shared_ptr<mindspore::CPUDeviceInfo> CreateCPUDeviceInfo() {
  auto device_info = std::make_shared<mindspore::CPUDeviceInfo>();
  if (device_info == nullptr) {
    std::cerr << "New CPUDeviceInfo failed." << std::endl;
    return nullptr;
  }
  return device_info;
}

std::shared_ptr<mindspore::GPUDeviceInfo> CreateGPUDeviceInfo(int32_t device_id) {
  auto device_info = std::make_shared<mindspore::GPUDeviceInfo>();
  if (device_info == nullptr) {
    std::cerr << "New GPUDeviceInfo failed." << std::endl;
    return nullptr;
  }
  device_info->SetDeviceID(device_id);
  return device_info;
}

std::shared_ptr<mindspore::AscendDeviceInfo> CreateAscendDeviceInfo(int32_t device_id) {
  // for Ascend 310, 310P
  auto device_info = std::make_shared<mindspore::AscendDeviceInfo>();
  if (device_info == nullptr) {
    std::cerr << "New AscendDeviceInfo failed." << std::endl;
    return nullptr;
  }
  device_info->SetDeviceID(device_id);
  return device_info;
}

std::shared_ptr<mindspore::Model> BuildModel(const CommandArgs &args) {
  auto device_type = args.device_type;
  auto device_id = args.device_id;
  auto model_path = args.model_path;
  auto config_file = args.config_file;
  // Create and init context, add CPU device info
  auto context = std::make_shared<mindspore::Context>();
  if (context == nullptr) {
    std::cerr << "New context failed." << std::endl;
    return nullptr;
  }
  auto &device_list = context->MutableDeviceInfo();
  std::shared_ptr<mindspore::DeviceInfoContext> device_info = nullptr;
  if (device_type == "CPU") {
    device_info = CreateCPUDeviceInfo();
  } else if (device_type == "GPU") {
    device_info = CreateGPUDeviceInfo(device_id);
  } else if (device_type == "Ascend") {
    device_info = CreateAscendDeviceInfo(device_id);
  }
  if (device_info == nullptr) {
    std::cerr << "Create " << device_type << "DeviceInfo failed." << std::endl;
    return nullptr;
  }
  device_list.push_back(device_info);

  // Create model
  auto model = std::make_shared<mindspore::Model>();
  if (model == nullptr) {
    std::cerr << "New Model failed." << std::endl;
    return nullptr;
  }
  if (!config_file.empty()) {
    if (model->LoadConfig(config_file) != mindspore::kSuccess) {
      std::cerr << "Failed to load config file " << config_file << std::endl;
      return nullptr;
    }
  }
  // Build model
  auto build_ret = model->Build(model_path, mindspore::kMindIR, context);
  if (build_ret != mindspore::kSuccess) {
    std::cerr << "Build model failed." << std::endl;
    return nullptr;
  }
  return model;
}

int ResizeModel(std::shared_ptr<mindspore::Model> model, int32_t batch_size) {
  std::vector<std::vector<int64_t>> new_shapes;
  auto inputs = model->GetInputs();
  for (auto &input : inputs) {
    auto shape = input.Shape();
    shape[0] = batch_size;
    new_shapes.push_back(shape);
  }
  if (model->Resize(inputs, new_shapes) != mindspore::kSuccess) {
    std::cerr << "Failed to resize to batch size " << batch_size << std::endl;
    return -1;
  }
  return 0;
}

class InferenceApp {
 public:
  std::vector<MemBuffer> &GetInferenceInputs(const std::vector<mindspore::MSTensor> &inputs) {
    if (AllocBufferMem(inputs, &input_buffer_) != 0) {
      std::cerr << "Failed to alloc inputs memory" << std::endl;
      input_buffer_.clear();
      return input_buffer_;
    }
    GenerateInputsData(&input_buffer_);
    return input_buffer_;
  }

  int SetInferenceInputs(std::vector<mindspore::MSTensor> *inputs) {
    if (AllocBufferMem(*inputs, &input_buffer_) != 0) {
      std::cerr << "Failed to alloc inputs memory" << std::endl;
      return -1;
    }
    GenerateInputsData(&input_buffer_);
    if (SetTensorHostData(inputs, &input_buffer_) != 0) {
      std::cerr << "Failed to set inputs data" << std::endl;
      return -1;
    }
    return 0;
  }

  std::vector<MemBuffer> &GetInferenceResultBuffer(const std::vector<mindspore::MSTensor> &outputs) {
    if (AllocBufferMem(outputs, &output_buffer_) != 0) {
      std::cerr << "Failed to alloc inputs memory" << std::endl;
      output_buffer_.clear();
    }
    return output_buffer_;
  }

  void OnInferenceResult(const std::vector<mindspore::MSTensor> &outputs) { PrintOutputsTensor(outputs); }

 private:
  std::vector<MemBuffer> input_buffer_;
  std::vector<MemBuffer> output_buffer_;

  template <typename T, typename Distribution>
  void GenerateRandomData(int size, void *data, Distribution distribution) {
    if (data == nullptr) {
      std::cerr << "data is nullptr." << std::endl;
      return;
    }
    std::mt19937 random_engine;
    int elements_num = size / sizeof(T);
    (void)std::generate_n(static_cast<T *>(data), elements_num,
                          [&]() { return static_cast<T>(distribution(random_engine)); });
  }

  void GenerateInputsData(std::vector<MemBuffer> *buffers) {
    for (auto &buffer : *buffers) {
      GenerateRandomData<float>(buffer.size(), buffer.data(), std::uniform_real_distribution<float>(0.1f, 1.0f));
    }
  }

  int AllocBufferMem(const std::vector<mindspore::MSTensor> &tensors, std::vector<MemBuffer> *buffers) {
    buffers->clear();
    for (auto &in_tensor : tensors) {
      auto data_size = in_tensor.DataSize();
      if (data_size == 0) {
        std::cerr << "Data size cannot be 0" << std::endl;
        return -1;
      }
      MemBuffer buffer;
      buffer.resize(data_size);
      buffers->push_back(std::move(buffer));
    }
    return 0;
  }

  template <class T>
  void PrintBuffer(const void *buffer, size_t elem_count) {
    auto data = reinterpret_cast<const T *>(buffer);
    constexpr size_t max_print_count = 50;
    for (size_t i = 0; i < elem_count && i <= max_print_count; i++) {
      std::cout << data[i] << " ";
    }
    std::cout << std::endl;
  }

  void PrintOutputsTensor(const std::vector<mindspore::MSTensor> &outputs) {
    for (auto &tensor : outputs) {
      auto elem_num = tensor.ElementNum();
      auto data_size = tensor.DataSize();
      std::vector<uint8_t> host_data;
      const void *print_data = tensor.Data().get();
      if (print_data == nullptr) {
        std::cerr << "Tensor data is nullptr";
        return;
      }
      std::cout << "tensor name is:" << tensor.Name() << " tensor size is:" << data_size
                << " tensor elements num is:" << elem_num << std::endl;
      auto data_type = tensor.DataType();
      if (data_type == mindspore::DataType::kNumberTypeFloat32) {
        PrintBuffer<float>(print_data, elem_num);
      } else if (data_type == mindspore::DataType::kNumberTypeFloat64) {
        PrintBuffer<double>(print_data, elem_num);
      } else if (data_type == mindspore::DataType::kNumberTypeInt64) {
        PrintBuffer<int64_t>(print_data, elem_num);
      } else if (data_type == mindspore::DataType::kNumberTypeInt32) {
        PrintBuffer<int32_t>(print_data, elem_num);
      } else if (data_type == mindspore::DataType::kNumberTypeInt16) {
        PrintBuffer<int16_t>(print_data, elem_num);
      } else if (data_type == mindspore::DataType::kNumberTypeInt8) {
        PrintBuffer<int8_t>(print_data, elem_num);
      } else if (data_type == mindspore::DataType::kNumberTypeUInt64) {
        PrintBuffer<uint64_t>(print_data, elem_num);
      } else if (data_type == mindspore::DataType::kNumberTypeUInt32) {
        PrintBuffer<uint32_t>(print_data, elem_num);
      } else if (data_type == mindspore::DataType::kNumberTypeUInt16) {
        PrintBuffer<uint16_t>(print_data, elem_num);
      } else if (data_type == mindspore::DataType::kNumberTypeUInt8) {
        PrintBuffer<uint8_t>(print_data, elem_num);
      } else if (data_type == mindspore::DataType::kNumberTypeBool) {
        PrintBuffer<bool>(print_data, elem_num);
      } else {
        std::cout << "Unsupported data type " << static_cast<int>(tensor.DataType()) << std::endl;
      }
    }
  }
};

int SpecifyInputDataExample(const CommandArgs &args) {
  auto batch_size = args.batch_size;
  auto model = BuildModel(args);
  if (model == nullptr) {
    std::cerr << "Create and build model failed." << std::endl;
    return -1;
  }
  // If the model inputs are dynamic, Model.Resize should be performed first.
  // Otherwise, the shapes get from GetInputs/GetOutputs may contain -1, and DataSize may be 0.
  if (ResizeModel(model, batch_size) != 0) {
    return -1;
  }
  auto inputs = model->GetInputs();
  InferenceApp app;
  // The input data for inference may come from the preprocessing result.
  auto &input_buffer = app.GetInferenceInputs(inputs);
  if (input_buffer.empty()) {
    return -1;
  }
  // Set the input data of the model, this inference input will be copied directly to the device.
  SetTensorHostData(&inputs, &input_buffer);

  std::vector<mindspore::MSTensor> outputs;
  auto predict_ret = model->Predict(inputs, &outputs);
  if (predict_ret != mindspore::kSuccess) {
    std::cerr << "Predict error " << predict_ret << std::endl;
    return -1;
  }
  app.OnInferenceResult(outputs);
  return 0;
}

int SpecifyOutputDataExample(const CommandArgs &args) {
  auto batch_size = args.batch_size;
  auto model = BuildModel(args);
  if (model == nullptr) {
    std::cerr << "Create and build model failed." << std::endl;
    return -1;
  }
  // If the model inputs are dynamic, Model.Resize should be performed first.
  // Otherwise, the shapes get from GetInputs/GetOutputs may contain -1, and DataSize may be 0.
  if (ResizeModel(model, batch_size) != 0) {
    return -1;
  }
  auto inputs = model->GetInputs();
  InferenceApp app;
  // The input data for inference may come from the preprocessing result.
  std::vector<MemBuffer> &input_buffer = app.GetInferenceInputs(inputs);
  if (input_buffer.empty()) {
    return -1;
  }
  // Set the input data of the model, this inference input will be copied directly to the device.
  SetTensorHostData(&inputs, &input_buffer);

  auto outputs = model->GetOutputs();
  auto &output_buffer = app.GetInferenceResultBuffer(outputs);
  if (output_buffer.empty()) {
    return -1;
  }
  // Set output buffer of the model, inference result will be copied directly from the device to this buffer.
  SetTensorHostData(&outputs, &output_buffer);

  auto predict_ret = model->Predict(inputs, &outputs);
  if (predict_ret != mindspore::kSuccess) {
    std::cerr << "Predict error " << predict_ret << std::endl;
    return -1;
  }
  app.OnInferenceResult(outputs);
  return 0;
}

int CopyInputDataExample(const CommandArgs &args) {
  auto batch_size = args.batch_size;
  auto model = BuildModel(args);
  if (model == nullptr) {
    std::cerr << "Create and build model failed." << std::endl;
    return -1;
  }
  // If the model inputs are dynamic, Model.Resize should be performed first.
  // Otherwise, the shapes get from GetInputs/GetOutputs may contain -1, and DataSize may be 0.
  if (ResizeModel(model, batch_size) != 0) {
    return -1;
  }
  auto inputs = model->GetInputs();
  InferenceApp app;
  // Generate model input data
  auto &input_buffer = app.GetInferenceInputs(inputs);
  if (input_buffer.empty()) {
    return -1;
  }
  // Set the input data of the model, copy data to the tensor buffer of Model.GetInputs.
  CopyTensorHostData(&inputs, &input_buffer);

  std::vector<mindspore::MSTensor> outputs;
  auto predict_ret = model->Predict(inputs, &outputs);
  if (predict_ret != mindspore::kSuccess) {
    std::cerr << "Predict error " << predict_ret << std::endl;
    return -1;
  }
  app.OnInferenceResult(outputs);
  return 0;
}

int Run(const CommandArgs &args) {
  SpecifyInputDataExample(args);
  SpecifyOutputDataExample(args);
  CopyInputDataExample(args);
  return 0;
}
