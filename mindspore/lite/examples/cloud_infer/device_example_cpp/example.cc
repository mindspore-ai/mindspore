/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include <cstring>
#include <memory>
#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/status.h"
#include "include/api/types.h"

#include "./mem_gpu.h"

namespace {
constexpr int kNumPrintOfOutData = 50;
}

static std::string ShapeToString(const std::vector<int64_t> &shape) {
  std::string result = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    result += std::to_string(shape[i]);
    if (i + 1 < shape.size()) {
      result += ", ";
    }
  }
  result += "]";
  return result;
}

template <typename T, typename Distribution>
void GenerateRandomData(int size, void *data, Distribution distribution) {
  std::random_device rd{};
  std::mt19937 random_engine{rd()};
  int elements_num = size / sizeof(T);
  (void)std::generate_n(static_cast<T *>(data), elements_num,
                        [&distribution, &random_engine]() { return static_cast<T>(distribution(random_engine)); });
}

int GenerateRandomInputData(std::vector<mindspore::MSTensor> inputs, std::vector<uint8_t *> *host_data_buffer) {
  for (auto tensor : inputs) {
    auto data_size = tensor.DataSize();
    if (data_size == 0) {
      std::cerr << "Data size cannot be 0, tensor shape: " << ShapeToString(tensor.Shape()) << std::endl;
      return -1;
    }
    auto host_data = new uint8_t[data_size];
    host_data_buffer->push_back(host_data);
    GenerateRandomData<float>(data_size, host_data, std::normal_distribution<float>(0.0f, 1.0f));
  }
  return 0;
}

int SetHostData(std::vector<mindspore::MSTensor> tensors, const std::vector<uint8_t *> &host_data_buffer) {
  for (size_t i = 0; i < tensors.size(); i++) {
    tensors[i].SetData(host_data_buffer[i], false);
    tensors[i].SetDeviceData(nullptr);
  }
  return 0;
}

int SetDeviceData(std::vector<mindspore::MSTensor> tensors, const std::vector<uint8_t *> &host_data_buffer,
                  std::vector<void *> *device_buffers) {
  for (size_t i = 0; i < tensors.size(); i++) {
    auto &tensor = tensors[i];
    auto host_data = host_data_buffer[i];
    auto data_size = tensor.DataSize();
    if (data_size == 0) {
      std::cerr << "Data size cannot be 0, tensor shape: " << ShapeToString(tensor.Shape()) << std::endl;
      return -1;
    }
    auto device_data = MallocDeviceMemory(data_size);
    if (device_data == nullptr) {
      std::cerr << "Failed to alloc device data, data size " << data_size << std::endl;
      return -1;
    }
    device_buffers->push_back(device_data);
    if (CopyMemoryHost2Device(device_data, data_size, host_data, data_size) != 0) {
      std::cerr << "Failed to copy data to device, data size " << data_size << std::endl;
      return -1;
    }
    tensor.SetDeviceData(device_data);
    tensor.SetData(nullptr, false);
  }
  return 0;
}

int SetOutputHostData(std::vector<mindspore::MSTensor> tensors, std::vector<uint8_t *> *host_buffers) {
  for (size_t i = 0; i < tensors.size(); i++) {
    auto &tensor = tensors[i];
    auto data_size = tensor.DataSize();
    if (data_size == 0) {
      std::cerr << "Data size cannot be 0, tensor shape: " << ShapeToString(tensor.Shape()) << std::endl;
      return -1;
    }
    auto host_data = new uint8_t[data_size];
    host_buffers->push_back(host_data);
    tensor.SetData(host_data, false);
    tensor.SetDeviceData(nullptr);
  }
  return 0;
}

int SetOutputDeviceData(std::vector<mindspore::MSTensor> tensors, std::vector<void *> *device_buffers) {
  for (size_t i = 0; i < tensors.size(); i++) {
    auto &tensor = tensors[i];
    auto data_size = tensor.DataSize();
    if (data_size == 0) {
      std::cerr << "Data size cannot be 0, tensor shape: " << ShapeToString(tensor.Shape()) << std::endl;
      return -1;
    }
    auto device_data = MallocDeviceMemory(data_size);
    if (device_data == nullptr) {
      std::cerr << "Failed to alloc device data, data size " << data_size << std::endl;
      return -1;
    }
    device_buffers->push_back(device_data);
    tensor.SetDeviceData(device_data);
    tensor.SetData(nullptr, false);
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

void PrintOutputsTensor(std::vector<mindspore::MSTensor> outputs) {
  for (auto tensor : outputs) {
    auto elem_num = tensor.ElementNum();
    auto data_size = tensor.DataSize();
    std::vector<uint8_t> host_data;
    const void *print_data;
    if (tensor.GetDeviceData() != nullptr) {
      host_data.resize(data_size);
      CopyMemoryDevice2Host(host_data.data(), host_data.size(), tensor.GetDeviceData(), data_size);
      print_data = host_data.data();
      std::cout << "Device data, tensor name is:" << tensor.Name() << " tensor size is:" << data_size
                << " tensor elements num is:" << elem_num << std::endl;
    } else {
      print_data = tensor.Data().get();
      std::cout << "Host data, tensor name is:" << tensor.Name() << " tensor size is:" << data_size
                << " tensor elements num is:" << elem_num << std::endl;
    }
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

int Predict(mindspore::Model *model, const std::vector<mindspore::MSTensor> &inputs,
            std::vector<mindspore::MSTensor> *outputs) {
  auto ret = model->Predict(inputs, outputs);
  if (ret != mindspore::kSuccess) {
    std::cerr << "Predict error " << ret << std::endl;
    return -1;
  }
  PrintOutputsTensor(*outputs);
  return 0;
}

class ResourceGuard {
 public:
  explicit ResourceGuard(std::function<void()> rel_func) : rel_func_(rel_func) {}
  ~ResourceGuard() {
    if (rel_func_) {
      rel_func_();
    }
  }

 private:
  std::function<void()> rel_func_ = nullptr;
};

int TestHostDeviceInput(mindspore::Model *model, uint32_t batch_size) {
  // Get Input
  auto inputs = model->GetInputs();
  std::vector<std::vector<int64_t>> input_shapes;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(input_shapes), [batch_size](auto &item) {
    auto shape = item.Shape();
    shape[0] = batch_size;
    return shape;
  });
  if (model->Resize(inputs, input_shapes) != mindspore::kSuccess) {
    std::cerr << "Failed to resize model batch size to " << batch_size << std::endl;
    return -1;
  }
  std::cout << "Success resize model batch size to " << batch_size << std::endl;

  // Generate random data as input data.
  std::vector<uint8_t *> host_buffers;
  ResourceGuard host_rel([&host_buffers]() {
    for (auto &item : host_buffers) {
      delete[] item;
    }
  });

  std::vector<void *> device_buffers;
  ResourceGuard device_rel([&device_buffers]() {
    for (auto &item : device_buffers) {
      FreeDeviceMemory(item);
    }
  });

  auto ret = GenerateRandomInputData(inputs, &host_buffers);
  if (ret != 0) {
    std::cerr << "Generate Random Input Data failed." << std::endl;
    return -1;
  }
  // empty outputs
  std::vector<mindspore::MSTensor> outputs;
  // Model Predict, input host memory
  SetHostData(inputs, host_buffers);
  if (Predict(model, inputs, &outputs) != 0) {
    return -1;
  }
  // Model Predict, input device memory
  outputs.clear();
  SetDeviceData(inputs, host_buffers, &device_buffers);
  if (Predict(model, inputs, &outputs) != 0) {
    return -1;
  }
  return 0;
}

int TestHostDeviceOutput(mindspore::Model *model, uint32_t batch_size) {
  // Get Input
  auto inputs = model->GetInputs();
  std::vector<std::vector<int64_t>> input_shapes;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(input_shapes), [batch_size](auto &item) {
    auto shape = item.Shape();
    shape[0] = batch_size;
    return shape;
  });
  if (model->Resize(inputs, input_shapes) != mindspore::kSuccess) {
    std::cerr << "Failed to resize model batch size to " << batch_size << std::endl;
    return -1;
  }
  std::cout << "Success resize model batch size to " << batch_size << std::endl;

  // Generate random data as input data.
  std::vector<uint8_t *> host_buffers;
  ResourceGuard host_rel([&host_buffers]() {
    for (auto &item : host_buffers) {
      delete[] item;
    }
  });

  std::vector<void *> device_buffers;
  ResourceGuard device_rel([&device_buffers]() {
    for (auto &item : device_buffers) {
      FreeDeviceMemory(item);
    }
  });

  auto ret = GenerateRandomInputData(inputs, &host_buffers);
  if (ret != 0) {
    std::cerr << "Generate Random Input Data failed." << std::endl;
    return -1;
  }
  // Get Output from model
  auto outputs = model->GetOutputs();
  // ---------------------- output host data
  std::vector<uint8_t *> output_host_buffers;
  ResourceGuard output_host_rel([&output_host_buffers]() {
    for (auto &item : output_host_buffers) {
      delete[] item;
    }
  });
  if (SetOutputHostData(outputs, &output_host_buffers) != 0) {
    std::cerr << "Failed to set output host data" << std::endl;
    return -1;
  }
  // Model Predict, input host memory
  SetHostData(inputs, host_buffers);
  if (Predict(model, inputs, &outputs) != 0) {
    return -1;
  }
  // Model Predict, input device memory
  if (SetDeviceData(inputs, host_buffers, &device_buffers) != 0) {
    std::cerr << "Failed to set input device data" << std::endl;
    return -1;
  }
  if (Predict(model, inputs, &outputs) != 0) {
    return -1;
  }
  // ---------------------- output device data
  std::vector<void *> output_device_buffers;
  ResourceGuard output_device_rel([&output_device_buffers]() {
    for (auto &item : output_device_buffers) {
      FreeDeviceMemory(item);
    }
  });
  if (SetOutputDeviceData(outputs, &output_device_buffers) != 0) {
    std::cerr << "Failed to set output device data" << std::endl;
    return -1;
  }
  // Model Predict, input host memory
  SetHostData(inputs, host_buffers);
  if (Predict(model, inputs, &outputs) != 0) {
    return -1;
  }
  // Model Predict, input device memory
  if (SetDeviceData(inputs, host_buffers, &device_buffers) != 0) {
    std::cerr << "Failed to set input device data" << std::endl;
    return -1;
  }
  if (Predict(model, inputs, &outputs) != 0) {
    return -1;
  }
  return 0;
}

int QuickStart(int argc, const char **argv) {
  if (argc < 2) {
    std::cerr << "Model file must be provided.\n";
    return -1;
  }
  // Read model file.
  std::string model_path = argv[1];
  if (model_path.empty()) {
    std::cerr << "Model path " << model_path << " is invalid.";
    return -1;
  }
  // Create and init context, add CPU device info
  auto context = std::make_shared<mindspore::Context>();
  if (context == nullptr) {
    std::cerr << "New context failed." << std::endl;
    return -1;
  }
  auto &device_list = context->MutableDeviceInfo();
  auto device_info = std::make_shared<mindspore::GPUDeviceInfo>();
  device_info->SetDeviceID(0);
  if (device_info == nullptr) {
    std::cerr << "New CPUDeviceInfo failed." << std::endl;
    return -1;
  }
  device_list.push_back(device_info);

  mindspore::Model model;
  // Build model
  auto build_ret = model.Build(model_path, mindspore::kMindIR, context);
  if (build_ret != mindspore::kSuccess) {
    std::cerr << "Build model error " << build_ret << std::endl;
    return -1;
  }
  TestHostDeviceInput(&model, 1);
  TestHostDeviceOutput(&model, 1);
  return 0;
}

int main(int argc, const char **argv) { return QuickStart(argc, argv); }
