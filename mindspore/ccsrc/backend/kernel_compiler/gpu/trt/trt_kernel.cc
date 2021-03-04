/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/gpu/trt/trt_kernel.h"

#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include "backend/kernel_compiler/gpu/data/dataset_utils.h"
#include "backend/kernel_compiler/gpu/trt/trt_utils.h"

namespace mindspore {
namespace kernel {
const std::vector<size_t> &TrtKernel::GetInputSizeList() const { return input_size_list_; }
const std::vector<size_t> &TrtKernel::GetOutputSizeList() const { return output_size_list_; }
const std::vector<size_t> &TrtKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool TrtKernel::Init(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t i = 0; i < input_num; i++) {
    auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, i);
    auto type_id = AnfAlgo::GetInputDeviceDataType(kernel_node, i);
    size_t unit_size = UnitSizeInBytes(type_id);
    auto size_in_byte = std::accumulate(input_shape.begin(), input_shape.end(), unit_size, std::multiplies<size_t>());
    input_size_list_.push_back(size_in_byte);
  }

  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  for (size_t j = 0; j < output_num; j++) {
    auto output_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, j);
    auto type_id = AnfAlgo::GetOutputDeviceDataType(kernel_node, j);
    size_t unit_size = UnitSizeInBytes(type_id);
    auto size_in_byte = std::accumulate(output_shape.begin(), output_shape.end(), unit_size, std::multiplies<size_t>());
    output_size_list_.push_back(size_in_byte);
  }

  runtime_ = TrtPtr(nvinfer1::createInferRuntime(Singleton<TrtLogger>::Instance()));
  MS_EXCEPTION_IF_NULL(runtime_);
  serialize_ = GetAttr<std::string>(kernel_node, "serialize_model");
  engine_ = TrtPtr(runtime_->deserializeCudaEngine(serialize_.c_str(), serialize_.size(), nullptr));
  MS_EXCEPTION_IF_NULL(engine_);
  if (SizeToInt(input_num + output_num) != engine_->getNbBindings()) {
    MS_LOG(EXCEPTION) << "Inputs and outputs num not match. Got: " << input_num + output_num
                      << ", expect: " << engine_->getNbBindings();
  }

  context_ = TrtPtr(engine_->createExecutionContext());
  MS_EXCEPTION_IF_NULL(context_);
  return true;
}

bool TrtKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                       const std::vector<AddressPtr> &outputs, void *stream) {
  MS_EXCEPTION_IF_NULL(context_);
  std::vector<void *> device_buffer;
  std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(device_buffer),
                 [](const AddressPtr &input) -> void * { return input->addr; });
  std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(device_buffer),
                 [](const AddressPtr &output) -> void * { return output->addr; });
  context_->enqueue(1, device_buffer.data(), reinterpret_cast<cudaStream_t>(stream), nullptr);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
