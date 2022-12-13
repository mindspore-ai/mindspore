/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/gpu/kernel/trt/trt_kernel.h"

#include <functional>
#include <algorithm>
#include "plugin/device/gpu/kernel/data/dataset_utils.h"
#include "plugin/device/gpu/kernel/trt/trt_utils.h"
#include "kernel/common_utils.h"
#include "plugin/device/gpu/hal/device/trt_loader.h"

namespace mindspore {
namespace kernel {
bool TrtKernelMod::Init(const CNodePtr &kernel_node) {
  auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  kernel_node_ = kernel_node;
  for (size_t i = 0; i < input_num; i++) {
    auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, i);
    auto type_id = AnfAlgo::GetInputDeviceDataType(kernel_node, i);
    size_t unit_size = UnitSizeInBytes(type_id);
    auto size_in_byte = unit_size * SizeOf(input_shape);
    input_size_list_.push_back(size_in_byte);
  }

  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  for (size_t j = 0; j < output_num; j++) {
    auto output_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, j);
    auto type_id = AnfAlgo::GetOutputDeviceDataType(kernel_node, j);
    size_t unit_size = UnitSizeInBytes(type_id);
    auto size_in_byte = unit_size * SizeOf(output_shape);
    output_size_list_.push_back(size_in_byte);
  }

  auto trt_loader = Singleton<device::gpu::TrtLoader>::Instance();
  if (!trt_loader.nvinfer_loaded()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', install Tensor-RT and export LD_LIBRARY_PATH=${TENSORRT_HOME}"
                      << "/lib:$LD_LIBRARY_PATH.";
  }
  runtime_ = trt_loader.CreateInferRuntime(&Singleton<TrtLogger>::Instance());
  MS_EXCEPTION_IF_NULL(runtime_);
  serialize_ = GetAttr<std::string>(kernel_node, "serialize_model");
  engine_ = TrtPtr(runtime_->deserializeCudaEngine(serialize_.c_str(), serialize_.size(), nullptr));
  MS_EXCEPTION_IF_NULL(engine_);
  if (SizeToInt(input_num + output_num) != engine_->getNbBindings()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs add the number of outputs should be "
                      << engine_->getNbBindings() << ", but got " << (input_num + output_num);
  }

  context_ = TrtPtr(engine_->createExecutionContext());
  MS_EXCEPTION_IF_NULL(context_);
  return true;
}

void TrtKernelMod::ReleaseResource() {
  // Make sure destroy trt object before TrtLoader destruct.
  context_.reset();
  engine_.reset();
  runtime_.reset();
}

bool TrtKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                          const std::vector<AddressPtr> &outputs, void *stream) {
  MS_EXCEPTION_IF_NULL(context_);
  std::vector<void *> device_buffer;
  std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(device_buffer),
                 [](const AddressPtr &input) { return input->addr; });
  std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(device_buffer),
                 [](const AddressPtr &output) { return output->addr; });
  return context_->enqueueV2(device_buffer.data(), reinterpret_cast<cudaStream_t>(stream), nullptr);
}
}  // namespace kernel
}  // namespace mindspore
