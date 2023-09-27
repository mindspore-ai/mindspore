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
bool TrtKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  size_t input_num = inputs.size();
  size_t output_num = outputs.size();
  for (size_t j = 0; j < output_num; j++) {
    const auto &output_shape = outputs[j]->GetShapeVector();
    auto type_id = outputs[j]->dtype_id();
    size_t unit_size = UnitSizeInBytes(type_id);
    auto size_in_byte = unit_size * SizeOf(output_shape);
    output_size_list_.push_back(size_in_byte);
  }

  auto trt_loader = Singleton<device::gpu::TrtLoader>::Instance();
  if (!trt_loader.nvinfer_loaded()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', install Tensor-RT and export LD_LIBRARY_PATH=${TENSORRT_HOME}"
                      << "/lib:$LD_LIBRARY_PATH.";
  }
  runtime_ = trt_loader.CreateInferRuntime(&Singleton<TrtLogger>::Instance());
  MS_EXCEPTION_IF_NULL(runtime_);
  serialize_ = GetValue<std::string>(primitive_->GetAttr("serialize_model"));
  engine_ = TrtPtr(runtime_->deserializeCudaEngine(serialize_.c_str(), serialize_.size(), nullptr));
  MS_EXCEPTION_IF_NULL(engine_);
  if (SizeToInt(input_num + output_num) != engine_->getNbBindings()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs add the number of outputs should be "
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

bool TrtKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                          const std::vector<KernelTensor *> &outputs, void *stream) {
  MS_EXCEPTION_IF_NULL(context_);
  std::vector<void *> device_buffer;
  std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(device_buffer),
                 [](const KernelTensor *input) { return input->device_ptr(); });
  std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(device_buffer),
                 [](const KernelTensor *output) { return output->device_ptr(); });
  return context_->enqueueV2(device_buffer.data(), reinterpret_cast<cudaStream_t>(stream), nullptr);
}
}  // namespace kernel
}  // namespace mindspore
