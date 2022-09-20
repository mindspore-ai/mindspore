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

#include "plugin/device/gpu/kernel/arrays/tensor_copy_gpu_kernel.h"
#include <algorithm>
#include <functional>
#include <map>
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
bool TensorCopyGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL_W_RET_VAL(base_operator, false);
  kernel_name_ = base_operator->name();

  auto input_shapes = inputs.at(kIndex0)->GetShapeVector();
  auto output_shapes = outputs.at(kIndex0)->GetShapeVector();
  auto input_type = inputs.at(kIndex0)->GetDtype();
  auto output_type = outputs.at(kIndex0)->GetDtype();
  if (input_type != output_type) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the type of 'input' and the type of 'output' should be same, but 'input' type is "
                  << input_type << "while 'output' type is " << output_type;
  }
  if (input_shapes != output_shapes) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'input' and the shape of 'output' should be same, but 'input' shape is "
                  << input_shapes << "while 'output' shape is " << output_shapes;
  }

  copy_size_ = GetTypeByte(TypeIdToType(input_type));
  copy_size_ = std::accumulate(input_shapes.begin(), input_shapes.end(), copy_size_, std::multiplies<size_t>());
  InitSizeLists();
  return true;
}
int TensorCopyGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost)) != KRET_OK) {
    return ret;
  }
  return KRET_OK;
}
void TensorCopyGpuKernelMod::InitSizeLists() {
  input_size_list_.push_back(copy_size_);
  output_size_list_.push_back(copy_size_);
}

bool TensorCopyGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                    const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto input = GetDeviceAddress<void>(inputs, 0);
  auto output = GetDeviceAddress<void>(outputs, 0);

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(output, input, copy_size_, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
    "Copy value failed.");
  return true;
}
}  // namespace kernel
}  // namespace mindspore
