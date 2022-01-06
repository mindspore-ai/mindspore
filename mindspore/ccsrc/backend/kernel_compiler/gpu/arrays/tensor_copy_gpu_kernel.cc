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

#include "backend/kernel_compiler/gpu/arrays/tensor_copy_gpu_kernel.h"
#include <algorithm>
#include <functional>
#include "backend/kernel_compiler/common_utils.h"

namespace mindspore {
namespace kernel {
const std::vector<size_t> &TensorCopyGPUKernel::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &TensorCopyGPUKernel::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &TensorCopyGPUKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool TensorCopyGPUKernel::Init(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);

  auto input_type = AnfAlgo::GetInputDeviceDataType(node, 0);
  auto input_shapes = AnfAlgo::GetInputDeviceShape(node, 0);
  auto output_type = AnfAlgo::GetOutputDeviceDataType(node, 0);
  auto output_shapes = AnfAlgo::GetOutputDeviceShape(node, 0);
  if ((input_type != output_type) || (input_shapes != output_shapes)) {
    MS_LOG(EXCEPTION) << "The input and output check invalid, kernel: " << node->fullname_with_scope();
  }

  copy_size_ = GetTypeByte(TypeIdToType(input_type));
  copy_size_ = std::accumulate(input_shapes.begin(), input_shapes.end(), copy_size_, std::multiplies<size_t>());
  InitSizeLists();
  return true;
}

void TensorCopyGPUKernel::InitSizeLists() {
  input_size_list_.push_back(copy_size_);
  output_size_list_.push_back(copy_size_);
}

bool TensorCopyGPUKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                 const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto input = GetDeviceAddress<void>(inputs, 0);
  auto output = GetDeviceAddress<void>(outputs, 0);

  CHECK_CUDA_RET_WITH_EXCEPT(
    kernel_node_,
    cudaMemcpyAsync(output, input, copy_size_, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
    "Copy value failed.");

  return true;
}
}  // namespace kernel
}  // namespace mindspore
