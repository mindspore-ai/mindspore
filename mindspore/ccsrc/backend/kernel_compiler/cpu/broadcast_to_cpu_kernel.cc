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

#include "backend/kernel_compiler/cpu/broadcast_to_cpu_kernel.h"
#include "backend/kernel_compiler/cpu/nnacl/errorcode.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBroadcastToInputsNum = 1;
constexpr size_t kBroadcastToOutputsNum = 1;
}  // namespace

template <typename T>
void BroadcastToCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  input_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  output_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  size_t input_shape_size = input_shape_.size();
  size_t output_shape_size = output_shape_.size();
  if (output_shape_size < input_shape_size) {
    MS_LOG(EXCEPTION) << "Cannot broadcast input tensor with shape " << input_shape_
                      << " to  a smaller dimension shape " << output_shape_ << ".";
  }
  if (output_shape_size > MAX_SHAPE_SIZE) {
    MS_LOG(EXCEPTION) << "Cannot broadcast input tensor with shape " << input_shape_ << " to a shape " << output_shape_
                      << " more than 8-D.";
  }
  size_t offset = output_shape_size - input_shape_size;
  for (size_t i = 0; i < input_shape_size; ++i) {
    if (input_shape_[i] != output_shape_[i + offset] && input_shape_[i] != 1) {
      MS_LOG(EXCEPTION) << "Cannot broadcast input tensor with shape " << input_shape_ << " to a shape "
                        << output_shape_ << ".";
    }
  }

  for (size_t i = 0; i < input_shape_size; ++i) {
    shape_info_.input_shape_[i] = SizeToInt(input_shape_[i]);
  }
  for (size_t i = 0; i < output_shape_size; ++i) {
    shape_info_.output_shape_[i] = SizeToInt(output_shape_[i]);
  }
  shape_info_.input_shape_size_ = SizeToInt(input_shape_size);
  shape_info_.output_shape_size_ = SizeToInt(output_shape_size);
}

template <typename T>
bool BroadcastToCPUKernel<T>::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                     const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kBroadcastToInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kBroadcastToOutputsNum, kernel_name_);
  const auto *input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  int status = static_cast<int>(NNACL_OK);
  if constexpr (std::is_same_v<T, bool>) {
    status = BROADCAST_TO(bool, input_addr, &shape_info_, output_addr);
  } else if constexpr (std::is_same_v<T, int>) {
    status = BROADCAST_TO(int, input_addr, &shape_info_, output_addr);
  } else if constexpr (std::is_same_v<T, float>) {
    status = BROADCAST_TO(float, input_addr, &shape_info_, output_addr);
  } else {
    MS_LOG(EXCEPTION) << "Not supported data type for BroadcastTo.";
  }

  if (status != static_cast<int>(NNACL_OK)) {
    MS_LOG(EXCEPTION) << "Broadcast tensor with shape " << input_shape_ << " to shape " << output_shape_
                      << " execute failed, error code: " << status;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
