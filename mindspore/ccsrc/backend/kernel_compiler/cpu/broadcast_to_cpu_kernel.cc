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
#include "nnacl/errorcode.h"

namespace mindspore {
namespace kernel {
template <typename T>
void BroadcastToCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
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
  if (inputs.size() != 1 || outputs.size() != 1) {
    MS_LOG(EXCEPTION) << "Wrong number of inputs or outputs!";
  }
  if ((inputs[0] == nullptr) || (inputs[0]->size == 0)) {
    MS_LOG(EXCEPTION) << "Input data is NULL!";
  }
  if ((outputs[0] == nullptr) || (outputs[0]->size == 0)) {
    MS_LOG(EXCEPTION) << "Output data is NULL!";
  }

  const auto input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  int ret = static_cast<int>(NNACL_ERR);
  if constexpr (std::is_same_v<T, bool>) {
    ret = BroadcastTo(bool, input_addr, &shape_info_, output_addr);
  } else if constexpr (std::is_same_v<T, int>) {
    ret = BroadcastTo(int, input_addr, &shape_info_, output_addr);
  } else if constexpr (std::is_same_v<T, float>) {
    ret = BroadcastTo(float, input_addr, &shape_info_, output_addr);
  } else {
    MS_LOG(EXCEPTION) << "Not supported data type for BroadcastTo.";
  }

  if (ret == NNACL_OK) {
    return true;
  }
  MS_LOG(ERROR) << "Broadcast tensor with shape " << input_shape_ << " to shape " << output_shape_
                << " execute failed.";
  return false;
}
}  // namespace kernel
}  // namespace mindspore
