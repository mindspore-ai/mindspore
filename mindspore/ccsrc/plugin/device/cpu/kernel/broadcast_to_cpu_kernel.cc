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

#include "plugin/device/cpu/kernel/broadcast_to_cpu_kernel.h"
#include "plugin/device/cpu/kernel/nnacl/errorcode.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBroadcastToOutputsNum = 1;
}  // namespace

template <typename T>
void BroadcastToCpuKernelMod<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  input_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  output_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  size_t input_shape_size = input_shape_.size();
  size_t output_shape_size = output_shape_.size();

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
void BroadcastToCpuKernelMod<T>::CheckArgs() {
  size_t input_shape_size = input_shape_.size();
  size_t output_shape_size = output_shape_.size();
  if (output_shape_size < input_shape_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', input tensor 'input_x' and target shape 'shape' can't "
                         "broadcast. The dimension of 'input_x' is "
                      << input_shape_size << ", and the dimension of target shape 'shape' is " << output_shape_size;
  }
  if (output_shape_size > MAX_SHAPE_SIZE) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', input tensor 'input_x' and target shape 'shape' should be "
                         "broadcast, and the dimension of target shape 'shape' should be at most 8. "
                         "But got the dimension of 'input_x': "
                      << input_shape_size << ", and the dimension of target shape 'shape': " << output_shape_size;
  }
  size_t offset = output_shape_size - input_shape_size;
  for (size_t i = 0; i < input_shape_size; ++i) {
    if (input_shape_[i] != output_shape_[i + offset] && input_shape_[i] != 1) {
      MS_LOG(EXCEPTION)
        << "For '" << kernel_name_ << "', when the " << i
        << "'th, the shape of input should be 1 and equal to the shape of output, but got the shape of input: "
        << Vector2Str(input_shape_) << ", and the shape of output: " << Vector2Str(output_shape_);
    }
  }
}

template <typename T>
bool BroadcastToCpuKernelMod<T>::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                        const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kBroadcastToOutputsNum, kernel_name_);
  CheckArgs();
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
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', not supported data type, the dtype of input should be bool, int, or float.";
  }

  if (status != static_cast<int>(NNACL_OK)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', each dimension pair, 'input_x' shape and target shape, "
                         "should be either equal or input is one or the target dimension is -1. "
                         "But got 'input_x' shape: "
                      << Vector2Str(input_shape_) << " and target shape: " << Vector2Str(output_shape_)
                      << ". Error code: " << status;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
