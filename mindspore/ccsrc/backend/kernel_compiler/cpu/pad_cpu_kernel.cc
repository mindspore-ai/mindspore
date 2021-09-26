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

#include "backend/kernel_compiler/cpu/pad_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kPadInputsNum = 1;
constexpr size_t kPadOutputsNum = 1;
}  // namespace

void PadCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  paddings_ = AnfAlgo::GetNodeAttr<std::vector<std::vector<int64_t>>>(kernel_node, "paddings");
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  std::vector<size_t> input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);

  shape_size_ = input_shape.size();
  if (shape_size_ == 4) {  // shape adjustment from 2d/3d to 4d
  } else if (shape_size_ == 3) {
    (void)input_shape.insert(input_shape.begin(), 1);  // batch padding
    shape_size_ = 4;
  } else if (shape_size_ == 2) {
    (void)input_shape.insert(input_shape.begin(), 2, 1);  // channel padding
    shape_size_ = 4;
  }

  for (size_t i = 0; i < shape_size_; ++i) {
    tensor_size_ *= input_shape[i];
    input_shape_.push_back(input_shape[i]);
  }

  if (paddings_.size() == 4) {  // shape adjustment from 2d/3d to 4d
  } else if (paddings_.size() == 3) {
    (void)paddings_.insert(paddings_.begin(), 1, {0, 0});  // batch padding
  } else if (paddings_.size() == 2) {
    (void)paddings_.insert(paddings_.begin(), 2, {0, 0});  // channel padding
  }

  for (size_t i = 0; i < shape_size_; i++) {
    size_t temp = input_shape[i] + LongToSize((paddings_[i][0] + paddings_[i][1]));  // compute new dim size
    output_size_ *= temp;
    output_shape_.push_back(temp);  // correct new dimension size
  }
}

bool PadCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                          const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kPadInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kPadOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Data type is " << TypeIdLabel(dtype_) << " which is not supported.";
  }
  return true;
}

template <typename T>
void PadCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) const {
  const auto *inputs_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto *outputs_addr = reinterpret_cast<T *>(outputs[0]->addr);

  const int pad_left = paddings_[3][0];
  const int pad_top = paddings_[2][0];
  const int pad_channel_before = paddings_[1][0];
  const int pad_channel_after = paddings_[1][1];
  const T pad_value = T(0);

  const int channels_orig = input_shape_[1];
  const int old_height = input_shape_[2];
  const int old_width = input_shape_[3];
  const int padded_height = output_shape_[2];
  const int padded_width = output_shape_[3];

  for (size_t pos = 0; pos < output_size_; ++pos) {
    int block_num = (SizeToInt(pos) / padded_width) / padded_height;
    const int padded_w = SizeToInt(pos) % padded_width;                    // x coordinate referred to by cur 'pos'
    const int padded_h = (SizeToInt(pos) / padded_width) % padded_height;  // y coordinate referred to by cur 'pos'

    int channels_new = channels_orig + pad_channel_after + pad_channel_before;  // new number of channels from padding
    int channel_num = block_num % channels_new;                                 // current channel
    int batch_item = block_num / channels_new;                                  // current item in batch

    if (padded_h - pad_top < 0 || padded_w - pad_left < 0 || padded_h - pad_top >= old_height ||
        padded_w - pad_left >= old_width || channel_num <= pad_channel_before - 1 ||
        channel_num > channels_orig + pad_channel_before - 1) {
      outputs_addr[pos] = pad_value;
    } else {
      // on a block/x,y position that isn't padding, copy data from the correct block/x,y pos the input
      // calculate from number of blocks of padding (due to channel padding) inserted prior
      int equiv_block_num = block_num - (batch_item * (pad_channel_before + pad_channel_after)) - pad_channel_before;
      outputs_addr[pos] =
        inputs_addr[(equiv_block_num * old_height + padded_h - pad_top) * old_width + padded_w - pad_left];
    }
  }
}
}  // namespace kernel
}  // namespace mindspore
