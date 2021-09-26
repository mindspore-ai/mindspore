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

#include "backend/kernel_compiler/cpu/mirror_pad_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
// preset size of paddings
constexpr int MAX_PADDINGS = 4;
constexpr int PADDING_SIZE = 2;

// define constants for kernel indexing use
constexpr int BATCH = 0 * PADDING_SIZE;
constexpr int CHANNEL = 1 * PADDING_SIZE;
constexpr int HEIGHT = 2 * PADDING_SIZE;
constexpr int WIDTH = 3 * PADDING_SIZE;
constexpr int TOP = 0;
constexpr int BOTTOM = 1;
constexpr int LEFT = 0;
constexpr int RIGHT = 1;
constexpr size_t kMirrorPadInputsNum = 2;
constexpr size_t kMirrorPadOutputsNum = 1;
}  // namespace

void MirrorPadCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  std::string mode = AnfAlgo::GetNodeAttr<std::string>(kernel_node, "mode");
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  if (mode == "REFLECT") {
    mode_ = 0;
  } else if (mode == "SYMMETRIC") {
    mode_ = 1;
  } else {
    MS_LOG(EXCEPTION) << "For mirror pad, only REFLECT and SYMMETRIC are supported.";
  }

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
    input_shape_.push_back(SizeToLong(input_shape[i]));
  }

  std::vector<size_t> padding_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  num_paddings_ = SizeToLong(padding_shape[0]);

  auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  for (auto x : output_shape) {
    output_size_ *= x;
    output_shape_.push_back(SizeToLong(x));
  }

  int64_t max_width = input_shape_[3];
  int64_t max_height = input_shape_[2];

  if (mode_ == 1) {  // symmetric
    max_width = max_width + (2 * max_width);
    max_height = max_height + (2 * max_height);
  } else {  // reflect
    max_width = max_width + (2 * (max_width - 1));
    max_height = max_height + (2 * (max_height - 1));
  }
  if (output_shape_[(output_shape_.size() - 2)] > max_height ||
      output_shape_[(output_shape_.size() - 2) + 1] > max_width) {
    MS_LOG(ERROR) << "ERROR: Padding value too high for input Tensor on 1 or more dims";
  }
}

void extract_paddings(const int64_t *paddings_arg, int64_t padd_dim, int64_t *extracted_paddings) {
  const int64_t paddings_offset = MAX_PADDINGS - padd_dim;
  for (int64_t i = 0; i < padd_dim; i++) {
    extracted_paddings[(paddings_offset + i) * PADDING_SIZE] = paddings_arg[i * PADDING_SIZE];
    extracted_paddings[(paddings_offset + i) * PADDING_SIZE + 1] = paddings_arg[i * PADDING_SIZE + 1];
  }
}

bool MirrorPadCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                                const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMirrorPadInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMirrorPadOutputsNum, kernel_name_);
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
void MirrorPadCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                      const std::vector<AddressPtr> &outputs) const {
  auto inputs_addr = reinterpret_cast<T *>(inputs[0]->addr);
  int64_t *paddings_arg = reinterpret_cast<int64_t *>(inputs[1]->addr);
  auto outputs_addr = reinterpret_cast<T *>(outputs[0]->addr);

  const int64_t old_batch = input_shape_[0];
  const int64_t old_channel = input_shape_[1];
  const int64_t old_height = input_shape_[2];
  const int64_t old_width = input_shape_[3];
  size_t dim_offset = output_shape_.size() - 2;

  const int64_t padded_height = output_shape_[dim_offset];
  const int64_t padded_width = output_shape_[dim_offset + 1];
  const int64_t padd_dim = num_paddings_;

  const int64_t mode = mode_;

  int64_t paddings[MAX_PADDINGS * PADDING_SIZE];  // local and fixed size to keep in registers
  for (int i = 0; i < MAX_PADDINGS * PADDING_SIZE; i++) {
    paddings[i] = 0;
  }
  extract_paddings(paddings_arg, padd_dim, paddings);
  // Create anchor points for non mirrored data inside new tensor
  int64_t ap1_x = paddings[WIDTH];
  int64_t ap2_x = paddings[WIDTH] + old_width - 1;
  int64_t ap1_y = paddings[HEIGHT];
  int64_t ap2_y = paddings[HEIGHT] + old_height - 1;
  int64_t ap1_channel = paddings[CHANNEL];
  int64_t ap2_channel = paddings[CHANNEL] + old_channel - 1;
  int64_t ap1_batch = paddings[BATCH];
  int64_t ap2_batch = paddings[BATCH] + old_batch - 1;
  int64_t channels_new = old_channel + paddings[CHANNEL] + paddings[CHANNEL + RIGHT];

  for (size_t pos = 0; pos < output_size_; ++pos) {
    int64_t block_num = (SizeToLong(pos) / padded_width) / padded_height;
    // cur position
    const int64_t padded_x = SizeToLong(pos) % padded_width;
    const int64_t padded_y = (SizeToLong(pos) / padded_width) % padded_height;
    const int64_t padded_channel = block_num % channels_new;
    const int64_t padded_batch = block_num / channels_new;

    // data to mirror from in new tensor dims
    int64_t matchval_x_index = padded_x;
    int64_t matchval_y_index = padded_y;
    int64_t matchval_channel_index = padded_channel;
    int64_t matchval_batch_index = padded_batch;
    int64_t equiv_block_num = 0;

    // update matching index in original tensor across all 4 dims
    if ((padded_x < ap1_x) || (padded_x > ap2_x)) {
      int64_t x_dist = (padded_x < ap1_x) ? (ap1_x - padded_x) : (padded_x - ap2_x);
      matchval_x_index = (padded_x < ap1_x) ? (ap1_x + x_dist - mode) : (ap2_x - x_dist + mode);
    }
    if ((padded_y < ap1_y) || (padded_y > ap2_y)) {
      int64_t y_dist = (padded_y < ap1_y) ? (ap1_y - padded_y) : (padded_y - ap2_y);
      matchval_y_index = (padded_y < ap1_y) ? (ap1_y + y_dist - mode) : (ap2_y - y_dist + mode);
    }
    if ((padded_channel < ap1_channel) || (padded_channel > ap2_channel)) {
      int64_t channel_dist =
        (padded_channel < ap1_channel) ? (ap1_channel - padded_channel) : (padded_channel - ap2_channel);
      matchval_channel_index =
        (padded_channel < ap1_channel) ? (ap1_channel + channel_dist - mode) : (ap2_channel - channel_dist + mode);
    }
    if ((padded_batch < ap1_batch) || (padded_batch > ap2_batch)) {
      int64_t batch_dist = (padded_batch < ap1_batch) ? (ap1_batch - padded_batch) : (padded_batch - ap2_batch);
      matchval_batch_index =
        (padded_batch < ap1_batch) ? (ap1_batch + batch_dist - mode) : (ap2_batch - batch_dist + mode);
    }

    // calculate equivalent block in input
    equiv_block_num =
      ((matchval_batch_index - paddings[BATCH]) * old_channel) + (matchval_channel_index - paddings[CHANNEL]);

    // copy data from equiv block and adjusted x and y values in unpadded tensor
    auto pos_index = (equiv_block_num * old_height + matchval_y_index - paddings[HEIGHT]) * old_width +
                     matchval_x_index - paddings[WIDTH];
    outputs_addr[pos] = inputs_addr[pos_index];
  }
}
}  // namespace kernel
}  // namespace mindspore
