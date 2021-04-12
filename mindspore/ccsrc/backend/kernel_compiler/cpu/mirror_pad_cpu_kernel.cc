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
void MirrorPadCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  std::string mode = AnfAlgo::GetNodeAttr<std::string>(kernel_node, "mode");
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
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
    auto it = input_shape.begin();
    input_shape.insert(it, 1);  // batch padding
    shape_size_ = 4;
  } else if (shape_size_ == 2) {
    auto it = input_shape.begin();
    input_shape.insert(it, 2, 1);  // channel padding
    shape_size_ = 4;
  }

  for (size_t i = 0; i < shape_size_; ++i) {
    tensor_size_ *= input_shape[i];
    input_shape_.push_back(input_shape[i]);
  }

  std::vector<size_t> padding_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  num_paddings_ = padding_shape[0];

  auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  for (auto x : output_shape) {
    output_size_ *= x;
    output_shape_.push_back(x);
  }

  size_t max_width = input_shape_[3];
  size_t max_height = input_shape_[2];

  if (mode_ == 1) {  // symmetric
    max_width = max_width + (2 * max_width);
    max_height = max_height + (2 * max_height);
  } else {  // reflect
    max_width = max_width + (2 * (max_width - 1));
    max_height = max_height + (2 * (max_height - 1));
  }
  if (output_shape_[(output_shape_.size() - 2) + 0] > max_height ||
      output_shape_[(output_shape_.size() - 2) + 1] > max_width) {
    MS_LOG(ERROR) << "ERROR: Padding value too high for input Tensor on 1 or more dims";
  }
}

void extract_paddings(const int64_t *paddings_arg, int padd_dim, int64_t *extracted_paddings) {
  const int paddings_offset = MAX_PADDINGS - padd_dim;
  for (int i = 0; i < padd_dim; i++) {
    extracted_paddings[(paddings_offset + i) * PADDING_SIZE] = paddings_arg[i * PADDING_SIZE];
    extracted_paddings[(paddings_offset + i) * PADDING_SIZE + 1] = paddings_arg[i * PADDING_SIZE + 1];
  }
}

bool MirrorPadCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                const std::vector<kernel::AddressPtr> & /*workspace*/,
                                const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Data type is " << TypeIdLabel(dtype_) << "is not support.";
  }
  return true;
}

template <typename T>
void MirrorPadCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  auto inputs_addr = reinterpret_cast<T *>(inputs[0]->addr);
  int64_t *paddings_arg = reinterpret_cast<int64_t *>(inputs[1]->addr);
  auto outputs_addr = reinterpret_cast<T *>(outputs[0]->addr);

  const int old_batch = input_shape_[0];
  const int old_channel = input_shape_[1];
  const int old_height = input_shape_[2];
  const int old_width = input_shape_[3];
  int dim_offset = output_shape_.size() - 2;

  const int padded_height = output_shape_[dim_offset + 0];
  const int padded_width = output_shape_[dim_offset + 1];
  const int padd_dim = num_paddings_;

  const int mode = mode_;

  int64_t paddings[MAX_PADDINGS * PADDING_SIZE];  // local and fixed size to keep in registers
  for (int i = 0; i < MAX_PADDINGS * PADDING_SIZE; i++) {
    paddings[i] = 0;
  }
  extract_paddings(paddings_arg, padd_dim, paddings);
  // Create anchor points for non mirrored data inside new tensor
  int ap1_x = paddings[WIDTH + LEFT];
  int ap2_x = paddings[WIDTH + LEFT] + old_width - 1;
  int ap1_y = paddings[HEIGHT + TOP];
  int ap2_y = paddings[HEIGHT + TOP] + old_height - 1;
  int ap1_channel = paddings[CHANNEL + LEFT];
  int ap2_channel = paddings[CHANNEL + LEFT] + old_channel - 1;
  int ap1_batch = paddings[BATCH + LEFT];
  int ap2_batch = paddings[BATCH + LEFT] + old_batch - 1;
  int channels_new = old_channel + paddings[CHANNEL + LEFT] + paddings[CHANNEL + RIGHT];

  for (size_t pos = 0; pos < output_size_; ++pos) {
    int block_num = (pos / padded_width) / padded_height;
    // cur position
    const int padded_x = pos % padded_width;
    const int padded_y = (pos / padded_width) % padded_height;
    const int padded_channel = block_num % channels_new;
    const int padded_batch = block_num / channels_new;

    // data to mirror from in new tensor dims
    int matchval_x_index = padded_x;
    int matchval_y_index = padded_y;
    int matchval_channel_index = padded_channel;
    int matchval_batch_index = padded_batch;
    int equiv_block_num = 0;

    // update matching index in original tensor across all 4 dims
    if ((padded_x < ap1_x) || (padded_x > ap2_x)) {
      int x_dist = (padded_x < ap1_x) ? (ap1_x - padded_x) : (padded_x - ap2_x);
      matchval_x_index = (padded_x < ap1_x) ? (ap1_x + x_dist - mode) : (ap2_x - x_dist + mode);
    }
    if ((padded_y < ap1_y) || (padded_y > ap2_y)) {
      int y_dist = (padded_y < ap1_y) ? (ap1_y - padded_y) : (padded_y - ap2_y);
      matchval_y_index = (padded_y < ap1_y) ? (ap1_y + y_dist - mode) : (ap2_y - y_dist + mode);
    }
    if ((padded_channel < ap1_channel) || (padded_channel > ap2_channel)) {
      int channel_dist =
        (padded_channel < ap1_channel) ? (ap1_channel - padded_channel) : (padded_channel - ap2_channel);
      matchval_channel_index =
        (padded_channel < ap1_channel) ? (ap1_channel + channel_dist - mode) : (ap2_channel - channel_dist + mode);
    }
    if ((padded_batch < ap1_batch) || (padded_batch > ap2_batch)) {
      int batch_dist = (padded_batch < ap1_batch) ? (ap1_batch - padded_batch) : (padded_batch - ap2_batch);
      matchval_batch_index =
        (padded_batch < ap1_batch) ? (ap1_batch + batch_dist - mode) : (ap2_batch - batch_dist + mode);
    }

    // calculate equivalent block in input
    equiv_block_num = ((matchval_batch_index - paddings[BATCH + LEFT]) * old_channel) +
                      (matchval_channel_index - paddings[CHANNEL + LEFT]);

    // copy data from equiv block and adjusted x and y values in unpadded tensor
    outputs_addr[pos] =
      inputs_addr[(equiv_block_num * old_height + matchval_y_index - paddings[HEIGHT + TOP]) * old_width +
                  matchval_x_index - paddings[WIDTH + LEFT]];
  }
}

void MirrorPadCPUKernel::CheckParam(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 2) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but MirrorPadCPUKernel needs 2 inputs.";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but MirrorPadCPUKernel needs 1 output.";
  }
}
}  // namespace kernel
}  // namespace mindspore
