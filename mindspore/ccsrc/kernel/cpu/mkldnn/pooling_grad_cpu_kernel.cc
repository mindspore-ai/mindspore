/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "kernel/cpu/mkldnn/pooling_grad_cpu_kernel.h"
#include <string>
#include <utility>
#include <algorithm>
#include "common/utils.h"
#include "kernel/cpu/mkldnn/mkl_kernel_engine.h"
#include "device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void PoolingGradCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  src_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  dst_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  std::vector<int> kernel_sizes = AnfAlgo::GetNodeAttr<std::vector<int>>(kernel_node, KSIZE);
  std::vector<int> strides = AnfAlgo::GetNodeAttr<std::vector<int>>(kernel_node, STRIDES);
  if (kernel_sizes.size() != 4 || strides.size() != 4 || src_shape_.size() != 4 || dst_shape_.size() != 4) {
    MS_LOG(EXCEPTION) << "pooling grad invalid input size";
  }
  std::vector<int> padding_r;
  const std::string pad_mode = AnfAlgo::GetNodeAttr<std::string>(kernel_node, PADDING);
  kernel_size_ = kernel_sizes[3];
  stride_ = strides[3];
  GetPadding(kernel_node, pad_mode, src_shape_, kernel_size_, stride_, &padding_l_, &padding_r);
}

void PoolingGradCPUKernel::RowPoolingGrad(const float *input, float *output, float diff,
                                          const std::vector<std::pair<size_t, size_t>> &box,
                                          std::vector<std::pair<size_t, float>> *row_max_pair) {
  float max_value = 0;
  size_t max_index = box[1].second;
  size_t src_width = src_shape_[3];
  size_t index_start;
  size_t index;
  for (size_t i = box[1].first; i < box[1].second; ++i) {
    if ((*row_max_pair)[i].first == 0) {
      index_start = box[0].first * src_width;
      for (size_t j = box[0].first; j < box[0].second; ++j) {
        index = index_start + i;
        if (input[index] > (*row_max_pair)[i].second || j == box[0].first) {
          (*row_max_pair)[i].second = input[index];
          (*row_max_pair)[i].first = index;
        }
        index_start += src_width;
      }
    }
    if ((*row_max_pair)[i].second > max_value || max_index == box[1].second) {
      max_value = (*row_max_pair)[i].second;
      max_index = i;
    }
  }

  output[(*row_max_pair)[max_index].first] += diff;
}

void PoolingGradCPUKernel::ChannelPoolingGrad(const float *input, const float *diff, float *output) {
  int src_width = SizeToInt(src_shape_[3]);
  int src_height = SizeToInt(src_shape_[2]);
  std::vector<std::pair<size_t, float>> row_max_pair(src_shape_[3]);
  std::vector<std::pair<size_t, size_t>> box(2);
  int h_start = -padding_l_[0];
  size_t diff_index = 0;
  for (size_t h = 0; h < dst_shape_[2]; ++h) {
    box[0].first = IntToSize(std::max(h_start, 0));
    box[0].second = IntToSize(std::min(h_start + kernel_size_, src_height));
    for (size_t w = 0; w < src_shape_[3]; ++w) {
      row_max_pair[w].first = 0;
      row_max_pair[w].second = 0;
    }
    int w_start = -padding_l_[1];
    for (size_t w = 0; w < dst_shape_[3]; ++w) {
      box[1].first = IntToSize(std::max(w_start, 0));
      box[1].second = IntToSize(std::min(w_start + kernel_size_, src_width));
      RowPoolingGrad(input, output, diff[diff_index], box, &row_max_pair);
      diff_index += 1;
      w_start += stride_;
    }
    h_start += stride_;
  }
}

bool PoolingGradCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> & /*workspace*/,
                                  const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() < 3 || outputs.empty()) {
    MS_LOG(EXCEPTION) << "pooling grad error input output size!";
  }

  auto input = reinterpret_cast<float *>(inputs[0]->addr);
  auto diff = reinterpret_cast<float *>(inputs[2]->addr);
  auto output = reinterpret_cast<float *>(outputs[0]->addr);
  auto ret = memset_s(output, outputs[0]->size, 0, outputs[0]->size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "pooling grad memset error";
  }
  size_t src_wh = src_shape_[2] * src_shape_[3];
  size_t dst_wh = dst_shape_[2] * dst_shape_[3];
  for (size_t n = 0; n < src_shape_[0]; ++n) {
    for (size_t c = 0; c < src_shape_[1]; ++c) {
      ChannelPoolingGrad(input, diff, output);
      input = input + src_wh;
      output = output + src_wh;
      diff = diff + dst_wh;
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
