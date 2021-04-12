/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/resize_nearest_neighbor_grad_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "backend/kernel_compiler/common_utils.h"

namespace mindspore {
namespace kernel {
void ResizeNearestNeighborGradCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);
  std::vector<size_t> input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  std::vector<size_t> output_size = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  align_corners_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "align_corners");
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
  batch_size_ = input_shape[0];
  channel_ = input_shape[1];
  in_height_ = input_shape[2];
  in_width_ = input_shape[3];
  out_height_ = output_size[2];
  out_width_ = output_size[3];
  height_scale_ = Scaling(out_height_, in_height_, align_corners_);
  width_scale_ = Scaling(out_width_, in_width_, align_corners_);
}

bool ResizeNearestNeighborGradCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                                const std::vector<kernel::AddressPtr> &,
                                                const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt32) {
    LaunchKernel<int32_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  }
  return true;
}

template <typename T>
void ResizeNearestNeighborGradCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                      const std::vector<AddressPtr> &outputs) {
  auto dloss_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);

  auto ret = memset_s(output_addr, outputs[0]->size, 0, outputs[0]->size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "Output buffer memset failed, ret:" << ret;
  }

  size_t in_hw_size = in_width_ * in_height_;
  size_t out_hw_size = out_width_ * out_height_;

  for (size_t b = 0; b < batch_size_; ++b) {
    for (size_t c = 0; c < channel_; ++c) {
      for (size_t h = 0; h < in_height_; ++h) {
        const size_t out_y = std::min((align_corners_) ? static_cast<size_t>(roundf(h * height_scale_))
                                                       : static_cast<size_t>(floorf(h * height_scale_)),
                                      out_height_ - 1);
        for (size_t w = 0; w < in_width_; ++w) {
          const size_t out_x = std::min((align_corners_) ? static_cast<size_t>(roundf(w * width_scale_))
                                                         : static_cast<size_t>(floorf(w * width_scale_)),
                                        out_width_ - 1);
          output_addr[out_y * out_width_ + out_x] += dloss_addr[h * in_width_ + w];
        }
      }
      output_addr += out_hw_size;
      dloss_addr += in_hw_size;
    }
  }
}

void ResizeNearestNeighborGradCPUKernel::CheckParam(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 1) {
    MS_LOG(EXCEPTION) << "ResizeBilinearGrad needs 1 inputs, but gets " << input_num;
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "ResizeBilinear Gradexpects 1 output, but gets" << output_num;
  }
}
}  // namespace kernel
}  // namespace mindspore
