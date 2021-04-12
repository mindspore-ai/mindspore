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

#include "backend/kernel_compiler/cpu/resize_bilinear_grad_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "backend/kernel_compiler/common_utils.h"

namespace mindspore {
namespace kernel {
void ResizeBilinearGradCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);
  shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  size_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  align_corners_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "align_corners");
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);

  size_t in_height = shape_[2];
  size_t in_width = shape_[3];
  size_t out_height = size_[2];
  size_t out_width = size_[3];

  height_scale = Scaling(out_height, in_height, align_corners_);
  width_scale = Scaling(out_width, in_width, align_corners_);
}

bool ResizeBilinearGradCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  }
  return true;
}

template <typename T>
void ResizeBilinearGradCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                               const std::vector<AddressPtr> &outputs) {
  auto dloss_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);

  auto ret = memset_s(output_addr, outputs[0]->size, 0, outputs[0]->size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "Output buffer memset failed, ret:" << ret;
  }

  size_t batch_size = shape_[0];
  size_t channel = shape_[1];
  size_t in_height = shape_[2];
  size_t in_width = shape_[3];
  size_t out_height = size_[2];
  size_t out_width = size_[3];
  size_t out_hw_size = out_height * out_width;
  size_t in_hw_size = in_height * in_width;

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t c = 0; c < channel; ++c) {
      for (size_t h = 0; h < in_height; ++h) {
        const float in_y = static_cast<float>(h) * height_scale;
        const size_t top_y_index = std::max(static_cast<size_t>(floorf(in_y)), static_cast<size_t>(0));
        const size_t bottom_y_index = std::min(static_cast<size_t>(ceilf(in_y)), out_height - 1);
        const float y_lerp = in_y - floorf(in_y);
        const float inverse_y_lerp = 1.0 - y_lerp;
        for (size_t w = 0; w < in_width; ++w) {
          const float in_x = static_cast<float>(w) * width_scale;
          const size_t left_x_index = std::max(static_cast<size_t>(floorf(in_x)), static_cast<size_t>(0));
          const size_t right_x_index = std::min(static_cast<size_t>(ceilf(in_x)), out_width - 1);
          const float x_lerp = in_x - floorf(in_x);
          const float inverse_x_lerp = 1.0 - x_lerp;
          output_addr[top_y_index * out_width + left_x_index] +=
            dloss_addr[h * in_width + w] * T(inverse_y_lerp * inverse_x_lerp);
          output_addr[top_y_index * out_width + right_x_index] +=
            dloss_addr[h * in_width + w] * T(inverse_y_lerp * x_lerp);
          output_addr[bottom_y_index * out_width + left_x_index] +=
            dloss_addr[h * in_width + w] * T(y_lerp * inverse_x_lerp);
          output_addr[bottom_y_index * out_width + right_x_index] += dloss_addr[h * in_width + w] * T(y_lerp * x_lerp);
        }
      }
      output_addr += out_hw_size;
      dloss_addr += in_hw_size;
    }
  }
}

void ResizeBilinearGradCPUKernel::CheckParam(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 2) {
    MS_LOG(EXCEPTION) << "ResizeBilinearGrad needs 2 inputs, but gets " << input_num;
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "ResizeBilinear Gradexpects 1 output, but gets" << output_num;
  }
}
}  // namespace kernel
}  // namespace mindspore
