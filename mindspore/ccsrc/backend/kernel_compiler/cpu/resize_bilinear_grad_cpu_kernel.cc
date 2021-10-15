/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
namespace {
constexpr size_t kResizeBilinearGradInputsNum = 2;
constexpr size_t kResizeBilinearGradOutputNum = 1;
constexpr size_t kResizeBilinearGradInputsDoutShapeSize = 4;
constexpr size_t kResizeBilinearGradInputsXShapeSize = 4;
}  // namespace

void ResizeBilinearGradCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  size_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  align_corners_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "align_corners");
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
  if (shape_.size() < kResizeBilinearGradInputsDoutShapeSize) {
    MS_LOG(EXCEPTION) << "Input dout shape should be " << kResizeBilinearGradInputsDoutShapeSize << ", but got "
                      << shape_.size();
  }
  if (size_.size() < kResizeBilinearGradInputsXShapeSize) {
    MS_LOG(EXCEPTION) << "Input x shape should be " << kResizeBilinearGradInputsXShapeSize << ", but got "
                      << size_.size();
  }

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
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kResizeBilinearGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kResizeBilinearGradOutputNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat16) {
    return LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    return LaunchKernel<float>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported input data type: " << dtype_;
    return false;
  }
}

template <typename T>
bool ResizeBilinearGradCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                               const std::vector<AddressPtr> &outputs) const {
  auto *output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  if (memset_s(output_addr, outputs[0]->size, 0, outputs[0]->size) != EOK) {
    MS_LOG(EXCEPTION) << "Output buffer memset failed.";
    return false;
  }
  float *float_dloss_addr = NULL;
  float *float_output_addr = NULL;
  if (dtype_ == kNumberTypeFloat16) {
    auto *input_addr_T = reinterpret_cast<T *>(inputs[0]->addr);
    size_t input_mem_size = inputs[0]->size / sizeof(T) * sizeof(float);
    float_dloss_addr = reinterpret_cast<float *>(malloc(input_mem_size));
    if (float_dloss_addr == NULL) {
      MS_LOG(ERROR) << "Malloc memory failed.";
      return false;
    }
    for (size_t i = 0; i < ((inputs[0]->size) / sizeof(T)); ++i) {
      float_dloss_addr[i] = static_cast<float>(input_addr_T[i]);
    }

    size_t output_mem_size = outputs[0]->size / sizeof(T) * sizeof(float);
    float_output_addr = reinterpret_cast<float *>(malloc(output_mem_size));
    if (float_output_addr == NULL) {
      free(float_dloss_addr);
      MS_LOG(ERROR) << "Malloc memory failed.";
      return false;
    }
    size_t memset_size = outputs[0]->size / sizeof(T) * sizeof(float);
    if (memset_s(float_output_addr, memset_size, 0, memset_size) != EOK) {
      free(float_dloss_addr);
      free(float_output_addr);
      MS_LOG(EXCEPTION) << "Output buffer memset failed.";
      return false;
    }
  } else if (dtype_ == kNumberTypeFloat32) {
    float_dloss_addr = reinterpret_cast<float *>(inputs[0]->addr);
    float_output_addr = reinterpret_cast<float *>(outputs[0]->addr);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported datatype.";
    return false;
  }

  size_t batch_size = shape_[0];
  size_t channel = shape_[1];
  size_t in_height = shape_[2];
  size_t in_width = shape_[3];
  size_t out_height = size_[2];
  size_t out_width = size_[3];
  size_t out_hw_size = out_height * out_width;
  size_t in_hw_size = in_height * in_width;

  float *cur_dloss_addr = float_dloss_addr;
  float *cur_output_addr = float_output_addr;
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
          cur_output_addr[top_y_index * out_width + left_x_index] +=
            cur_dloss_addr[h * in_width + w] * static_cast<float>(inverse_y_lerp * inverse_x_lerp);
          cur_output_addr[top_y_index * out_width + right_x_index] +=
            cur_dloss_addr[h * in_width + w] * static_cast<float>(inverse_y_lerp * x_lerp);
          cur_output_addr[bottom_y_index * out_width + left_x_index] +=
            cur_dloss_addr[h * in_width + w] * static_cast<float>(y_lerp * inverse_x_lerp);
          cur_output_addr[bottom_y_index * out_width + right_x_index] +=
            cur_dloss_addr[h * in_width + w] * static_cast<float>(y_lerp * x_lerp);

          output_addr[top_y_index * out_width + left_x_index] =
            static_cast<T>(cur_output_addr[top_y_index * out_width + left_x_index]);
          output_addr[top_y_index * out_width + right_x_index] =
            static_cast<T>(cur_output_addr[top_y_index * out_width + right_x_index]);
          output_addr[bottom_y_index * out_width + left_x_index] =
            static_cast<T>(cur_output_addr[bottom_y_index * out_width + left_x_index]);
          output_addr[bottom_y_index * out_width + right_x_index] =
            static_cast<T>(cur_output_addr[bottom_y_index * out_width + right_x_index]);
        }
      }
      output_addr += out_hw_size;
      cur_dloss_addr += in_hw_size;
      cur_output_addr += out_hw_size;
    }
  }
  if (dtype_ == kNumberTypeFloat16) {
    free(float_dloss_addr);
    free(float_output_addr);
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
