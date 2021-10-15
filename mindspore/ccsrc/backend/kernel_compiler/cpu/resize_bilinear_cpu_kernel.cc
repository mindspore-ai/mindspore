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

#include "backend/kernel_compiler/cpu/resize_bilinear_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "backend/kernel_compiler/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kResizeBilinearInputsNum = 1;
constexpr size_t kResizeBilinearOutputsNum = 1;
constexpr size_t kResizeBilinearInputsShapeSize = 4;
constexpr size_t kResizeBilinearAttrSize = 2;
}  // namespace

void ResizeBilinearCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  size_ = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, SIZE);
  align_corners_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "align_corners");
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  if (shape_.size() != kResizeBilinearInputsShapeSize) {
    MS_LOG(EXCEPTION) << "Input shape size should be " << kResizeBilinearInputsShapeSize << ", but got "
                      << shape_.size();
  }
  if (size_.size() != kResizeBilinearAttrSize) {
    MS_LOG(EXCEPTION) << "Size attr requires " << kResizeBilinearAttrSize << " elements, but got " << size_.size();
  }
  size_t in_height = shape_[2];
  size_t in_width = shape_[3];
  size_t out_height = size_[0];
  size_t out_width = size_[1];
  height_scale = Scaling(in_height, out_height, align_corners_);
  width_scale = Scaling(in_width, out_width, align_corners_);
}

bool ResizeBilinearCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kResizeBilinearInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kResizeBilinearOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat16) {
    return LaunchKernel<float16, float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    return LaunchKernel<float, float>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported input data type: " << dtype_;
    return false;
  }
}

template <typename T1, typename T2>
bool ResizeBilinearCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &outputs) const {
  auto *output_addr_T2 = reinterpret_cast<T2 *>(outputs[0]->addr);
  float *float_input_addr = NULL;
  float *float_output_addr = NULL;
  if (dtype_ == kNumberTypeFloat16) {
    auto *input_addr_T1 = reinterpret_cast<T1 *>(inputs[0]->addr);
    size_t input_mem_size = inputs[0]->size / sizeof(T1) * sizeof(float);
    float_input_addr = reinterpret_cast<float *>(malloc(input_mem_size));
    if (float_input_addr == NULL) {
      MS_LOG(ERROR) << "Malloc memory failed.";
      return false;
    }
    for (size_t i = 0; i < ((inputs[0]->size) / sizeof(T1)); ++i) {
      float_input_addr[i] = static_cast<float>(input_addr_T1[i]);
    }

    size_t output_mem_size = outputs[0]->size / sizeof(T2) * sizeof(float);
    float_output_addr = reinterpret_cast<float *>(malloc(output_mem_size));
    if (float_output_addr == NULL) {
      free(float_input_addr);
      MS_LOG(ERROR) << "Malloc memory failed.";
      return false;
    }
  } else if (dtype_ == kNumberTypeFloat32) {
    float_input_addr = reinterpret_cast<float *>(inputs[0]->addr);
    float_output_addr = reinterpret_cast<float *>(outputs[0]->addr);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported datatype.";
  }

  size_t batch_size = shape_[0];
  size_t channel = shape_[1];
  size_t in_height = shape_[2];
  size_t in_width = shape_[3];
  size_t out_height = size_[0];
  size_t out_width = size_[1];
  size_t out_hw_size = out_height * out_width;
  size_t in_hw_size = in_height * in_width;
  size_t bhwc_size = in_hw_size * channel * batch_size;

  if (out_height == in_height && out_width == in_width) {
    for (size_t i = 0; i < bhwc_size; ++i) {
      float_output_addr[i] = static_cast<float>(float_input_addr[i]);
    }
  }

  std::vector<CachedInterpolation> ys(out_height + 1);
  std::vector<CachedInterpolation> xs(out_width + 1);
  ComputeInterpolationWeights(out_height, in_height, height_scale, ys.data());
  ComputeInterpolationWeights(out_width, in_width, width_scale, xs.data());

  float *cur_input_addr = float_input_addr;
  float *cur_output_addr = float_output_addr;
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t c = 0; c < channel; ++c) {
      for (size_t h = 0; h < out_height; ++h) {
        const float *ys_input_lower_ptr = cur_input_addr + ys[h].lower * in_width;
        const float *ys_input_upper_ptr = cur_input_addr + ys[h].upper * in_width;
        const float ys_lerp = static_cast<float>(ys[h].lerp);
        for (size_t w = 0; w < out_width; ++w) {
          const size_t xs_lower = xs[w].lower;
          const size_t xs_upper = xs[w].upper;
          const float xs_lerp = static_cast<float>(xs[w].lerp);
          const float top_left(ys_input_lower_ptr[xs_lower]);
          const float top_right(ys_input_lower_ptr[xs_upper]);
          const float bottom_left(ys_input_upper_ptr[xs_lower]);
          const float bottom_right(ys_input_upper_ptr[xs_upper]);
          cur_output_addr[h * out_width + w] =
            ComputeLerp(top_left, top_right, bottom_left, bottom_right, xs_lerp, ys_lerp);
          output_addr_T2[h * out_width + w] = static_cast<T2>(cur_output_addr[h * out_width + w]);
        }
      }
      output_addr_T2 += out_hw_size;
      cur_input_addr += in_hw_size;
      cur_output_addr += out_hw_size;
    }
  }
  if (dtype_ == kNumberTypeFloat16) {
    free(float_input_addr);
    free(float_output_addr);
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
