/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include "plugin/device/cpu/kernel/ps_roi_pooling_v2_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "abstract/utils.h"
#include "plugin/device/cpu/kernel/atomic_add.h"
#include "mindspore/core/ops/ps_roi_pooling.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int INPUT_NUM = 2;
constexpr int OUTPUT_NUM = 1;
constexpr int INPUT_SHAPE_SIZE = 4;
constexpr int OUTPUT_SHAPE_SIZE = 4;
constexpr int ROI_SHAPE_SIZE = 3;
constexpr int ROI_SECOND_SHAPE = 5;
constexpr size_t kBatchIndex = 0;
constexpr size_t kNumberIndex = 2;
constexpr size_t kInputChannelsIndex = 1;
constexpr size_t kHeightIndex = 2;
constexpr size_t kWidthIndex = 3;
}  // namespace

template <typename T>
void PSROIPoolingCpuKernelMod::PSROIPoolForward(size_t start, size_t end, const T *input, const T *roi_boxes,
                                                T *output_data) {
  auto feature_channels = feature_channels_;
  auto feature_width = width_;
  auto feature_height = height_;
  auto pooled_width = pooled_width_;
  auto pooled_height = pooled_height_;
  auto output_channels = output_channels_;
  auto spatial_scale = (T)spatial_scale_;
  auto group_size = group_size_;
  auto elements_per_roi_box = 5;
  constexpr float zero = 0;

  for (auto index = start; index < end; ++index) {
    int width_offset_n = index % pooled_width;
    int height_offset_n = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_channels;
    int n = index / pooled_width / pooled_height / output_channels;

    const T *offset_rois = roi_boxes + n * elements_per_roi_box;
    int roi_batch_ind = static_cast<int>(offset_rois[0]);

    // floor round not support half
    T roi_start_width = static_cast<T>(round(static_cast<float>(offset_rois[1] * spatial_scale)));
    T roi_start_height = static_cast<T>(round(static_cast<float>(offset_rois[2] * spatial_scale)));
    T roi_end_width = static_cast<T>(round(static_cast<float>(offset_rois[3] * spatial_scale)));
    T roi_end_height = static_cast<T>(round(static_cast<float>(offset_rois[4] * spatial_scale)));

    // Force malformed ROIs to be 1x1
    T roi_width = std::max(roi_end_width - roi_start_width, (T)0.1);  // avoid 0
    T roi_height = std::max(roi_end_height - roi_start_height, (T)0.1);

    T bin_height = (T)(roi_height) / (T)(pooled_height);
    T bin_width = (T)(roi_width) / (T)(pooled_width);

    int pooling_start_x = static_cast<int>(floor(static_cast<T>(height_offset_n) * bin_height));
    int pooling_start_y = static_cast<int>(floor(static_cast<T>(width_offset_n) * bin_width));
    int pooling_end_x = static_cast<int>(ceil(static_cast<T>(height_offset_n + 1) * bin_height));
    int pooling_end_y = static_cast<int>(ceil(static_cast<T>(width_offset_n + 1) * bin_width));

    // Add roi offsets and clip to input boundaries
    pooling_start_x = std::min(std::max(pooling_start_x + static_cast<int>(roi_start_height), 0), feature_height - 1);
    pooling_end_x = std::min(std::max(pooling_end_x + static_cast<int>(roi_start_height), 0), feature_height - 1);
    pooling_start_y = std::min(std::max(pooling_start_y + static_cast<int>(roi_start_width), 0), feature_width - 1);
    pooling_end_y = std::min(std::max(pooling_end_y + static_cast<int>(roi_start_width), 0), feature_width - 1);
    bool is_empty = (pooling_end_x <= pooling_start_x) || (pooling_end_y <= pooling_start_y);

    int gw = width_offset_n;
    int gh = height_offset_n;
    int c = (ctop * group_size + gh) * group_size + gw;

    const T *offset_input = input + (roi_batch_ind * feature_channels + c) * feature_height * feature_width;
    T out_sum = T(zero);
    for (int h = pooling_start_x; h < pooling_end_x; ++h) {
      for (int w = pooling_start_y; w < pooling_end_y; ++w) {
        int bottom_index = h * feature_width + w;
        out_sum += offset_input[bottom_index];
      }
    }
    T bin_area = static_cast<T>((pooling_end_x - pooling_start_x) * (pooling_end_y - pooling_start_y));
    output_data[index] = is_empty ? T(zero) : out_sum / bin_area;
  }
}

bool PSROIPoolingCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(tensor_attr, GetOpSupport()).first;
  if (!is_match) {
    MS_LOG_ERROR << "Can not match kernel based on given attr!";
    return false;
  }

  if (Resize(base_operator, inputs, outputs) == KRET_RESIZE_FAILED) {
    MS_LOG_ERROR << "Resize failed!";
    return false;
  }
  return true;
}

int PSROIPoolingCpuKernelMod::ResizeCheckInputs(const std::vector<KernelTensorPtr> &inputs) {
  input_shape = inputs[0]->GetShapeVector();
  if (input_shape.size() != INPUT_SHAPE_SIZE) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the rank of input[features] should be " << INPUT_SHAPE_SIZE
                  << ", but got the rank of input[features]: " << input_shape.size() << ".";
    return KRET_RESIZE_FAILED;
  }

  rois_shape = inputs[1]->GetShapeVector();
  if (rois_shape.size() != ROI_SHAPE_SIZE) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the rank of input[rois] should be " << ROI_SHAPE_SIZE
                  << ", but got the rank of input[rois]: " << rois_shape.size() << ".";
    return KRET_RESIZE_FAILED;
  }

  if (rois_shape[1] != ROI_SECOND_SHAPE) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input[rois].shape[1] is expected to be " << ROI_SECOND_SHAPE
                  << ", but got " << rois_shape[1] << ".";
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

int PSROIPoolingCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto ret = ResizeCheckInputs(inputs);
  if (ret != KRET_OK) {
    MS_LOG(ERROR) << "Inputs check failed, see above message for details.";
    return ret;
  }

  CHECK_KERNEL_INPUTS_NUM(inputs.size(), INPUT_NUM, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), OUTPUT_NUM, kernel_name_);

  output_shape = outputs[0]->GetShapeVector();
  if (output_shape.size() != OUTPUT_SHAPE_SIZE) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the rank of outputs[0] should be " << OUTPUT_SHAPE_SIZE
                  << ", but got the rank of outputs[0]: " << output_shape.size() << ".";
    return KRET_RESIZE_FAILED;
  }

  data_type_id_ = inputs[0]->GetDtype();

  auto input_size = inputs[0]->GetShapeVector();
  feature_channels_ = static_cast<int32_t>(input_size[kInputChannelsIndex]);
  height_ = static_cast<int32_t>(input_size[kHeightIndex]);
  width_ = static_cast<int32_t>(input_size[kWidthIndex]);

  rois_shape = inputs[1]->GetShapeVector();
  batch_size_ = static_cast<int32_t>(rois_shape[kBatchIndex]);
  rois_num_ = static_cast<int32_t>(rois_shape[kNumberIndex]);
  output_n_ = batch_size_ * rois_num_;

  auto spatial_scale_ptr = base_operator->GetAttr("spatial_scale");
  MS_EXCEPTION_IF_NULL(spatial_scale_ptr);
  spatial_scale_ = GetValue<float>(spatial_scale_ptr);

  auto group_size_ptr = base_operator->GetAttr("group_size");
  MS_EXCEPTION_IF_NULL(group_size_ptr);
  pooled_height_ = LongToInt(GetValue<int64_t>(group_size_ptr));
  pooled_width_ = LongToInt(GetValue<int64_t>(group_size_ptr));
  group_size_ = LongToInt(GetValue<int64_t>(group_size_ptr));

  auto output_dim_ptr = base_operator->GetAttr("output_dim");
  output_channels_ = LongToInt(GetValue<int64_t>(output_dim_ptr));

  for (auto tensor_ptr : inputs) {
    if (tensor_ptr->IsDynamicShape()) return KRET_UNKNOWN_SHAPE;
  }

  for (auto tensor_ptr : outputs) {
    if (tensor_ptr->IsDynamicShape()) return KRET_UNKNOWN_OUT_SHAPE;
  }

  input_shape = inputs[0]->GetShapeVector();
  if (input_shape[1] != group_size_ * group_size_ * output_channels_) {
    MS_LOG_ERROR << "For '" << kernel_name_ << "', input[features].shape[1](" << input_shape[1]
                 << ") should be equal to group_size(" << group_size_ << ") * group_size(" << group_size_
                 << ") * output_dim(" << output_channels_ << "), but it's not true.";
    return KRET_RESIZE_FAILED;
  }

  input_size_list_.clear();
  workspace_size_list_.clear();
  output_size_list_.clear();

  for (auto tensor_ptr : inputs) {
    input_size_list_.push_back(tensor_ptr->GetSizeInBytes());
  }

  for (auto tensor_ptr : outputs) {
    output_size_list_.push_back(tensor_ptr->GetSizeInBytes());
  }

  return KRET_OK;
}

template <typename T>
bool PSROIPoolingCpuKernelMod::PSROIPoolingLauncher(const std::vector<AddressPtr> &inputs,
                                                    const std::vector<AddressPtr> &outputs, const int output_size) {
  auto input_data = reinterpret_cast<T *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(input_data);
  auto rois = reinterpret_cast<T *>(inputs[1]->addr);
  MS_EXCEPTION_IF_NULL(rois);
  auto output_data = reinterpret_cast<T *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(output_data);

  constexpr size_t unit_size = sizeof(T);
  auto memset_task = [&](size_t start, size_t end) {
    (void)memset_s(output_data + start, (end - start) * unit_size, '\0', (end - start) * unit_size);
  };
  ParallelLaunchAutoSearch(memset_task, outputs[0]->size / unit_size, this, &parallel_search_info_);

  auto task = [&](size_t start, size_t end) { return PSROIPoolForward<T>(start, end, input_data, rois, output_data); };
  ParallelLaunchAutoSearch(task, output_size, this, &parallel_search_info_);
  return true;
}

bool PSROIPoolingCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                      const std::vector<AddressPtr> &outputs) {
  auto output_size = output_channels_ * pooled_height_ * pooled_width_ * output_n_;

  if (data_type_id_ == kNumberTypeFloat64) {
    return PSROIPoolingLauncher<double>(inputs, outputs, output_size);
  }

  if (data_type_id_ == kNumberTypeFloat32) {
    return PSROIPoolingLauncher<float>(inputs, outputs, output_size);
  }

  if (data_type_id_ == kNumberTypeFloat16) {
    return PSROIPoolingLauncher<float16>(inputs, outputs, output_size);
  }

  MS_LOG(ERROR) << "For '" << kernel_name_ << "', data_type_id " << data_type_id_ << " is not supported.";
  return false;
}

std::vector<KernelAttr> PSROIPoolingCpuKernelMod::GetOpSupport() {
  return {
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)};
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, PSROIPooling, PSROIPoolingCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
