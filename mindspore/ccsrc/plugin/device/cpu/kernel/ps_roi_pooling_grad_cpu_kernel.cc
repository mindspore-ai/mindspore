/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/kernel/ps_roi_pooling_grad_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "abstract/utils.h"
#include "plugin/device/cpu/kernel/atomic_add.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kDyOutputDimIndex = 1;
constexpr int kDyHeightIndex = 2;
constexpr int kDyWidthIndex = 3;
constexpr int INPUT_NUM = 2;
constexpr int OUTPUT_NUM = 1;
constexpr int OUT_PUT_SHAPE_SIZE = 4;
constexpr int DY_SHAPE_SIZE = 4;
constexpr int DX_SHAPE_SIZE = 4;
constexpr int ROI_SHAPE_SIZE = 3;
constexpr int ROIS_NUM_INDEX = 2;
}  // namespace

template <typename T>
void PSROIPoolingGradCpuKernelMod::PSROIPoolBackward(size_t start, size_t end, const T *input_diff, T *output_diff,
                                                     T *roi_boxes) const {
  auto output_channels = output_channels_;
  auto pooled_width = pooled_width_;
  auto pooled_height = pooled_height_;
  auto feature_channels = feature_channels_;
  auto feature_width = width_;
  auto feature_height = height_;
  auto spatial_scale = static_cast<T>(spatial_scale_);
  auto rois_num = rois_num_;
  auto elements_per_roi_box = 5;
  constexpr float zero = 0;

  for (auto index = start; index < end; ++index) {
    int width_offset_n = index % pooled_width;
    int height_offset_n = (index / pooled_width) % pooled_height;
    int n = index / pooled_width / pooled_height / output_channels;

    int n_batch = n / rois_num;
    int n_rois_num = n % rois_num;

    // find pooling box index
    T *p_roi_batch_index = roi_boxes + n_batch * (rois_num * elements_per_roi_box) + n_rois_num;
    int roi_batch_index = static_cast<int>(*p_roi_batch_index);

    T *p_roi_start_width = p_roi_batch_index + rois_num;
    T roi_start_width_before_round = (*p_roi_start_width) * spatial_scale;
    T roi_start_width = static_cast<T>(roundf(static_cast<float>(roi_start_width_before_round)));

    T *p_roi_start_height = p_roi_start_width + rois_num;
    T roi_start_height_before_round = (*p_roi_start_height) * spatial_scale;
    T roi_start_height = static_cast<T>(roundf(static_cast<float>(roi_start_height_before_round)));

    T *p_roi_end_width = p_roi_start_height + rois_num;
    T roi_end_width_before_round = (*p_roi_end_width) * spatial_scale;
    T roi_end_width = static_cast<T>(roundf(static_cast<float>(roi_end_width_before_round)));

    T *p_roi_end_height = p_roi_end_width + rois_num;
    T roi_end_height_before_round = (*p_roi_end_height) * spatial_scale;
    T roi_end_height = static_cast<T>(roundf(static_cast<float>(roi_end_height_before_round)));

    // let min roi len and width bigger than 0.1
    T roi_width = std::max(roi_end_width - roi_start_width, static_cast<T>(0.1));
    T roi_height = std::max(roi_end_height - roi_start_height, static_cast<T>(0.1));

    // Compute bin_width and bin_height
    T bin_height = roi_height / static_cast<T>(pooled_height);
    T bin_width = roi_width / static_cast<T>(pooled_width);
    // compute pooling area's position
    int pooling_start_x =
      static_cast<int>(floor(static_cast<float>(static_cast<T>(height_offset_n) * bin_height + roi_start_height)));
    int pooling_start_y =
      static_cast<int>(floor(static_cast<float>(static_cast<T>(width_offset_n) * bin_width + roi_start_width)));
    int pooling_end_x =
      static_cast<int>(ceil(static_cast<float>(static_cast<T>(height_offset_n + 1) * bin_height + roi_start_height)));
    int pooling_end_y =
      static_cast<int>(ceil(static_cast<float>(static_cast<T>(width_offset_n + 1) * bin_width + roi_start_width)));
    // Add roi offsets and clip to input boundaries
    pooling_start_x = std::min(std::max(pooling_start_x, 0), feature_height);
    pooling_end_x = std::min(std::max(pooling_end_x, 0), feature_height);
    pooling_start_y = std::min(std::max(pooling_start_y, 0), feature_width);
    pooling_end_y = std::min(std::max(pooling_end_y, 0), feature_width);

    int c = index % (pooled_height * pooled_width * output_channels);

    T *offset_bottom_diff = output_diff + (roi_batch_index * feature_channels + c) * feature_height * feature_width;
    T bin_area = T((pooling_end_x - pooling_start_x) * (pooling_end_y - pooling_start_y));
    T diff_val = T(zero);
    if (static_cast<float>(bin_area) > zero) {
      diff_val = input_diff[index] / bin_area;
    }

    for (int h = pooling_start_x; h < pooling_end_x; ++h) {
      for (int w = pooling_start_y; w < pooling_end_y; ++w) {
        int bottom_index = h * feature_width + w;
        AtomicAdd(offset_bottom_diff + bottom_index, diff_val);
      }
    }
  }
}

bool PSROIPoolingGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
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

bool PSROIPoolingGradCpuKernelMod::IsSupportedDtype(TypeId type_id) {
  if (type_id == kNumberTypeFloat32 || type_id == kNumberTypeFloat16) {
    return true;
  }
  return false;
}

int PSROIPoolingGradCpuKernelMod::ResizeCheckInputs(const std::vector<KernelTensorPtr> &inputs) const {
  size_t input_num = inputs.size();
  if (input_num != INPUT_NUM) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input number is expected to be " << input_num
                  << ", but PSROIPoolingBackV2GpuKernelMod needs " << INPUT_NUM << " input.";
    return KRET_RESIZE_FAILED;
  }

  auto dy_type = inputs[0]->GetDtype();
  if (!IsSupportedDtype(dy_type)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input[0] is expected to have type_id kNumberTypeFloat32("
                  << kNumberTypeFloat32 << ") or kNumberTypeFloat16(" << kNumberTypeFloat16 << "), but got type_id "
                  << dy_type << ".";
    return KRET_RESIZE_FAILED;
  }

  auto dy_shape = inputs[0]->GetShapeVector();
  if (dy_shape.size() != DY_SHAPE_SIZE) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the rank of input[0] should be " << DY_SHAPE_SIZE
                  << ", but got the rank of input[0]: " << dy_shape.size() << ".";
    return KRET_RESIZE_FAILED;
  }

  auto rois_type = inputs[1]->GetDtype();
  if (!IsSupportedDtype(rois_type)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input[1] is expected to have type_id kNumberTypeFloat32("
                  << kNumberTypeFloat32 << ") or kNumberTypeFloat16(" << kNumberTypeFloat16 << "), but got type_id "
                  << rois_type << ".";
    return KRET_RESIZE_FAILED;
  }

  if (dy_type != rois_type) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', input[1] is expected to have the same type with Input[2], but the type_ids are " << dy_type
                  << ", " << rois_type << ".";
    return KRET_RESIZE_FAILED;
  }

  auto rois_shape = inputs[1]->GetShapeVector();
  if (rois_shape.size() != ROI_SHAPE_SIZE) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the rank of input[1] should be " << ROI_SHAPE_SIZE
                  << ", but got the rank of input[1]: " << rois_shape.size() << ".";
    return KRET_RESIZE_FAILED;
  }

  return KRET_OK;
}

int PSROIPoolingGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto ret = ResizeCheckInputs(inputs);
  if (ret != KRET_OK) {
    MS_LOG(ERROR) << "Inputs check failed, see above message for details.";
    return ret;
  }

  // Get the number of output args
  size_t output_num = outputs.size();
  if (output_num != OUTPUT_NUM) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', output number is expected to be " << output_num << ", but got "
                  << OUTPUT_NUM << " output.";
    return KRET_RESIZE_FAILED;
  }

  auto dx_shape = outputs[0]->GetShapeVector();
  if (dx_shape.size() != DX_SHAPE_SIZE) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the rank of outputs[0] should be " << DX_SHAPE_SIZE
                  << ", but got the rank of outputs[0]: " << dx_shape.size() << ".";
    return KRET_RESIZE_FAILED;
  }

  data_type_id_ = inputs[0]->GetDtype();
  auto rois_shape = inputs[1]->GetShapeVector();
  batch_size_ = static_cast<int32_t>(rois_shape[0]);
  rois_num_ = static_cast<int32_t>(rois_shape[ROIS_NUM_INDEX]);
  output_n_ = batch_size_ * rois_num_;

  auto spatial_scale_ptr = base_operator->GetAttr("spatial_scale");
  MS_EXCEPTION_IF_NULL(spatial_scale_ptr);
  spatial_scale_ = GetValue<float>(spatial_scale_ptr);

  auto input_size_ptr = base_operator->GetAttr("input_size");
  MS_EXCEPTION_IF_NULL(input_size_ptr);
  auto input_size = GetValue<std::vector<int64_t>>(input_size_ptr);
  height_ = static_cast<int32_t>(input_size[0]);
  width_ = static_cast<int32_t>(input_size[1]);

  auto group_size_ptr = base_operator->GetAttr("group_size");
  MS_EXCEPTION_IF_NULL(group_size_ptr);
  pooled_height_ = static_cast<int32_t>(GetValue<int64_t>(group_size_ptr));
  pooled_width_ = static_cast<int32_t>(GetValue<int64_t>(group_size_ptr));
  group_size_ = static_cast<int32_t>(GetValue<int64_t>(group_size_ptr));

  auto output_dim_ptr = base_operator->GetAttr("output_dim");
  output_channels_ = static_cast<int32_t>(GetValue<int64_t>(output_dim_ptr));
  feature_channels_ = output_channels_ * group_size_ * group_size_;

  for (auto tensor_ptr : inputs) {
    if (tensor_ptr->IsDynamicShape()) {
      return KRET_UNKNOWN_SHAPE;
    }
  }

  for (auto tensor_ptr : outputs) {
    if (tensor_ptr->IsDynamicShape()) {
      return KRET_UNKNOWN_OUT_SHAPE;
    }
  }

  auto dy_shape = inputs[0]->GetShapeVector();
  if (dy_shape[0] != batch_size_ * rois_num_) {
    MS_LOG_ERROR << "For '" << kernel_name_ << "', input[0].shape[0](" << dy_shape[0]
                 << ") should be equal to input[1].shape[0](" << rois_shape[0] << ") * input[1].shape[2]("
                 << rois_shape[ROIS_NUM_INDEX] << "), but it's not true.";
    return KRET_RESIZE_FAILED;
  }

  if (dy_shape[kDyOutputDimIndex] != output_channels_) {
    MS_LOG_ERROR << "For '" << kernel_name_ << "', input[0].shape[" << kDyOutputDimIndex << "]("
                 << dy_shape[kDyOutputDimIndex] << ") should be equal to output_dim(" << output_channels_
                 << "), but it's not true.";
    return KRET_RESIZE_FAILED;
  }

  if (dy_shape[kDyHeightIndex] != group_size_) {
    MS_LOG_ERROR << "For '" << kernel_name_ << "', input[0].shape[" << kDyHeightIndex << "]("
                 << dy_shape[kDyHeightIndex] << ") should be equal to group_size(" << group_size_
                 << "), but it's not true.";
    return KRET_RESIZE_FAILED;
  }

  if (dy_shape[kDyWidthIndex] != group_size_) {
    MS_LOG_ERROR << "For '" << kernel_name_ << "', input[0].shape[" << kDyWidthIndex << "](" << dy_shape[kDyWidthIndex]
                 << ") should be equal to group_size(" << group_size_ << "), but it's not true.";
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

bool PSROIPoolingGradCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                          const std::vector<AddressPtr> &outputs) {
  auto output_size = output_channels_ * pooled_height_ * pooled_width_ * output_n_;
  if (data_type_id_ == kNumberTypeFloat32) {
    auto top_diff = static_cast<float *>(inputs[0]->addr);
    MS_EXCEPTION_IF_NULL(top_diff);
    auto rois = static_cast<float *>(inputs[1]->addr);
    MS_EXCEPTION_IF_NULL(rois);
    auto output_diff = static_cast<float *>(outputs[0]->addr);
    MS_EXCEPTION_IF_NULL(output_diff);

    constexpr size_t unit_size = sizeof(float);
    auto memset_task = [&](size_t start, size_t end) {
      (void)memset_s(output_diff + start, (end - start) * unit_size, '\0', (end - start) * unit_size);
    };
    ParallelLaunchAutoSearch(memset_task, outputs[0]->size / unit_size, this, &parallel_search_info_);

    auto task = [&](size_t start, size_t end) {
      return PSROIPoolBackward<float>(start, end, top_diff, output_diff, rois);
    };
    ParallelLaunchAutoSearch(task, output_size, this, &parallel_search_info_);
    return true;
  }

  if (data_type_id_ == kNumberTypeFloat16) {
    auto top_diff = static_cast<float16 *>(inputs[0]->addr);
    MS_EXCEPTION_IF_NULL(top_diff);
    auto rois = static_cast<float16 *>(inputs[1]->addr);
    MS_EXCEPTION_IF_NULL(rois);
    auto output_diff = static_cast<float16 *>(outputs[0]->addr);
    MS_EXCEPTION_IF_NULL(output_diff);

    constexpr size_t unit_size = sizeof(float16);
    auto memset_task = [&](size_t start, size_t end) {
      (void)memset_s(output_diff + start, (end - start) * unit_size, '\0', (end - start) * unit_size);
    };
    ParallelLaunchAutoSearch(memset_task, outputs[0]->size / unit_size, this, &parallel_search_info_);

    auto task = [&](size_t start, size_t end) {
      return PSROIPoolBackward<float16>(start, end, top_diff, output_diff, rois);
    };
    ParallelLaunchAutoSearch(task, output_size, this, &parallel_search_info_);
    return true;
  }

  MS_LOG(ERROR) << "For '" << kernel_name_ << "', data_type_id " << data_type_id_ << " is not supported.";
  return false;
}

std::vector<KernelAttr> PSROIPoolingGradCpuKernelMod::GetOpSupport() {
  return {
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)};
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, PSROIPoolingGrad, PSROIPoolingGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
