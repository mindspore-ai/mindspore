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
#include "kernel/gpu/cuda/nn/ps_roi_pooling_v2_gpu_kernel.h"
#include <utility>
#include "mindspore/ops/infer/ps_roi_pooling.h"

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

bool PSROIPoolingV2GpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(tensor_attr, GetOpSupport()).first;
  if (!is_match) {
    MS_LOG(ERROR) << "Can not match kernel based on given attr!";
    return false;
  }
  return true;
}

int PSROIPoolingV2GpuKernelMod::ResizeCheckInputs(const std::vector<KernelTensor *> &inputs) {
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

int PSROIPoolingV2GpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), INPUT_NUM, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), OUTPUT_NUM, kernel_name_);
  auto ret = ResizeCheckInputs(inputs);
  if (ret != KRET_OK) {
    MS_LOG(ERROR) << "Inputs check failed, see above message for details.";
    return ret;
  }

  output_shape = outputs[0]->GetShapeVector();
  if (output_shape.size() != OUTPUT_SHAPE_SIZE) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the rank of outputs[0] should be " << OUTPUT_SHAPE_SIZE
                  << ", but got the rank of outputs[0]: " << output_shape.size() << ".";
    return KRET_RESIZE_FAILED;
  }

  data_type_id_ = inputs[0]->dtype_id();

  auto input_size = inputs[0]->GetShapeVector();
  feature_channels_ = static_cast<int32_t>(input_size[kInputChannelsIndex]);
  height_ = static_cast<int32_t>(input_size[kHeightIndex]);
  width_ = static_cast<int32_t>(input_size[kWidthIndex]);

  rois_shape = inputs[1]->GetShapeVector();
  batch_size_ = static_cast<int32_t>(rois_shape[kBatchIndex]);
  rois_num_ = static_cast<int32_t>(rois_shape[kNumberIndex]);
  output_n_ = batch_size_ * rois_num_;

  spatial_scale_ = GetValue<float>(primitive_->GetAttr("spatial_scale"));

  pooled_height_ = LongToInt(GetValue<int64_t>(primitive_->GetAttr("group_size")));
  pooled_width_ = LongToInt(GetValue<int64_t>(primitive_->GetAttr("group_size")));
  group_size_ = LongToInt(GetValue<int64_t>(primitive_->GetAttr("group_size")));

  output_channels_ = LongToInt(GetValue<int64_t>(primitive_->GetAttr("output_dim")));

  for (auto tensor_ptr : inputs) {
    if (tensor_ptr->IsDynamicShape()) return KRET_UNKNOWN_SHAPE;
  }

  for (auto tensor_ptr : outputs) {
    if (tensor_ptr->IsDynamicShape()) return KRET_UNKNOWN_OUT_SHAPE;
  }

  input_shape = inputs[0]->GetShapeVector();
  if (input_shape[1] != group_size_ * group_size_ * output_channels_) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input[features].shape[1](" << input_shape[1]
                  << ") should be equal to group_size(" << group_size_ << ") * group_size(" << group_size_
                  << ") * output_dim(" << output_channels_ << "), but it's not true.";
    return KRET_RESIZE_FAILED;
  }

  workspace_size_list_.clear();
  output_size_list_.clear();

  auto output_type = outputs[kIndex0]->dtype_id();
  auto output_ele_size = GetTypeByte(TypeIdToType(output_type));
  auto output_ele = batch_size_ * group_size_ * group_size_ * output_channels_;
  output_size_list_.push_back(static_cast<size_t>(output_ele) * output_ele_size);

  return KRET_OK;
}

template <typename T>
bool PSROIPoolingV2GpuKernelMod::PSROIPoolingLauncher(const std::vector<KernelTensor *> &inputs,
                                                      const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  auto input_data = reinterpret_cast<T *>(inputs[kIndex0]->device_ptr());
  MS_EXCEPTION_IF_NULL(input_data);
  auto rois = reinterpret_cast<T *>(inputs[kIndex1]->device_ptr());
  MS_EXCEPTION_IF_NULL(rois);
  MS_EXCEPTION_IF_NULL(rois);
  auto output_data = reinterpret_cast<T *>(outputs[kIndex0]->device_ptr());
  MS_EXCEPTION_IF_NULL(output_data);

  auto status = PSROIPoolForwardV2Launcher(input_data, static_cast<T>(spatial_scale_), output_n_, height_, width_,
                                           feature_channels_, pooled_height_, pooled_width_, rois, group_size_,
                                           output_channels_, output_data, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

bool PSROIPoolingV2GpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &workspace,
                                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (data_type_id_ == kNumberTypeFloat64) {
    return PSROIPoolingLauncher<double>(inputs, outputs, stream_ptr);
  }

  if (data_type_id_ == kNumberTypeFloat32) {
    return PSROIPoolingLauncher<float>(inputs, outputs, stream_ptr);
  }

  if (data_type_id_ == kNumberTypeFloat16) {
    return PSROIPoolingLauncher<half>(inputs, outputs, stream_ptr);
  }

  MS_LOG(ERROR) << "For '" << kernel_name_ << "', data_type_id " << data_type_id_ << " is not supported.";
  return false;
}

std::vector<KernelAttr> PSROIPoolingV2GpuKernelMod::GetOpSupport() {
  return {
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)};
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, PSROIPooling, PSROIPoolingV2GpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
