/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include <utility>
#include "kernel/gpu/cuda/nn/ps_roi_pooling_grad_v2_gpu_kernel.h"

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

bool PSROIPoolingBackV2GpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(tensor_attr, GetOpSupport()).first;
  if (!is_match) {
    MS_LOG(ERROR) << "Can not match kernel based on given attr!";
    return false;
  }

  if (Resize(inputs, outputs) == KRET_RESIZE_FAILED) {
    MS_LOG(ERROR) << "Resize failed!";
    return false;
  }
  return true;
}

bool PSROIPoolingBackV2GpuKernelMod::IsSupportedDtype(TypeId type_id) {
  if (type_id == kNumberTypeFloat32 || type_id == kNumberTypeFloat16) {
    return true;
  }
  return false;
}

int PSROIPoolingBackV2GpuKernelMod::ResizeCheckInputs(const std::vector<KernelTensor *> &inputs) {
  size_t input_num = inputs.size();
  if (input_num != INPUT_NUM) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input number is expected to be " << input_num
                  << ", but PSROIPoolingBackV2GpuKernelMod needs " << INPUT_NUM << " input.";
    return KRET_RESIZE_FAILED;
  }

  auto dy_type = inputs[0]->dtype_id();
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

  auto rois_type = inputs[1]->dtype_id();
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

int PSROIPoolingBackV2GpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  constexpr size_t kInputsNum = 2;
  constexpr size_t kOutputsNum = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
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

  data_type_id_ = inputs[0]->dtype_id();
  auto rois_shape = inputs[1]->GetShapeVector();
  batch_size_ = static_cast<int32_t>(rois_shape[0]);
  rois_num_ = static_cast<int32_t>(rois_shape[ROIS_NUM_INDEX]);
  output_n_ = batch_size_ * rois_num_;

  auto spatial_scale_ptr = primitive_->GetAttr("spatial_scale");
  MS_EXCEPTION_IF_NULL(spatial_scale_ptr);
  spatial_scale_ = GetValue<float>(spatial_scale_ptr);

  auto input_size_ptr = primitive_->GetAttr("input_size");
  MS_EXCEPTION_IF_NULL(input_size_ptr);
  auto input_size = GetValue<std::vector<int64_t>>(input_size_ptr);
  height_ = static_cast<int32_t>(input_size[0]);
  width_ = static_cast<int32_t>(input_size[1]);

  auto group_size_ptr = primitive_->GetAttr("group_size");
  MS_EXCEPTION_IF_NULL(group_size_ptr);
  pooled_height_ = static_cast<int32_t>(GetValue<int64_t>(group_size_ptr));
  pooled_width_ = static_cast<int32_t>(GetValue<int64_t>(group_size_ptr));
  group_size_ = static_cast<int32_t>(GetValue<int64_t>(group_size_ptr));

  auto output_dim_ptr = primitive_->GetAttr("output_dim");
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
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input[0].shape[0](" << dy_shape[0]
                  << ") should be equal to input[1].shape[0](" << rois_shape[0] << ") * input[1].shape[2]("
                  << rois_shape[ROIS_NUM_INDEX] << "), but it's not true.";
    return KRET_RESIZE_FAILED;
  }

  if (dy_shape[kDyOutputDimIndex] != output_channels_) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input[0].shape[" << kDyOutputDimIndex << "]("
                  << dy_shape[kDyOutputDimIndex] << ") should be equal to output_dim(" << output_channels_
                  << "), but it's not true.";
    return KRET_RESIZE_FAILED;
  }

  if (dy_shape[kDyHeightIndex] != group_size_) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input[0].shape[" << kDyHeightIndex << "]("
                  << dy_shape[kDyHeightIndex] << ") should be equal to group_size(" << group_size_
                  << "), but it's not true.";
    return KRET_RESIZE_FAILED;
  }

  if (dy_shape[kDyWidthIndex] != group_size_) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input[0].shape[" << kDyWidthIndex << "](" << dy_shape[kDyWidthIndex]
                  << ") should be equal to group_size(" << group_size_ << "), but it's not true.";
    return KRET_RESIZE_FAILED;
  }

  workspace_size_list_.clear();
  output_size_list_.clear();

  for (auto tensor_ptr : outputs) {
    output_size_list_.push_back(tensor_ptr->size());
  }

  return KRET_OK;
}

bool PSROIPoolingBackV2GpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &workspace,
                                            const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (data_type_id_ == kNumberTypeFloat32) {
    auto top_diff = static_cast<float *>(inputs[kIndex0]->device_ptr());
    MS_EXCEPTION_IF_NULL(top_diff);
    auto rois = static_cast<float *>(inputs[kIndex1]->device_ptr());
    MS_EXCEPTION_IF_NULL(rois);
    auto output_diff = static_cast<float *>(outputs[kIndex0]->device_ptr());
    MS_EXCEPTION_IF_NULL(output_diff);
    auto status =
      PSROIPoolBackwardV2Launcher(top_diff, batch_size_, output_n_, static_cast<float>(spatial_scale_),
                                  feature_channels_, height_, width_, pooled_width_, pooled_height_, output_channels_,
                                  output_diff, rois, static_cast<cudaStream_t>(stream_ptr), rois_num_, group_size_);
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }

  if (data_type_id_ == kNumberTypeFloat16) {
    auto top_diff = static_cast<half *>(inputs[0]->device_ptr());
    MS_EXCEPTION_IF_NULL(top_diff);
    auto rois = static_cast<half *>(inputs[1]->device_ptr());
    MS_EXCEPTION_IF_NULL(rois);
    auto output_diff = static_cast<half *>(outputs[0]->device_ptr());
    MS_EXCEPTION_IF_NULL(output_diff);
    auto status =
      PSROIPoolBackwardV2Launcher(top_diff, batch_size_, output_n_, static_cast<half>(spatial_scale_),
                                  feature_channels_, height_, width_, pooled_width_, pooled_height_, output_channels_,
                                  output_diff, rois, static_cast<cudaStream_t>(stream_ptr), rois_num_, group_size_);
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }

  MS_LOG(ERROR) << "For '" << kernel_name_ << "', data_type_id " << data_type_id_ << " is not supported.";
  return false;
}

std::vector<KernelAttr> PSROIPoolingBackV2GpuKernelMod::GetOpSupport() {
  return {
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)};
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, PSROIPoolingGrad, PSROIPoolingBackV2GpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
