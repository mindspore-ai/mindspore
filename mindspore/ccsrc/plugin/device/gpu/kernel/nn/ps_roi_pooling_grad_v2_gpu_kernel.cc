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
#include "plugin/device/gpu/kernel/nn/ps_roi_pooling_grad_v2_gpu_kernel.h"

namespace mindspore {
namespace kernel {
constexpr int DY_OUTPUT_DIM_INDEX = 1;
constexpr int DY_HEIGHT_INDEX = 2;
constexpr int DY_WIDTH_INDEX = 3;

bool PSROIPoolingBackV2GpuKernelMod::Init(const BaseOperatorPtr &base_operator,
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

bool PSROIPoolingBackV2GpuKernelMod::IsSupportedDtype(TypeId type_id) {
  if (type_id == kNumberTypeFloat32 || type_id == kNumberTypeFloat16) {
    return true;
  }
  return false;
}

int PSROIPoolingBackV2GpuKernelMod::ResizeCheckInputs(const std::vector<KernelTensorPtr> &inputs) {
  size_t input_num = inputs.size();
  if (input_num != INPUT_NUM) {
    MS_LOG(ERROR) << "Input number is " << input_num << ", but PSROIPoolingBackV2GpuKernelMod needs " << INPUT_NUM
                  << " input.";
    return KRET_RESIZE_FAILED;
  }

  auto dy_type = inputs[0]->GetDtype();
  if (!IsSupportedDtype(dy_type)) {
    MS_LOG(ERROR) << "Input[0] is expected to have type_id kNumberTypeFloat32(" << kNumberTypeFloat32
                  << ") or kNumberTypeFloat16(" << kNumberTypeFloat16 << "), but get type_id " << dy_type << ".";
    return KRET_RESIZE_FAILED;
  }

  auto dy_shape = inputs[0]->GetShapeVector();
  if (dy_shape.size() != DY_SHAPE_SIZE) {
    MS_LOG(ERROR) << "The rank of input[0] should be " << DY_SHAPE_SIZE
                  << ", but got the rank of input[0]: " << dy_shape.size() << ".";
    return KRET_RESIZE_FAILED;
  }

  auto rois_type = inputs[1]->GetDtype();
  if (!IsSupportedDtype(rois_type)) {
    MS_LOG(ERROR) << "Input[1] is expected to have type_id kNumberTypeFloat32(" << kNumberTypeFloat32
                  << ") or kNumberTypeFloat16(" << kNumberTypeFloat16 << "), but get type_id " << rois_type << ".";
    return KRET_RESIZE_FAILED;
  }

  if (dy_type != rois_type) {
    MS_LOG(ERROR) << "Input[1] is expected to have the same type with Input[2], but the type_ids are " << dy_type
                  << ", " << rois_type << ".";
    return KRET_RESIZE_FAILED;
  }

  auto rois_shape = inputs[1]->GetShapeVector();
  if (rois_shape.size() != ROI_SHAPE_SIZE) {
    MS_LOG(ERROR) << "The rank of input[1] should be " << ROI_SHAPE_SIZE
                  << ", but got the rank of input[1]: " << rois_shape.size() << ".";
    return KRET_RESIZE_FAILED;
  }

  return KRET_OK;
}

int PSROIPoolingBackV2GpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs,
                                           const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto ret = ResizeCheckInputs(inputs);
  if (ret != KRET_OK) {
    MS_LOG(ERROR) << "Inputs check failed";
    return ret;
  }

  // Get the number of output args
  size_t output_num = outputs.size();
  if (output_num != OUTPUT_NUM) {
    MS_LOG(ERROR) << "Output number is " << output_num << ", but PSROIPoolingBackV2GpuKernelMod needs " << OUTPUT_NUM
                  << " output.";
    return KRET_RESIZE_FAILED;
  }

  auto dx_shape = outputs[0]->GetShapeVector();
  if (dx_shape.size() != DX_SHAPE_SIZE) {
    MS_LOG(ERROR) << "The rank of outputs[0] should be " << DX_SHAPE_SIZE
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
    if (tensor_ptr->IsDynamicShape()) return KRET_UNKNOWN_SHAPE;
  }

  for (auto tensor_ptr : outputs) {
    if (tensor_ptr->IsDynamicShape()) return KRET_UNKNOWN_OUT_SHAPE;
  }

  auto dy_shape = inputs[0]->GetShapeVector();
  if (dy_shape[0] != batch_size_ * rois_num_) {
    MS_LOG_ERROR << "Shape mismatch! We expect input[0].shape[0](" << dy_shape[0] << ") == input[1].shape[0]("
                 << rois_shape[0] << ") * input[1].shape[2](" << rois_shape[ROIS_NUM_INDEX] << "), but it's not true.";
    return KRET_RESIZE_FAILED;
  }

  if (dy_shape[DY_OUTPUT_DIM_INDEX] != output_channels_) {
    MS_LOG_ERROR << "Shape mismatch. We expect input[0].shape[" << DY_OUTPUT_DIM_INDEX << "]("
                 << dy_shape[DY_OUTPUT_DIM_INDEX] << ") == output_dim(" << output_channels_ << "), but it's not true.";
    return KRET_RESIZE_FAILED;
  }

  if (dy_shape[DY_HEIGHT_INDEX] != group_size_) {
    MS_LOG_ERROR << "Shape mismatch. We expect input[0].shape[" << DY_HEIGHT_INDEX << "](" << dy_shape[DY_HEIGHT_INDEX]
                 << ") == group_size(" << group_size_ << "), but it's not true.";
    return KRET_RESIZE_FAILED;
  }

  if (dy_shape[DY_WIDTH_INDEX] != group_size_) {
    MS_LOG_ERROR << "Shape mismatch. We expect input[0].shape[" << DY_WIDTH_INDEX << "](" << dy_shape[DY_WIDTH_INDEX]
                 << ") == group_size(" << group_size_ << "), but it's not true.";
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

bool PSROIPoolingBackV2GpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &workspace,
                                            const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (data_type_id_ == kNumberTypeFloat32) {
    auto top_diff = reinterpret_cast<float *>(inputs[0]->addr);
    MS_EXCEPTION_IF_NULL(top_diff);
    auto rois = reinterpret_cast<float *>(inputs[1]->addr);
    MS_EXCEPTION_IF_NULL(rois);
    auto output_diff = reinterpret_cast<float *>(outputs[0]->addr);
    MS_EXCEPTION_IF_NULL(output_diff);
    PSROIPoolBackwardV2Launcher(top_diff, batch_size_, output_n_, static_cast<float>(spatial_scale_), feature_channels_,
                                height_, width_, pooled_width_, pooled_height_, output_channels_, output_diff, rois,
                                reinterpret_cast<cudaStream_t>(stream_ptr), rois_num_, group_size_);
    return true;
  }

  if (data_type_id_ == kNumberTypeFloat16) {
    auto top_diff = reinterpret_cast<half *>(inputs[0]->addr);
    MS_EXCEPTION_IF_NULL(top_diff);
    auto rois = reinterpret_cast<half *>(inputs[1]->addr);
    MS_EXCEPTION_IF_NULL(rois);
    auto output_diff = reinterpret_cast<half *>(outputs[0]->addr);
    MS_EXCEPTION_IF_NULL(output_diff);
    PSROIPoolBackwardV2Launcher(top_diff, batch_size_, output_n_, static_cast<half>(spatial_scale_), feature_channels_,
                                height_, width_, pooled_width_, pooled_height_, output_channels_, output_diff, rois,
                                reinterpret_cast<cudaStream_t>(stream_ptr), rois_num_, group_size_);
    return true;
  }

  MS_LOG(ERROR) << "Data_type_id " << data_type_id_ << " is not supported.";
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
