/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless req_uired by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "plugin/device/gpu/kernel/math/combined_non_max_suppression_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/combined_non_max_suppression_impl.cuh"
#include <utility>
#include <algorithm>
#include <iostream>

namespace mindspore {
namespace kernel {
constexpr int DimSize4 = 4;
void CombinedNonMaxSuppressionGpuKernelMod::ResetResource() noexcept {
  cuda_stream_ = nullptr;
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

void CombinedNonMaxSuppressionGpuKernelMod::InitSizeLists() {
  input_size_list_.push_back(batch_size_ * num_boxes_ * q_ * DimSize4 * sizeof(T));
  input_size_list_.push_back(batch_size_ * num_boxes_ * num_classes_ * sizeof(T));
  input_size_list_.push_back(sizeof(int));
  input_size_list_.push_back(sizeof(int));
  input_size_list_.push_back(sizeof(T));
  input_size_list_.push_back(sizeof(T));
  output_size_list_.push_back(batch_size_ * per_detections_ * DimSize4 * sizeof(T));
  output_size_list_.push_back(batch_size_ * per_detections_ * sizeof(T));
  output_size_list_.push_back(batch_size_ * per_detections_ * sizeof(T));
  output_size_list_.push_back(batch_size_ * sizeof(int));
  workspace_size_list_.push_back(q_ * batch_size_ * num_boxes_ * DimSize4 * sizeof(float));  // new_boxes
  workspace_size_list_.push_back(batch_size_ * num_classes_ * num_boxes_ * sizeof(float));   // new_scores
  workspace_size_list_.push_back(batch_size_ * q_ * num_boxes_ * DimSize4 * sizeof(float));  // boxes_result
  workspace_size_list_.push_back(batch_size_ * num_classes_ * num_boxes_ * sizeof(int));     // index
  workspace_size_list_.push_back(batch_size_ * num_classes_ * num_boxes_ * sizeof(bool));    // sel
  workspace_size_list_.push_back(batch_size_ * num_classes_ * num_boxes_ * num_boxes_ * sizeof(bool));
}

bool CombinedNonMaxSuppressionGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                 const std::vector<KernelTensorPtr> &inputs,
                                                 const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::CombinedNonMaxSuppression>(base_operator);
  pad_per_class_ = kernel_ptr->get_pad_per_class();
  clip_boxes_ = kernel_ptr->get_clip_boxes();
  kernel_name_ = kernel_ptr->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the kernel type should be in "
                  << "[float32,int32], but got: " << kernel_attr;
  }
  is_need_retrieve_output_shape_ = true;
  kernel_func_ = func_list_[index].second;
  return true;
}

int CombinedNonMaxSuppressionGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                  const std::vector<KernelTensorPtr> &inputs,
                                                  const std::vector<KernelTensorPtr> &outputs,
                                                  const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  ResetResource();
  std::vector<size_t> input0_shape = std::vector<size_t>(inputs[kIndex0]->GetDeviceShapeAdaptively().begin(),
                                                         inputs[kIndex0]->GetDeviceShapeAdaptively().end());
  std::vector<size_t> input1_shape = std::vector<size_t>(inputs[kIndex1]->GetDeviceShapeAdaptively().begin(),
                                                         inputs[kIndex1]->GetDeviceShapeAdaptively().end());
  std::vector<size_t> output0_shape = std::vector<size_t>(outputs[kIndex0]->GetDeviceShapeAdaptively().begin(),
                                                          outputs[kIndex0]->GetDeviceShapeAdaptively().end());
  batch_size_ = static_cast<int>(input0_shape[kIndex0]);
  num_boxes_ = static_cast<int>(input0_shape[kIndex1]);
  q_ = static_cast<int>(input0_shape[kIndex2]);
  num_classes_ = static_cast<int>(input1_shape[kIndex2]);
  per_detections_ = static_cast<int>(output0_shape[kIndex1]);

  InitSizeLists();
  return KRET_OK;
}

template <typename T>
bool CombinedNonMaxSuppressionGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                         const std::vector<AddressPtr> &workspace,
                                                         const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);
  T *boxes = GetDeviceAddress<T>(inputs, kIndex0);
  T *scores = GetDeviceAddress<T>(inputs, kIndex1);
  int *max_output_size_per_class = GetDeviceAddress<int>(inputs, kIndex2);
  T *iou_threshold = GetDeviceAddress<T>(inputs, kIndex4);
  T *score_threshold = GetDeviceAddress<T>(inputs, kIndex5);
  T *nmsed_boxes = GetDeviceAddress<T>(outputs, kIndex0);
  T *nmsed_scores = GetDeviceAddress<T>(outputs, kIndex1);
  T *nmsed_classes = GetDeviceAddress<T>(outputs, kIndex2);
  int *valid_detections = GetDeviceAddress<int>(outputs, kIndex3);
  float *new_boxes = GetDeviceAddress<float>(workspace, kIndex0);
  float *new_scores = GetDeviceAddress<float>(workspace, kIndex1);
  float *boxes_result = GetDeviceAddress<float>(workspace, kIndex2);
  int *index = GetDeviceAddress<int>(workspace, kIndex3);
  bool *sel = GetDeviceAddress<bool>(workspace, kIndex4);
  bool *mask = GetDeviceAddress<bool>(workspace, kIndex5);

  auto status = CalSort(scores, index, score_threshold, num_classes_, boxes, new_boxes, new_scores, batch_size_,
                        num_boxes_, boxes_result, q_, sel, device_id_, cuda_stream_);
  CHECK_CUDA_STATUS(status, kernel_name_);
  status = Calnms(batch_size_, num_classes_, iou_threshold, sel, boxes_result, index, q_, num_boxes_,
                  max_output_size_per_class, new_scores, mask, device_id_, cuda_stream_);
  CHECK_CUDA_STATUS(status, kernel_name_);
  status =
    Caloutput(batch_size_, per_detections_, index, new_scores, sel, new_boxes, nmsed_classes, nmsed_scores, nmsed_boxes,
              valid_detections, clip_boxes_, num_classes_, num_boxes_, q_, device_id_, cuda_stream_);
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

void CombinedNonMaxSuppressionGpuKernelMod::SyncOutputShape() {
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream_), "cudaStreamSynchronized failed");
  std::vector<int64_t> shape0 = {batch_size_, per_detections_, DimSize4};
  std::vector<int64_t> shape1 = {batch_size_, per_detections_};
  std::vector<int64_t> shape2 = {batch_size_, per_detections_};
  std::vector<int64_t> shape3 = {batch_size_};
  outputs_[kIndex0]->SetShapeVector(shape0);
  outputs_[kIndex1]->SetShapeVector(shape1);
  outputs_[kIndex2]->SetShapeVector(shape2);
  outputs_[kIndex3]->SetShapeVector(shape3);
}

std::vector<std::pair<KernelAttr, CombinedNonMaxSuppressionGpuKernelMod::CombinedNonMaxSuppressionLaunchFunc>>
  CombinedNonMaxSuppressionGpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt32),
     &CombinedNonMaxSuppressionGpuKernelMod::LaunchKernel<float>},
};

std::vector<KernelAttr> CombinedNonMaxSuppressionGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, CombinedNonMaxSuppressionLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, CombinedNonMaxSuppression, CombinedNonMaxSuppressionGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
