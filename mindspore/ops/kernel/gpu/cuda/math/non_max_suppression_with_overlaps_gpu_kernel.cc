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
#include <map>
#include <utility>
#include <functional>

#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "kernel/gpu/cuda_impl/cuda_ops/non_max_suppression_with_overlaps_impl.cuh"
#include "kernel/gpu/cuda/math/non_max_suppression_with_overlaps_gpu_kernel.h"

namespace mindspore {
namespace kernel {
#define NMS_OVERLAPS_GPU_REGISTER(T_DT, T) \
  KernelAttr()                             \
    .AddInputAttr(T_DT)                    \
    .AddInputAttr(T_DT)                    \
    .AddInputAttr(kNumberTypeInt32)        \
    .AddInputAttr(T_DT)                    \
    .AddInputAttr(T_DT)                    \
    .AddOutputAttr(kNumberTypeInt32),      \
    &NMSWithOverlapsFwdGpuKernelMod::LaunchKernel<T>
constexpr int64_t INPUT_DIMS_2 = 2;
constexpr int64_t INPUT_DIMS_1 = 1;
constexpr int64_t INPUT_DIMS_0 = 0;
void NMSWithOverlapsFwdGpuKernelMod::ResetResource() {
  stream_ptr_ = nullptr;
  is_null_input_ = false;
  output_size_list_.clear();
  workspace_size_list_.clear();
}

void NMSWithOverlapsFwdGpuKernelMod::InitSizeLists() {
  output_size_list_.push_back(num_input_ * sizeof(int));

  workspace_size_list_.push_back(ceil_power_2 * sizeof(int));              // index buff
  workspace_size_list_.push_back(num_input_ * num_input_ * sizeof(bool));  // row mask list
  workspace_size_list_.push_back(num_input_ * sizeof(bool));               // sel_box list
  workspace_size_list_.push_back(ceil_power_2 * data_unit_size_);          // up_score list
  workspace_size_list_.push_back(sizeof(int));                             // up_score list
}

bool NMSWithOverlapsFwdGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  data_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  std::vector<int64_t> overlaps_shape = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeVector().begin(),
                                                             inputs.at(kIndex0)->GetDeviceShapeVector().end());
  std::vector<int64_t> scores_shape = std::vector<int64_t>(inputs.at(kIndex1)->GetDeviceShapeVector().begin(),
                                                           inputs.at(kIndex1)->GetDeviceShapeVector().end());
  std::vector<int64_t> max_output_size_shape = std::vector<int64_t>(inputs.at(kIndex2)->GetDeviceShapeVector().begin(),
                                                                    inputs.at(kIndex2)->GetDeviceShapeVector().end());
  std::vector<int64_t> overlap_threshold_shape = std::vector<int64_t>(
    inputs.at(kIndex3)->GetDeviceShapeVector().begin(), inputs.at(kIndex3)->GetDeviceShapeVector().end());
  std::vector<int64_t> score_threshold_shape = std::vector<int64_t>(inputs.at(kIndex4)->GetDeviceShapeVector().begin(),
                                                                    inputs.at(kIndex4)->GetDeviceShapeVector().end());
  int64_t overlaps_dims = overlaps_shape.size();
  int64_t scores_dims = scores_shape.size();
  int64_t max_output_size_dims = max_output_size_shape.size();
  int64_t overlap_threshold_dims = overlap_threshold_shape.size();
  int64_t score_threshold_dims = score_threshold_shape.size();
  if (overlaps_dims != INPUT_DIMS_2) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'overlaps' should be 2-D, but got "
                  << overlaps_dims << "-D.";
    return false;
  }
  if (scores_dims != INPUT_DIMS_1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'scores' should be 1-D, but got " << scores_dims
                  << "-D.";
    return false;
  }
  if (overlaps_shape[kIndex0] != overlaps_shape[kIndex1]) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the two dimensions's length of 'overlaps' should be "
                     "same, but got length "
                  << overlaps_shape[kIndex0] << " and " << overlaps_shape[kIndex1] << ".";
    return false;
  }
  if (overlaps_shape[kIndex0] != scores_shape[kIndex0]) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension's length of 'scores' should be equal the first "
                     "dimensions's length of 'overlaps', but got length "
                  << scores_shape[kIndex0] << ".";
    return false;
  }
  if (max_output_size_dims != INPUT_DIMS_0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'max_output_size' should be 0-D, but got "
                  << max_output_size_dims << "-D.";
    return false;
  }
  if (overlap_threshold_dims != INPUT_DIMS_0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'overlap_threshold' should be 0-D, but got "
                  << overlap_threshold_dims << "-D.";
    return false;
  }
  if (score_threshold_dims != INPUT_DIMS_0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'score_threshold' should be 0-D, but got "
                  << score_threshold_dims << "-D.";
    return false;
  }
  return true;
}

int NMSWithOverlapsFwdGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "Got empty inputs or outputs, which is invalid.";
    return false;
  }
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just
    // return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  std::vector<size_t> shape =
    std::vector<size_t>(inputs[kIndex0]->GetDeviceShapeVector().begin(), inputs[kIndex0]->GetDeviceShapeVector().end());
  is_null_input_ = CHECK_SHAPE_NULL(shape, kernel_name_, "input");
  if (!is_null_input_) {
    num_input_ = shape[kIndex0];  // Get N value in [N,N] data
    ceil_power_2 = NumRoundUpPower2(num_input_);
  }
  InitSizeLists();
  return KRET_OK;
}

template <typename T>
bool NMSWithOverlapsFwdGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &workspace,
                                                  const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  T *overlaps = GetDeviceAddress<T>(inputs, kIndex0);
  T *scores = GetDeviceAddress<T>(inputs, kIndex1);
  int *max_output_size = GetDeviceAddress<int>(inputs, kIndex2);
  T *iou_thershold = GetDeviceAddress<T>(inputs, kIndex3);
  T *score_thershold = GetDeviceAddress<T>(inputs, kIndex4);
  int *index_buff = GetDeviceAddress<int>(workspace, kIndex0);
  bool *row_mask = GetDeviceAddress<bool>(workspace, kIndex1);
  bool *sel_boxes = GetDeviceAddress<bool>(workspace, kIndex2);
  T *up_score = GetDeviceAddress<T>(workspace, kIndex3);
  int *valid_score_num = GetDeviceAddress<int>(workspace, kIndex4);
  int *sel_idx = GetDeviceAddress<int>(outputs, kIndex0);
  int max_output_size_host = 0;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&max_output_size_host, max_output_size, sizeof(int), cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(stream_ptr)),
    "For 'NonMaxSuppressionWithOverlaps', cudaMemcpyAsync value variable failed.");
  if (cudaStreamQuery(reinterpret_cast<cudaStream_t>(stream_ptr)) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr)),
                                       "cuda Stream Sync Failed.");
  }

  if (max_output_size_host < 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', max_output_size must be greater than zero , but got "
                  << max_output_size << ".";
    return false;
  }
  // num_output_ -> valid_num
  auto status = CalSort(num_input_, index_buff, scores, up_score, valid_score_num, score_thershold, device_id_,
                        reinterpret_cast<cudaStream_t>(stream_ptr), &num_output_);
  CHECK_CUDA_STATUS(status, kernel_name_);
  status = CalPreprocess(num_output_, sel_boxes, row_mask, device_id_, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  status = CalNms(num_output_, num_input_, iou_thershold, overlaps, sel_boxes, row_mask, index_buff, device_id_,
                  reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  status = CalPostprocess(num_output_, max_output_size, valid_score_num, score_thershold, index_buff, sel_idx,
                          sel_boxes, device_id_, reinterpret_cast<cudaStream_t>(stream_ptr), &num_output_);
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

void NMSWithOverlapsFwdGpuKernelMod::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                                              const std::vector<KernelTensor *> &outputs) {
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr_)),
                                     "cudaStreamSynchronized failed");

  std::vector<int64_t> shape = outputs[kIndex0]->GetShapeVector();
  shape[kIndex0] = num_output_;
  outputs[kIndex0]->SetShapeVector(std::vector<int64_t>(shape.begin(), shape.end()));
  outputs[kIndex0]->set_size(LongToSize(std::accumulate(
    shape.begin(), shape.end(), UnitSizeInBytes(outputs[kIndex0]->dtype_id()), std::multiplies<int64_t>())));
}

std::vector<std::pair<KernelAttr, NMSWithOverlapsFwdGpuKernelMod::NMSWithOverlapsFunc>>
  NMSWithOverlapsFwdGpuKernelMod::func_list_ = {{NMS_OVERLAPS_GPU_REGISTER(kNumberTypeFloat16, half)},
                                                {NMS_OVERLAPS_GPU_REGISTER(kNumberTypeFloat32, float)},
                                                {NMS_OVERLAPS_GPU_REGISTER(kNumberTypeFloat64, double)}};

std::vector<KernelAttr> NMSWithOverlapsFwdGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, NMSWithOverlapsFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, NonMaxSuppressionWithOverlaps, NMSWithOverlapsFwdGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
