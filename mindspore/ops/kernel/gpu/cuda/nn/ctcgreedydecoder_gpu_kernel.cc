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
#include "kernel/gpu/cuda/nn/ctcgreedydecoder_gpu_kernel.h"

#include <utility>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <numeric>
#include <iostream>
#include <vector>

#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "mindspore/ops/infer/ctc_greedy_decoder.h"
#include "kernel/gpu/cuda_impl/cuda_ops/ctcgreedydecoder_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t kInputNum = 2;
constexpr size_t kOutputNum = 4;
constexpr size_t kInputsRank = 3;
constexpr size_t kDecodedIndicesRank = 2;
constexpr size_t kSeqLenRank = 1;

void CTCGreedyDecoderGpuKernelMod::ResetResource() {
  stream_ptr_ = nullptr;
  is_null_input_ = false;
  workspace_size_list_.clear();
  output_size_list_.clear();
}

void CTCGreedyDecoderGpuKernelMod::InitSizeLists() {
  max_time_ = inputs_x_shape_[kIndex0];
  batch_size_ = inputs_x_shape_[kIndex1];
  bound_ = inputs_x_shape_[kIndex2];

  workspace_size_list_.push_back(sizeof(int64_t));
  workspace_size_list_.push_back(batch_size_ * sizeof(int64_t));
  workspace_size_list_.push_back(max_time_ * batch_size_ * sizeof(int64_t));

  output_size_list_.push_back(max_time_ * batch_size_ * sizeof(int64_t) * kDecodedIndicesRank);
  output_size_list_.push_back(max_time_ * batch_size_ * sizeof(int64_t));
  output_size_list_.push_back(kDecodedIndicesRank * sizeof(int64_t));
  output_size_list_.push_back(max_time_ * batch_size_ * data_unit_size_);
}

bool CTCGreedyDecoderGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);

  merge_repeated_ = GetValue<bool>(primitive_->GetAttr("merge_repeated"));

  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }

  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }

  kernel_func_ = func_list_[index].second;
  data_unit_size_ = abstract::TypeIdSize(inputs[kIndex0]->dtype_id());
  return true;
}

int CTCGreedyDecoderGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    return KRET_OK;
  }
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }

  ResetResource();

  inputs_x_shape_ = inputs[kIndex0]->GetShapeVector();
  sequence_shape_ = inputs[kIndex1]->GetShapeVector();

  if (inputs_x_shape_.size() != kInputsRank) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', inputs's dim must be 3, but got: " << inputs_x_shape_.size()
                      << ".";
    return KRET_RESIZE_FAILED;
  }

  if (sequence_shape_.size() != kSeqLenRank) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', sequence_length's dims must be 1, but got: " << sequence_shape_.size() << ".";
    return KRET_RESIZE_FAILED;
  }

  if (inputs_x_shape_[1] != sequence_shape_[0]) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', inputs batch_size must be the same with sequence_length batch_size"
                      << ".";
    return KRET_RESIZE_FAILED;
  }

  InitSizeLists();

  if (inputs.size() != kInputNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', Input size list should be " << kInputNum << ", but got "
                  << inputs.size() << ".";
    return KRET_RESIZE_FAILED;
  }

  return KRET_OK;
}

template <typename T>
bool CTCGreedyDecoderGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &workspace,
                                                const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  stream_ptr_ = stream_ptr;
  T *inputs_x = GetDeviceAddress<T>(inputs, kIndex0);
  int *sequence_length = GetDeviceAddress<int>(inputs, kIndex1);
  int64_t *nums_count = GetDeviceAddress<int64_t>(workspace, kIndex1);
  int64_t *decoded_values_temp = GetDeviceAddress<int64_t>(workspace, kIndex2);

  int64_t *decoded_indices = GetDeviceAddress<int64_t>(outputs, kIndex0);
  int64_t *decoded_values = GetDeviceAddress<int64_t>(outputs, kIndex1);
  int64_t *decoded_shape = GetDeviceAddress<int64_t>(outputs, kIndex2);
  T *log_probability = GetDeviceAddress<T>(outputs, kIndex3);

  std::vector<int> seq_host(sequence_shape_[0]);
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemcpyAsync(seq_host.data(), sequence_length, sequence_shape_[0] * sizeof(int32_t), cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(stream_ptr_)),
    "For 'CTCGreedyDecoder', cudaMemcpy beta failed");
  if (cudaStreamQuery(reinterpret_cast<cudaStream_t>(stream_ptr_)) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr_)),
                                       "For 'CTCGreedyDecoder', cudaStreamSyncFailed");
  }
  for (int b = 0; b < sequence_shape_[0]; b++) {
    if (seq_host[b] > static_cast<int>(max_time_)) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', sequence_length[" << b << "] should be less than "
                        << max_time_ << ", but got " << seq_host[b] << ".";
    }
  }

  auto status = CalCTCGreedyDecoder(inputs_x, bound_, max_time_ * batch_size_, batch_size_, decoded_values_temp,
                                    log_probability, device_id_, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);

  status = Calmerge(decoded_values_temp, sequence_length, batch_size_, bound_, merge_repeated_, log_probability,
                    nums_count, device_id_, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);

  status = Calindices(decoded_values_temp, nums_count, batch_size_, decoded_indices, decoded_values, decoded_shape,
                      device_id_, reinterpret_cast<cudaStream_t>(stream_ptr), &element_cnt_);
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

void CTCGreedyDecoderGpuKernelMod::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                                            const std::vector<KernelTensor *> &outputs) {
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr_)),
                                     "cudaStreamSynchronized failed");

  std::vector<int64_t> indices_shape = outputs[kIndex0]->GetShapeVector();
  indices_shape[kIndex0] = element_cnt_;
  outputs[kIndex0]->SetShapeVector(std::vector<int64_t>(indices_shape.begin(), indices_shape.end()));
  outputs[kIndex0]->set_size(
    LongToSize(std::accumulate(indices_shape.begin(), indices_shape.end(),
                               UnitSizeInBytes(outputs[kIndex0]->dtype_id()), std::multiplies<int64_t>())));

  std::vector<int64_t> values_shape = outputs[kIndex1]->GetShapeVector();
  values_shape[kIndex0] = element_cnt_;
  outputs[kIndex1]->SetShapeVector(std::vector<int64_t>(values_shape.begin(), values_shape.end()));
  outputs[kIndex1]->set_size(
    LongToSize(std::accumulate(values_shape.begin(), values_shape.end(), UnitSizeInBytes(outputs[kIndex1]->dtype_id()),
                               std::multiplies<int64_t>())));

  std::vector<int64_t> log_shape = outputs[kIndex3]->GetShapeVector();
  log_shape[kIndex0] = inputs_x_shape_[1];
  outputs[kIndex3]->SetShapeVector(std::vector<int64_t>(log_shape.begin(), log_shape.end()));
  outputs[kIndex3]->set_size(LongToSize(std::accumulate(
    log_shape.begin(), log_shape.end(), UnitSizeInBytes(outputs[kIndex3]->dtype_id()), std::multiplies<int64_t>())));
}

std::vector<std::pair<KernelAttr, CTCGreedyDecoderGpuKernelMod::CTCGreedyDecoderFunc>>
  CTCGreedyDecoderGpuKernelMod::func_list_ = {{KernelAttr()
                                                 .AddInputAttr(kNumberTypeFloat32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddOutputAttr(kNumberTypeInt64)
                                                 .AddOutputAttr(kNumberTypeInt64)
                                                 .AddOutputAttr(kNumberTypeInt64)
                                                 .AddOutputAttr(kNumberTypeFloat32),
                                               &CTCGreedyDecoderGpuKernelMod::LaunchKernel<float>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeFloat64)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddOutputAttr(kNumberTypeInt64)
                                                 .AddOutputAttr(kNumberTypeInt64)
                                                 .AddOutputAttr(kNumberTypeInt64)
                                                 .AddOutputAttr(kNumberTypeFloat64),
                                               &CTCGreedyDecoderGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> CTCGreedyDecoderGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, CTCGreedyDecoderFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, CTCGreedyDecoder, CTCGreedyDecoderGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
