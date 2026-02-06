/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/arrays/unsorted_segment_min_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
using KernelRunFunc = UnsortedSegmentMinGpuKernelMod::KernelRunFunc;
}  // namespace
#define UNSORTED_SEGMENT_MIN_GPU_REGISTER(T_DT, S_DT, T, S)               \
  KernelAttr().AddInputAttr(T_DT).AddInputAttr(S_DT).AddOutputAttr(T_DT), \
    &UnsortedSegmentMinGpuKernelMod::LaunchKernel<T, S>
#define UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(T_DT, S_DT, DT, T, S)                         \
  KernelAttr().AddInputAttr(T_DT).AddInputAttr(S_DT).AddInputAttr(DT).AddOutputAttr(T_DT), \
    &UnsortedSegmentMinGpuKernelMod::LaunchKernel<T, S>

void UnsortedSegmentMinGpuKernelMod::ResetResource() {
  input_dim0_ = 1;
  input_dim1_ = 1;
  output_dim0_ = 1;
  output_dim1_ = 1;
  output_size_list_.clear();
  workspace_size_list_.clear();
}

void UnsortedSegmentMinGpuKernelMod::InitSizeLists() {
  output_size_list_.push_back(batch_size_ * output_dim0_ * output_dim1_ * data_unit_size_);
}

bool UnsortedSegmentMinGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "Got empty inputs or outputs, which is invalid.";
    return false;
  }

  if (primitive_->HasAttr(ops::kBatchRank)) {
    batch_rank_ = GetValue<int64_t>(primitive_->GetAttr(ops::kBatchRank));
  }

  data_unit_size_ = abstract::TypeIdSize(inputs.at(kIndex0)->dtype_id());
  ids_unit_size_ = abstract::TypeIdSize(inputs.at(kIndex1)->dtype_id());

  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  return true;
}

int UnsortedSegmentMinGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  ResetResource();

  auto input_shapes = inputs[kIndex0]->GetDeviceShapeVector();
  auto ids_shapes = inputs[kIndex1]->GetDeviceShapeVector();
  auto output_shapes = outputs[kIndex0]->GetDeviceShapeVector();

  batch_size_ = 1;
  for (int64_t i = 0; i < batch_rank_; i++) {
    batch_size_ *= input_shapes[i];
  }
  in_stride_ = 1;
  for (size_t i = batch_rank_; i < input_shapes.size(); i++) {
    in_stride_ *= input_shapes[i];
  }
  ids_stride_ = 1;
  for (size_t i = batch_rank_; i < ids_shapes.size(); i++) {
    ids_stride_ *= ids_shapes[i];
  }
  out_stride_ = 1;
  for (size_t i = batch_rank_; i < output_shapes.size(); i++) {
    out_stride_ *= output_shapes[i];
  }

  auto axis = ids_shapes.size();
  for (size_t i = batch_rank_; i < input_shapes.size(); i++) {
    if (i < axis) {
      input_dim0_ *= input_shapes[i];
    } else {
      input_dim1_ *= input_shapes[i];
    }
  }

  output_dim0_ = output_shapes[batch_rank_];
  for (size_t j = batch_rank_ + 1; j < output_shapes.size(); j++) {
    output_dim1_ *= output_shapes[j];
  }

  InitSizeLists();
  return KRET_OK;
}

template <typename T, typename S>
bool UnsortedSegmentMinGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &workspace,
                                                  const std::vector<KernelTensor *> &outputs) {
  T *input_addr = GetDeviceAddress<T>(inputs, kIndex0);
  S *ids_addr = GetDeviceAddress<S>(inputs, kIndex1);
  T *output_addr = GetDeviceAddress<T>(outputs, kIndex0);

  for (int64_t i = 0; i < batch_size_; i++) {
    T *input_batch_addr = input_addr + i * in_stride_;
    S *ids_batch_addr = ids_addr + i * ids_stride_;
    T *output_batch_addr = output_addr + i * out_stride_;
    UnsortedSegmentMin(input_dim0_, input_dim1_, output_dim0_, output_dim1_, input_batch_addr, ids_batch_addr,
                       output_batch_addr, reinterpret_cast<cudaStream_t>(stream_ptr_), device_id_);
  }

  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &UnsortedSegmentMinGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {UNSORTED_SEGMENT_MIN_GPU_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, double, int)},
    {UNSORTED_SEGMENT_MIN_GPU_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, double, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, float, int)},
    {UNSORTED_SEGMENT_MIN_GPU_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, float, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_REGISTER(kNumberTypeFloat16, kNumberTypeInt32, half, int)},
    {UNSORTED_SEGMENT_MIN_GPU_REGISTER(kNumberTypeFloat16, kNumberTypeInt64, half, int64_t)},

    {UNSORTED_SEGMENT_MIN_GPU_REGISTER(kNumberTypeInt64, kNumberTypeInt32, int64_t, int)},
    {UNSORTED_SEGMENT_MIN_GPU_REGISTER(kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_REGISTER(kNumberTypeInt32, kNumberTypeInt32, int32_t, int)},
    {UNSORTED_SEGMENT_MIN_GPU_REGISTER(kNumberTypeInt32, kNumberTypeInt64, int32_t, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_REGISTER(kNumberTypeInt16, kNumberTypeInt32, int16_t, int)},
    {UNSORTED_SEGMENT_MIN_GPU_REGISTER(kNumberTypeInt16, kNumberTypeInt64, int16_t, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_REGISTER(kNumberTypeInt8, kNumberTypeInt32, int8_t, int)},
    {UNSORTED_SEGMENT_MIN_GPU_REGISTER(kNumberTypeInt8, kNumberTypeInt64, int8_t, int64_t)},

    {UNSORTED_SEGMENT_MIN_GPU_REGISTER(kNumberTypeUInt64, kNumberTypeInt32, uint64_t, int)},
    {UNSORTED_SEGMENT_MIN_GPU_REGISTER(kNumberTypeUInt64, kNumberTypeInt64, uint64_t, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_REGISTER(kNumberTypeUInt32, kNumberTypeInt32, uint32_t, int)},
    {UNSORTED_SEGMENT_MIN_GPU_REGISTER(kNumberTypeUInt32, kNumberTypeInt64, uint32_t, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_REGISTER(kNumberTypeUInt16, kNumberTypeInt32, uint16_t, int)},
    {UNSORTED_SEGMENT_MIN_GPU_REGISTER(kNumberTypeUInt16, kNumberTypeInt64, uint16_t, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_REGISTER(kNumberTypeUInt8, kNumberTypeInt32, uint8_t, int)},
    {UNSORTED_SEGMENT_MIN_GPU_REGISTER(kNumberTypeUInt8, kNumberTypeInt64, uint8_t, int64_t)},

    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, kNumberTypeInt32, double, int)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeInt32, double, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeInt32, float, int)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeInt32, float, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeInt32, half, int)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeFloat16, kNumberTypeInt64, kNumberTypeInt32, half, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, kNumberTypeInt64, double, int)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeInt64, double, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeInt64, float, int)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeInt64, float, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeInt64, half, int)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeFloat16, kNumberTypeInt64, kNumberTypeInt64, half, int64_t)},

    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt32, int8_t, int)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeInt8, kNumberTypeInt64, kNumberTypeInt32, int8_t, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt64, int8_t, int)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeInt8, kNumberTypeInt64, kNumberTypeInt64, int8_t, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt32, int16_t, int)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeInt16, kNumberTypeInt64, kNumberTypeInt32, int16_t, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt64, int16_t, int)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeInt16, kNumberTypeInt64, kNumberTypeInt64, int16_t, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int, int)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt32, int, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt64, int, int)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt64, int, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt32, int64_t, int)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt32, int64_t, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt64, int64_t, int)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t)},

    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeInt32, uint8_t, int)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeUInt8, kNumberTypeInt64, kNumberTypeInt32, uint8_t, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeInt64, uint8_t, int)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeUInt8, kNumberTypeInt64, kNumberTypeInt64, uint8_t, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeUInt16, kNumberTypeInt32, kNumberTypeInt32, uint16_t, int)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeUInt16, kNumberTypeInt64, kNumberTypeInt32, uint16_t, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeUInt16, kNumberTypeInt32, kNumberTypeInt64, uint16_t, int)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeUInt16, kNumberTypeInt64, kNumberTypeInt64, uint16_t, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeUInt32, kNumberTypeInt32, kNumberTypeInt32, uint32_t, int)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeUInt32, kNumberTypeInt64, kNumberTypeInt32, uint32_t, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeUInt32, kNumberTypeInt32, kNumberTypeInt64, uint32_t, int)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeUInt32, kNumberTypeInt64, kNumberTypeInt64, uint32_t, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeUInt64, kNumberTypeInt32, kNumberTypeInt32, uint64_t, int)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeUInt64, kNumberTypeInt64, kNumberTypeInt32, uint64_t, int64_t)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeUInt64, kNumberTypeInt32, kNumberTypeInt64, uint64_t, int)},
    {UNSORTED_SEGMENT_MIN_GPU_DY_REGISTER(kNumberTypeUInt64, kNumberTypeInt64, kNumberTypeInt64, uint64_t, int64_t)}};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, UnsortedSegmentMin, UnsortedSegmentMinGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
