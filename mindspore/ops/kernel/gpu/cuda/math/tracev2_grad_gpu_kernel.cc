/**
 * Copyright 2024 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http:www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel/gpu/cuda/math/tracev2_grad_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
constexpr size_t kInputNum = 5;
constexpr size_t kOutputNum = 1;
constexpr size_t kIndexDout = 0;
constexpr size_t kIndexInShape = 1;
constexpr size_t kIndexOffset = 2;
constexpr size_t kIndexAxis1 = 3;
constexpr size_t kIndexAxis2 = 4;
constexpr size_t kIndexDin = 0;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
bool TraceV2GradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  auto prim = primitive_;
  MS_EXCEPTION_IF_NULL(prim);

  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the kernel type should be in [bool, unit8, uint16, uint32, uint64, int8, "
                  << "int16, int32, int64, float16, float32, float64, complex64, complex128], but got: " << kernel_attr
                  << ".";
    return false;
  }
  data_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndexDout).dtype);
  kernel_func_ = func_list_[index].second;
  return true;
}
int TraceV2GradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  data_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndexDout).dtype);
  std::vector<int64_t> din_shape = outputs.at(kIndexDin)->GetShapeVector();
  din_size_ = SizeOf(din_shape);

  workspace_size_list_ = {din_size_ * data_unit_size_};
  output_size_list_ = {din_size_ * data_unit_size_};
  return KRET_OK;
}

template <typename T>
bool TraceV2GradGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &workspace,
                                           const std::vector<KernelTensor *> &outputs) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream_);
  T *dout_addr = GetDeviceAddress<T>(inputs, kIndexDout);
  int64_t *offset_addr = GetDeviceAddress<int64_t>(inputs, kIndexOffset);
  int64_t *axis1_addr = GetDeviceAddress<int64_t>(inputs, kIndexAxis1);
  int64_t *axis2_addr = GetDeviceAddress<int64_t>(inputs, kIndexAxis2);
  T *din_addr = GetDeviceAddress<T>(outputs, kIndexDin);
  MS_EXCEPTION_IF_NULL(dout_addr);
  MS_EXCEPTION_IF_NULL(din_addr);

  T *trans_input_addr = GetDeviceAddress<T>(workspace, kIndex0);
  MS_EXCEPTION_IF_NULL(trans_input_addr);

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemsetAsync(trans_input_addr, 0, din_size_ * data_unit_size_, stream),
                                     "Init padded long array with cudamemset failed");

  int64_t *axis1_h = static_cast<int64_t *>(malloc(sizeof(int64_t)));
  if (axis1_h == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', malloc [axis1_h] memory failed.";
    return false;
  }
  int64_t *axis2_h = static_cast<int64_t *>(malloc(sizeof(int64_t)));
  if (axis2_h == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', malloc [axis2_h] memory failed.";
    free(axis1_h);
    axis1_h = nullptr;
    return false;
  }
  int64_t *offset_h = static_cast<int64_t *>(malloc(sizeof(int64_t)));
  if (offset_h == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', malloc [offset_h] memory failed.";
    free(axis1_h);
    axis1_h = nullptr;
    free(axis2_h);
    axis2_h = nullptr;
    return false;
  }
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemcpyAsync(axis1_h, axis1_addr, sizeof(int64_t), cudaMemcpyDeviceToHost, stream),
    "tracev2_grad cuda copy device to host Fail");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemcpyAsync(axis2_h, axis2_addr, sizeof(int64_t), cudaMemcpyDeviceToHost, stream),
    "tracev2_grad cuda copy device to host Fail");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemcpyAsync(offset_h, offset_addr, sizeof(int64_t), cudaMemcpyDeviceToHost, stream),
    "tracev2_grad cuda copy device to host Fail");
  std::vector<int64_t> din_shape = outputs.at(kIndexDin)->GetShapeVector();
  int64_t din_rank = din_shape.size();
  axis1_h[0] = axis1_h[0] < 0 ? axis1_h[0] + din_rank : axis1_h[0];
  axis2_h[0] = axis2_h[0] < 0 ? axis2_h[0] + din_rank : axis2_h[0];
  int64_t row_size = din_shape[*axis2_h];
  int64_t col_size = din_shape[*axis1_h];
  int64_t mat_size = row_size * col_size;
  int64_t batch_size = din_size_ / mat_size;
  int64_t row_st;
  int64_t col_st;
  if (*offset_h > 0) {
    row_st = 0;
    col_st = *offset_h;
  } else {
    col_st = 0;
    row_st = -*offset_h;
  }
  int64_t diag_count = std::min(row_size - col_st, col_size - row_st);
  Tracev2GradCalc(trans_input_addr, dout_addr, row_st, col_st, diag_count, row_size, mat_size, batch_size, device_id_,
                  stream);
  std::vector<int64_t> input_shape;
  std::vector<int32_t> input_perm;
  int32_t idx = 0;
  for (int64_t i = 0; i < din_rank; i++) {
    if (i == *axis1_h) {
      input_perm.emplace_back(static_cast<int32_t>(din_rank) - kIndex2);
    } else if (i == *axis2_h) {
      input_perm.emplace_back(static_cast<int32_t>(din_rank) - kIndex1);
    } else {
      input_perm.emplace_back(idx);
      idx++;
      input_shape.emplace_back(din_shape[i]);
    }
  }
  input_shape.emplace_back(din_shape[*axis1_h]);
  input_shape.emplace_back(din_shape[*axis2_h]);
  SimplifyTranspose(input_shape, input_perm, &input_shape_, &input_perm_);
  TransposeInfo info;
  info.input_shape = input_shape_;
  info.perm = input_perm_;
  auto status_trans = CalTranspose<T, false>(din_size_, trans_input_addr, info, din_addr, stream);

  CHECK_CUDA_STATUS(status_trans, kernel_name_);
  free(axis1_h);
  free(axis2_h);
  free(offset_h);
  return true;
}

template <typename T>
using Complex = mindspore::utils::Complex<T>;
std::vector<std::pair<KernelAttr, TraceV2GradGpuKernelMod::TraceV2GradFunc>> TraceV2GradGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat16),
   &TraceV2GradGpuKernelMod::LaunchKernel<half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &TraceV2GradGpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   &TraceV2GradGpuKernelMod::LaunchKernel<double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   &TraceV2GradGpuKernelMod::LaunchKernel<double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex64),
   &TraceV2GradGpuKernelMod::LaunchKernel<Complex<float>>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex128),
   &TraceV2GradGpuKernelMod::LaunchKernel<Complex<double>>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex128),
   &TraceV2GradGpuKernelMod::LaunchKernel<Complex<double>>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt8),
   &TraceV2GradGpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt8),
   &TraceV2GradGpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt16),
   &TraceV2GradGpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt16),
   &TraceV2GradGpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt32),
   &TraceV2GradGpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt32),
   &TraceV2GradGpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &TraceV2GradGpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt64),
   &TraceV2GradGpuKernelMod::LaunchKernel<uint64_t>},
};

std::vector<KernelAttr> TraceV2GradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, TraceV2GradFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, TraceV2Grad, TraceV2GradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
