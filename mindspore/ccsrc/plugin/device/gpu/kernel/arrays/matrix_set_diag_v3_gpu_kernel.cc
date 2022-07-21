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

#include "plugin/device/gpu/kernel/arrays/matrix_set_diag_v3_gpu_kernel.h"
#include <algorithm>
#include <tuple>
#include <map>
#include <functional>
#include "kernel/common_utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/matrix_set_diag_impl.cuh"
#include "mindspore/core/ops/matrix_set_diag_v3.h"

namespace mindspore {
namespace kernel {
bool MatrixSetDiagV3GpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (kernel_name_ != prim::kPrimMatrixSetDiagV3->name()) {
    MS_LOG(ERROR) << "For 'MatrixSetDiagV3GpuKernelMod', it should get MatrixSetDiagV3 but got invalid kernel name: "
                  << kernel_name_;
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  auto kernel_ptr = std::make_shared<ops::MatrixSetDiagV3>(base_operator->GetPrim());
  auto alignment = kernel_ptr->get_align();
  alignment_ = GetAlignments(alignment);
  return true;
}

int MatrixSetDiagV3GpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  ResetResource();
  auto origin_input_shape = inputs.at(kIndex0)->GetShapeVector();
  std::vector<size_t> input_shape = LongVecToSizeVec(origin_input_shape);
  auto origin_diag_shape = inputs.at(kIndex1)->GetShapeVector();
  std::vector<size_t> diag_shape = LongVecToSizeVec(origin_diag_shape);
  auto origin_k_shape = inputs.at(kIndex2)->GetShapeVector();
  std::vector<size_t> k_shape = LongVecToSizeVec(origin_k_shape);
  auto origin_output_shape = outputs.at(kIndex0)->GetShapeVector();
  std::vector<size_t> output_shape = LongVecToSizeVec(origin_output_shape);
  // For k_shape maybe empty, just ignore it's checking.
  is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input_shape") ||
                   CHECK_SHAPE_NULL(diag_shape, kernel_name_, "diag_shape") ||
                   CHECK_SHAPE_NULL(output_shape, kernel_name_, "output_shape");
  if (is_null_input_) {
    return KRET_OK;
  }

  is_single_diag_ = k_shape.empty();
  constexpr int last_2d_dim = 2;
  constexpr int last_1d_dim = 1;
  // The Correctness of input_shape, diag_shape, k_shape, output_shape have been ensured by c++ primitive infer
  // function. So jut deal with is ok.
  int input_rank = SizeToInt(input_shape.size());
  for (size_t i = 0; i < IntToSize(input_rank - last_2d_dim); ++i) {
    outer_batch_ *= SizeToInt(input_shape.at(i));
  }
  inner_rows_ = SizeToInt(input_shape.at(IntToSize(input_rank - last_2d_dim)));
  inner_cols_ = SizeToInt(input_shape.at(IntToSize(input_rank - last_1d_dim)));
  expected_num_diags_ =
    SizeToInt(diag_shape.size()) == input_rank ? SizeToInt(diag_shape.at(IntToSize(input_rank - last_2d_dim))) : 1;
  diagonal_count_ = std::accumulate(diag_shape.begin(), diag_shape.end(), size_t(1), std::multiplies<size_t>());
  k_count_ = std::accumulate(k_shape.begin(), k_shape.end(), size_t(1), std::multiplies<size_t>());
  k_count_ = k_count_ == 0 ? 1 : k_count_;
  return KRET_OK;
}

template <typename T>
bool MatrixSetDiagV3GpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                               const std::vector<kernel::AddressPtr> &outputs) {
  if (is_null_input_) {
    return true;
  }
  auto input = inputs.at(kIndex0);
  auto diag = inputs.at(kIndex1);
  auto k = inputs.at(kIndex2);
  auto output = outputs.at(kIndex0);
  auto input_device_address = reinterpret_cast<T *>(input->addr);
  auto diag_device_address = reinterpret_cast<T *>(diag->addr);
  auto k_device_address = reinterpret_cast<int *>(k->addr);
  auto output_device_address = reinterpret_cast<T *>(output->addr);

  std::vector<int> host_k_vec;
  size_t k_length = k->size / sizeof(int);
  host_k_vec.resize(k_length);
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemcpyAsync(host_k_vec.data(), k_device_address, k->size, cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "MatrixSetDiagV3GpuKernelMod cuda copy device to host Fail");

  lower_ = host_k_vec.at(kIndex0);
  upper_ = host_k_vec.at(kIndex0);
  if (!is_single_diag_ && k_length > 1) {
    upper_ = host_k_vec.at(kIndex1);
    is_single_diag_ = (lower_ == upper_);
  }

  if (lower_ <= -inner_rows_ || lower_ >= inner_cols_) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of diag_region's lower_diag_index is invalid, which must be between "
                  << -inner_rows_ << " and " << inner_cols_;
    return false;
  }
  if (upper_ <= -inner_rows_ || upper_ >= inner_cols_) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of diag_region's upper_diag_index is invalid, which must be between "
                  << -inner_rows_ << " and " << inner_cols_;
    return false;
  }
  if (lower_ > upper_) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of diag_region's lower_diag_index_ must less than upper_diag_index, but got "
                  << lower_ << " vs " << upper_;
    return false;
  }
  num_diags_ = upper_ - lower_ + 1;
  if (!is_single_diag_ && expected_num_diags_ != num_diags_) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of diag_region's lower_diag_index and upper_diag_index are not consistent with "
                     "input shape. Please check input's valid.";
    return false;
  }
  max_diag_len_ = std::min(inner_rows_ + std::min(upper_, 0), inner_cols_ - std::max(lower_, 0));

  // Copy input to output first, then set diagonal value to output.
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemcpy(output_device_address, input_device_address, input->size, cudaMemcpyDeviceToDevice),
    "MatrixSetDiagV3GpuKernelMod cuda copy input to output Fail");
  bool right_align_super_diagonal = (alignment_.first == MatrixDiag::RIGHT);
  bool right_align_sub_diagonal = (alignment_.second == MatrixDiag::RIGHT);
  MatrixSetDiag(outer_batch_, inner_rows_, inner_cols_, num_diags_, max_diag_len_, lower_, upper_,
                right_align_super_diagonal, right_align_sub_diagonal, is_single_diag_, diag_device_address,
                output_device_address, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

void MatrixSetDiagV3GpuKernelMod::ResetResource() noexcept {
  is_null_input_ = false;
  is_single_diag_ = true;
  k_count_ = 1;
  lower_ = 0;
  upper_ = 0;
  num_diags_ = 0;
  diagonal_count_ = 1;
  max_diag_len_ = 0;
  outer_batch_ = 1;
  inner_rows_ = 0;
  inner_cols_ = 0;
}

std::vector<KernelAttr> MatrixSetDiagV3GpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MatrixDiagV3Func> &pair) { return pair.first; });
  return support_list;
}

std::vector<std::pair<KernelAttr, MatrixSetDiagV3GpuKernelMod::MatrixDiagV3Func>>
  MatrixSetDiagV3GpuKernelMod::func_list_ = {{KernelAttr()
                                                .AddInputAttr(kNumberTypeInt8)
                                                .AddInputAttr(kNumberTypeInt8)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeInt8),
                                              &MatrixSetDiagV3GpuKernelMod::LaunchKernel<int8_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt16)
                                                .AddInputAttr(kNumberTypeInt16)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeInt16),
                                              &MatrixSetDiagV3GpuKernelMod::LaunchKernel<int16_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeInt32),
                                              &MatrixSetDiagV3GpuKernelMod::LaunchKernel<int32_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeInt64),
                                              &MatrixSetDiagV3GpuKernelMod::LaunchKernel<int64_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeUInt8)
                                                .AddInputAttr(kNumberTypeUInt8)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeUInt8),
                                              &MatrixSetDiagV3GpuKernelMod::LaunchKernel<uint8_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeUInt16)
                                                .AddInputAttr(kNumberTypeUInt16)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeUInt16),
                                              &MatrixSetDiagV3GpuKernelMod::LaunchKernel<uint16_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeUInt32)
                                                .AddInputAttr(kNumberTypeUInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeUInt32),
                                              &MatrixSetDiagV3GpuKernelMod::LaunchKernel<uint32_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeUInt64)
                                                .AddInputAttr(kNumberTypeUInt64)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeUInt64),
                                              &MatrixSetDiagV3GpuKernelMod::LaunchKernel<uint64_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeFloat16),
                                              &MatrixSetDiagV3GpuKernelMod::LaunchKernel<half>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeFloat32),
                                              &MatrixSetDiagV3GpuKernelMod::LaunchKernel<float>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeFloat64),
                                              &MatrixSetDiagV3GpuKernelMod::LaunchKernel<double>}};
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MatrixSetDiagV3, MatrixSetDiagV3GpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
