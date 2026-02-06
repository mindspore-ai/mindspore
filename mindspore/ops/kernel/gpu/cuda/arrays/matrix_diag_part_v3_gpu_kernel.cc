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

#include "kernel/gpu/cuda/arrays/matrix_diag_part_v3_gpu_kernel.h"
#include <map>
#include <memory>
#include <algorithm>
#include "utils/convert_utils_base.h"
#include "mindspore/ops/infer/matrix_diag_part_v3.h"
namespace mindspore {
namespace kernel {
namespace {
using KernelRunFunc = MatrixDiagPartV3GpuKernelMod::KernelRunFunc;
constexpr int kMatrixDiagPartV3InputsNum = 3;
constexpr int kMatrixDiagPartV3OutputsNum = 1;
constexpr int kMatrixDiagPartV3MinOutputShape = 2;
}  // namespace

bool MatrixDiagPartV3GpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMatrixDiagPartV3InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMatrixDiagPartV3OutputsNum, kernel_name_);

  auto align = GetValue<std::string>(primitive_->GetAttr("align"));
  left_align_super_diag_ = (align == "LEFT_LEFT" || align == "LEFT_RIGHT");
  left_align_sub_diag_ = (align == "LEFT_LEFT" || align == "RIGHT_LEFT");

  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }

  return true;
}

int MatrixDiagPartV3GpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto matrix_shape = inputs.at(kIndex0)->GetShapeVector();
  CHECK_SHAPE_NULL(LongVecToSizeVec(matrix_shape), kernel_name_, "input");
  auto k_shape = inputs.at(kIndex1)->GetShapeVector();
  auto diag_shape = outputs.at(kIndex0)->GetShapeVector();

  diag_size_ = std::accumulate(diag_shape.begin(), diag_shape.end(), int64_t(1), std::multiplies{});
  num_cols_ = matrix_shape.at(matrix_shape.size() - kIndex1);
  num_rows_ = matrix_shape.at(matrix_shape.size() - kIndex2);

  k_size_ = std::accumulate(k_shape.begin(), k_shape.end(), int64_t(1), std::multiplies{});
  max_diag_len_ = diag_shape.back();

  return KRET_OK;
}

template <typename T>
bool MatrixDiagPartV3GpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &workspace,
                                                const std::vector<KernelTensor *> &outputs) {
  auto matrix_ptr = GetDeviceAddress<T>(inputs, kIndex0);
  auto k_ptr = GetDeviceAddress<IndexType>(inputs, kIndex1);
  auto padding_value_ptr = GetDeviceAddress<T>(inputs, kIndex2);
  auto diag_ptr = GetDeviceAddress<T>(outputs, kIndex0);
  auto any = [](auto &&... args) -> bool { return ((args == nullptr) || ...); };
  if (any(cuda_stream_, matrix_ptr, k_ptr, padding_value_ptr, diag_ptr)) {
    return false;
  }

  // Get 'k' and store as [lower_diag_index, upper_diag_index].
  IndexType k_stand;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&k_stand, k_ptr, sizeof(IndexType), cudaMemcpyDeviceToHost, cuda_stream_),
    "For '" << kernel_name_ << "', cudaMemcpyAsync input 'k' to host failed.");
  if (cudaStreamQuery(cuda_stream_) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream_), "cuda Stream Sync Failed");
  }
  int64_t upper_diag_index, lower_diag_index = IntToLong(k_stand);
  if (k_size_ == 1) {
    upper_diag_index = lower_diag_index;
  } else {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(&k_stand, k_ptr + 1, sizeof(IndexType), cudaMemcpyDeviceToHost, cuda_stream_),
      "For '" << kernel_name_ << "', cudaMemcpyAsync input 'k' to host failed.");
    if (cudaStreamQuery(cuda_stream_) != cudaSuccess) {
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream_), "cuda Stream Sync Failed");
    }
    upper_diag_index = IntToLong(k_stand);
  }

  auto status =
    MatrixDiagPartV3(matrix_ptr, padding_value_ptr, diag_ptr, num_rows_, num_cols_, lower_diag_index, upper_diag_index,
                     diag_size_, max_diag_len_, left_align_super_diag_, left_align_sub_diag_, device_id_, cuda_stream_);
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

#define DTYPE_REGISTER(INPUT, K, PADDING, OUTPUT, T)                                              \
  {                                                                                               \
    KernelAttr().AddInputAttr(INPUT).AddInputAttr(K).AddInputAttr(PADDING).AddOutputAttr(OUTPUT), \
      &MatrixDiagPartV3GpuKernelMod::LaunchKernel<T>                                              \
  }

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &MatrixDiagPartV3GpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    DTYPE_REGISTER(kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt8, kNumberTypeInt8, int8_t),
    DTYPE_REGISTER(kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeUInt8, kNumberTypeUInt8, uint8_t),
    DTYPE_REGISTER(kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt16, kNumberTypeInt16, int16_t),
    DTYPE_REGISTER(kNumberTypeUInt16, kNumberTypeInt32, kNumberTypeUInt16, kNumberTypeUInt16, uint16_t),
    DTYPE_REGISTER(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int32_t),
    DTYPE_REGISTER(kNumberTypeUInt32, kNumberTypeInt32, kNumberTypeUInt32, kNumberTypeUInt32, uint32_t),
    DTYPE_REGISTER(kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt64, int64_t),
    DTYPE_REGISTER(kNumberTypeUInt64, kNumberTypeInt32, kNumberTypeUInt64, kNumberTypeUInt64, uint64_t),
    DTYPE_REGISTER(kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeFloat16, kNumberTypeFloat16, half),
    DTYPE_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeFloat32, float),
    DTYPE_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, kNumberTypeFloat64, kNumberTypeFloat64, double),
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MatrixDiagPartV3, MatrixDiagPartV3GpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
