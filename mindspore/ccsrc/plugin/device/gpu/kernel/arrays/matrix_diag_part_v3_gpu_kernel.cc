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

#include "plugin/device/gpu/kernel/arrays/matrix_diag_part_v3_gpu_kernel.h"
#include <map>
#include <memory>
#include <algorithm>
#include "mindspore/core/utils/convert_utils_base.h"
#include "mindspore/core/ops/matrix_diag_part_v3.h"
namespace mindspore {
namespace kernel {
namespace {
using KernelRunFunc = MatrixDiagPartV3GpuKernelMod::KernelRunFunc;
constexpr int kMatrixDiagPartV3InputsNum = 3;
constexpr int kMatrixDiagPartV3OutputsNum = 1;
constexpr int kMatrixDiagPartV3MinOutputShape = 2;
}  // namespace

bool MatrixDiagPartV3GpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMatrixDiagPartV3InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMatrixDiagPartV3OutputsNum, kernel_name_);

  auto matrix_prim = std::make_shared<ops::MatrixDiagPartV3>(base_operator->GetPrim());
  auto align = matrix_prim->get_align();
  left_align_super_diag_ = (align == "LEFT_LEFT" || align == "LEFT_RIGHT");
  left_align_sub_diag_ = (align == "LEFT_LEFT" || align == "RIGHT_LEFT");

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}

int MatrixDiagPartV3GpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
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
bool MatrixDiagPartV3GpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                const std::vector<AddressPtr> &workspace,
                                                const std::vector<AddressPtr> &outputs) {
  auto matrix_ptr = GetDeviceAddress<T>(inputs, kIndex0);
  auto k_ptr = GetDeviceAddress<IndexType>(inputs, kIndex1);
  auto padding_value_ptr = GetDeviceAddress<T>(inputs, kIndex2);
  auto diag_ptr = GetDeviceAddress<T>(outputs, kIndex0);

  // Get 'k' and store as [lower_diag_index, upper_diag_index].
  IndexType k_stand;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpy(&k_stand, k_ptr, sizeof(IndexType), cudaMemcpyDeviceToHost),
                                     "cudaMemcpy input 'k' to host failed.");
  int64_t upper_diag_index, lower_diag_index = IntToLong(k_stand);
  if (k_size_ == 1) {
    upper_diag_index = lower_diag_index;
  } else {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpy(&k_stand, k_ptr + 1, sizeof(IndexType), cudaMemcpyDeviceToHost),
                                       "cudaMemcpy input 'k' to host failed.");
    upper_diag_index = IntToLong(k_stand);
  }

  MatrixDiagPartV3(matrix_ptr, padding_value_ptr, diag_ptr, num_rows_, num_cols_, lower_diag_index, upper_diag_index,
                   diag_size_, max_diag_len_, left_align_super_diag_, left_align_sub_diag_, device_id_, stream_ptr_);

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
