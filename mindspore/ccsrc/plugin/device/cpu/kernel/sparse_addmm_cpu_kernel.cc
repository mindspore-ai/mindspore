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

#include "plugin/device/cpu/kernel/sparse_addmm_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include <utility>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseAddmmInputsNum = 7;
constexpr size_t kSparseAddmmOutputsNum = 1;
constexpr size_t kSparseAddmmOutputShapeSize = 2;
constexpr size_t kSparseAddmmDenseShapeSize = 2;
constexpr size_t kIndicesSizeNum = 2;
constexpr size_t kIndices2rdDimNum = 2;
constexpr size_t kShapeValue = 0;
constexpr size_t kIndex0 = 0;
constexpr size_t kIndex1 = 1;
constexpr size_t kIndex2 = 2;
constexpr size_t kIndex3 = 3;
constexpr size_t kIndex4 = 4;
constexpr size_t kIndex5 = 5;
constexpr size_t kIndex6 = 6;
}  // namespace

bool SparseAddmmCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "SparseAddmm does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int SparseAddmmCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto indices_shape = inputs.at(kIndex0)->GetShapeVector();
  if (indices_shape.size() != kIndicesSizeNum && LongToSize(indices_shape[1]) != kIndices2rdDimNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', it requires 'indices' should be a 2-D Tensor and the second dimension length "
                         "should be 2, but got 'indices' shape: "
                      << Vector2Str(indices_shape);
  }
  auto values_shape = inputs.at(kIndex1)->GetShapeVector();
  if (values_shape.size() != 1 || values_shape[0] != indices_shape[0]) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', it requires 'values' should be a 1-D Tensor and the first dimension length "
                         " should be equal to the first dimension length of 'indices', but got 'values' shape: "
                      << Vector2Str(values_shape) << " and 'indices' shape: " << Vector2Str(indices_shape);
  }
  output_shape_ = Convert2SizeT(outputs[0]->GetShapeVector());
  values_size_ = LongToSize(values_shape[0]);
  b_shape_ = Convert2SizeT(inputs.at(kIndex3)->GetShapeVector());
  if (b_shape_.size() != kSparseAddmmDenseShapeSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of 'dense' should be "
                      << kSparseAddmmDenseShapeSize << "-D, but got " << b_shape_.size() << "-D";
  }
  if (output_shape_.size() != kSparseAddmmOutputShapeSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of output should be "
                      << kSparseAddmmOutputShapeSize << "-D, but got " << output_shape_.size() << "-D";
  }
  return KRET_OK;
}

template <typename I, typename T>
bool SparseAddmmCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseAddmmInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseAddmmOutputsNum, kernel_name_);
  auto ret = memset_s(outputs[0]->addr, outputs[0]->size, 0, outputs[0]->size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset output failed. Error no: " << ret;
  }

  auto *a_indices = static_cast<I *>(inputs[kIndex0]->addr);
  auto *a_values = static_cast<T *>(inputs[kIndex1]->addr);
  auto *x1_shape = static_cast<I *>(inputs[kIndex2]->addr);
  auto *b = static_cast<T *>(inputs[kIndex3]->addr);
  auto *c = static_cast<T *>(inputs[kIndex4]->addr);
  auto *alpha = static_cast<T *>(inputs[kIndex5]->addr);
  auto *beta = static_cast<T *>(inputs[kIndex6]->addr);
  auto *out = static_cast<T *>(outputs[kIndex0]->addr);

  const size_t indices_length = inputs[kIndex0]->size / sizeof(I);
  const size_t values_length = inputs[kIndex1]->size / sizeof(T);
  const size_t b_length = inputs[kIndex3]->size / sizeof(T);

  const size_t dim_num = 2;
  const size_t out_dim_0 = output_shape_[0];
  const size_t out_dim_1 = output_shape_[1];
  const size_t b_dim_0 = b_shape_[0];
  const size_t b_dim_1 = b_shape_[1];
  const size_t same_dim = b_dim_0;

  const I x1_shape_0 = x1_shape[0];
  const I x1_shape_1 = x1_shape[1];

  const size_t x1_shape_0_s = IntToSize(x1_shape_0);
  const size_t x1_shape_1_s = IntToSize(x1_shape_1);
  if (x1_shape_0_s <= kShapeValue || x1_shape_1_s <= kShapeValue) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'x1_shape' should be greater than 0.";
  }
  if (x1_shape_1_s != b_dim_0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the col of 'x1_shape' should be equal to the row of 'x2_dense',"
                         " but got col: "
                      << x1_shape_1_s << ", row: " << b_dim_0;
  }

  for (size_t i = 0; i < values_size_; ++i) {
    if (i * dim_num + 1 >= indices_length) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the index of 'indices' out of bounds.";
    }
    if (i >= values_length) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the index of 'values' out of bounds.";
    }

    const int row = a_indices[i * dim_num];
    const int col = a_indices[i * dim_num + 1];
    if (row >= SizeToInt(out_dim_0) || row < 0 || col >= SizeToInt(same_dim) || col < 0) {
      MS_EXCEPTION(ValueError) << "The indices including out of bounds index, row range: [0, " << out_dim_0
                               << "), col range: [0, " << same_dim << "), but got row: " << row << ", col: " << col;
    }

    const size_t row_s = IntToSize(row);
    const size_t col_s = IntToSize(col);
    const T alpha_value = *(alpha);
    for (size_t n = 0; n < out_dim_1; ++n) {
      if (col_s * b_dim_1 + n >= b_length) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the index of 'b' out of bounds.";
      }
      const T b_value = b[col_s * b_dim_1 + n];
      out[row_s * out_dim_1 + n] += alpha_value * a_values[i] * b_value;
    }
  }

  const T beta_value = *(beta);
  for (size_t i = 0; i < out_dim_0; ++i) {
    for (size_t j = 0; j < out_dim_1; ++j) {
      const T c_value = c[i * out_dim_1 + j];
      out[i * out_dim_1 + j] += beta_value * c_value;
    }
  }

  return true;
}

std::vector<std::pair<KernelAttr, SparseAddmmCpuKernelMod::SparseAddmmFunc>> SparseAddmmCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &SparseAddmmCpuKernelMod::LaunchKernel<int32_t, int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &SparseAddmmCpuKernelMod::LaunchKernel<int64_t, int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   &SparseAddmmCpuKernelMod::LaunchKernel<int32_t, int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   &SparseAddmmCpuKernelMod::LaunchKernel<int64_t, int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &SparseAddmmCpuKernelMod::LaunchKernel<int32_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &SparseAddmmCpuKernelMod::LaunchKernel<int64_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &SparseAddmmCpuKernelMod::LaunchKernel<int32_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &SparseAddmmCpuKernelMod::LaunchKernel<int64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &SparseAddmmCpuKernelMod::LaunchKernel<int32_t, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &SparseAddmmCpuKernelMod::LaunchKernel<int64_t, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &SparseAddmmCpuKernelMod::LaunchKernel<int32_t, double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &SparseAddmmCpuKernelMod::LaunchKernel<int64_t, double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   &SparseAddmmCpuKernelMod::LaunchKernel<int32_t, uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   &SparseAddmmCpuKernelMod::LaunchKernel<int64_t, uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeUInt16)
     .AddOutputAttr(kNumberTypeUInt16),
   &SparseAddmmCpuKernelMod::LaunchKernel<int32_t, uint16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeUInt16)
     .AddOutputAttr(kNumberTypeUInt16),
   &SparseAddmmCpuKernelMod::LaunchKernel<int64_t, uint16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeUInt32)
     .AddOutputAttr(kNumberTypeUInt32),
   &SparseAddmmCpuKernelMod::LaunchKernel<int32_t, uint32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeUInt32)
     .AddOutputAttr(kNumberTypeUInt32),
   &SparseAddmmCpuKernelMod::LaunchKernel<int64_t, uint32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeUInt64)
     .AddOutputAttr(kNumberTypeUInt64),
   &SparseAddmmCpuKernelMod::LaunchKernel<int32_t, uint64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeUInt64)
     .AddOutputAttr(kNumberTypeUInt64),
   &SparseAddmmCpuKernelMod::LaunchKernel<int64_t, uint64_t>}};

std::vector<KernelAttr> SparseAddmmCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SparseAddmmFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseAddmm, SparseAddmmCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
