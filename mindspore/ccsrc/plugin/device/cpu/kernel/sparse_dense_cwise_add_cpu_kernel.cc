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

#include "plugin/device/cpu/kernel/sparse_dense_cwise_add_cpu_kernel.h"
#include <string>
#include <vector>
#include <algorithm>

namespace mindspore {
namespace kernel {
namespace {
constexpr int64_t kSparseDenseCwiseInputsNum = 4;
constexpr int64_t kSparseDenseCwiseOutputsNum = 1;
const int64_t kIndex0 = 0;
const int64_t kIndex1 = 1;
const int64_t kIndex2 = 2;
const int64_t kIndex3 = 3;
}  // namespace

bool SparseDenseCwiseAddCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs) {
  data_type_ = inputs.at(kIndex3)->GetDtype();
  return true;
}

int SparseDenseCwiseAddCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs,
                                            const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  indices_shape_ = inputs.at(kIndex0)->GetShapeVector();
  values_shape_ = inputs.at(kIndex1)->GetShapeVector();
  shape_shape_ = inputs.at(kIndex2)->GetShapeVector();
  dense_shape_ = inputs.at(kIndex3)->GetShapeVector();
  return KRET_OK;
}

template <typename T>
void SparseDenseCwiseAddCpuKernelMod::ComputeAdd(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &outputs) {
  auto indices_data = static_cast<int64_t *>(inputs[kIndex0]->addr);
  auto sparse_shape_data = static_cast<int64_t *>(inputs[kIndex2]->addr);
  int64_t index_num = indices_shape_[kIndex0];
  int64_t dimension = indices_shape_[kIndex1];
  int64_t dense_dims = static_cast<int64_t>(dense_shape_.size());

  for (int64_t i = 0; i < index_num; i++) {
    for (int64_t j = 0; j < dimension; j++) {
      if (indices_data[static_cast<size_t>(i * dimension + j)] >= sparse_shape_data[static_cast<size_t>(j)]) {
        MS_EXCEPTION(ValueError) << "For SparseDenseCwiseAdd, the indices cannot go out of bounds.";
      }
    }
  }

  std::vector<int64_t> sparse_shape(dimension);
  for (size_t i = 0; i < static_cast<size_t>(dimension); i++) {
    sparse_shape[i] = static_cast<int64_t>(sparse_shape_data[i]);
  }
  int64_t dense_num = 1;
  for (size_t i = 0; i < static_cast<size_t>(dense_dims); i++) {
    dense_num *= dense_shape_[i];
  }

  bool isNeedBcast = (dense_shape_ == sparse_shape || dense_num == 1);
  if (isNeedBcast) {
    SparseDenseCwiseAddNoBcastCompute<T>(inputs, outputs);
  } else if (dense_dims <= dimension) {
    for (int64_t i = dense_dims - 1; i >= 0; --i) {
      if ((dense_shape_[static_cast<size_t>(i)] != 1) &&
          (dense_shape_[static_cast<size_t>(i)] != sparse_shape[static_cast<size_t>(i + dimension - dense_dims)])) {
        MS_EXCEPTION(ValueError) << "For SparseDenseCwiseAdd, the shape of 'x2' can't broadcast to 'x1_shape'. "
                                 << "In order to broadcast, the size of the trailing axes for 'x2' and "
                                 << "sparse in an operation must either be the same size or size of the "
                                 << "trailing axes for 'x2' must be one.";
      }
    }
    SparseDenseCwiseAddBcastCompute<T>(inputs, outputs);
  } else {
    MS_EXCEPTION(ValueError) << "For SparseDenseCwiseAdd, dims of 'x2' should be smaller or equal to Number of "
                             << "elements of 'x1_shape'. Because broadcast direction can only be from dense to sparse. "
                             << "But got dims of dense:" << dense_dims << "dims of sparse:" << dimension << ".";
  }
}

template <typename T>
void SparseDenseCwiseAddCpuKernelMod::SparseDenseCwiseAddNoBcastCompute(const std::vector<AddressPtr> &inputs,
                                                                        const std::vector<AddressPtr> &outputs) {
  auto sparse_indices_data = static_cast<int64_t *>(inputs[kIndex0]->addr);
  auto sparse_values_data = static_cast<T *>(inputs[kIndex1]->addr);
  auto sparse_shape_data = static_cast<int64_t *>(inputs[kIndex2]->addr);
  auto dense_data = static_cast<T *>(inputs[kIndex3]->addr);
  auto output_data = static_cast<T *>(outputs[kIndex0]->addr);
  int64_t value_nums = indices_shape_[kIndex0];
  int64_t dimension = indices_shape_[kIndex1];
  int64_t data_num = values_shape_[kIndex0];
  int64_t dense_dims = static_cast<int64_t>(dense_shape_.size());

  std::vector<T> sparse_values_vec(data_num);
  for (size_t i = 0; i < static_cast<size_t>(data_num); i++) {
    sparse_values_vec[i] = static_cast<T>((sparse_values_data[i]));
  }
  if (dimension == dense_dims) {
    for (int64_t i = 0; i < value_nums; i++) {
      int index = 0;
      for (int64_t j = 0; j < dimension - 1; j++) {
        int c = 1;
        for (int64_t k = j + 1; k < dimension; k++) {
          c = c * sparse_shape_data[k];
        }
        index += c * sparse_indices_data[j + i * dimension];
      }
      index += sparse_indices_data[(i + 1) * dimension - 1];
      output_data[i] = static_cast<T>(sparse_values_vec[i] + dense_data[index]);
    }
  } else {
    for (size_t i = 0; i < static_cast<size_t>(value_nums); i++) {
      output_data[i] = static_cast<T>(sparse_values_data[i] + *(dense_data));
    }
  }
}

template <typename T>
void SparseDenseCwiseAddCpuKernelMod::SparseDenseCwiseAddBcastCompute(const std::vector<AddressPtr> &inputs,
                                                                      const std::vector<AddressPtr> &outputs) {
  auto sparse_indices_data = static_cast<int64_t *>(inputs[kIndex0]->addr);
  auto sparse_values_data = static_cast<T *>(inputs[kIndex1]->addr);
  auto sparse_shape_data = static_cast<int64_t *>(inputs[kIndex2]->addr);
  auto dense_data = static_cast<T *>(inputs[kIndex3]->addr);
  auto output_data = static_cast<T *>(outputs[kIndex0]->addr);
  int64_t value_nums = indices_shape_[kIndex0];
  int64_t dimension = indices_shape_[kIndex1];
  int64_t data_num = values_shape_[kIndex0];
  int64_t dims = shape_shape_[kIndex0];

  int64_t Sparse_numelements = 1;
  for (int64_t i = 0; i < dims; i++) {
    Sparse_numelements *= sparse_shape_data[i];
  }

  std::vector<T> sparse_values_vec(data_num);
  for (size_t i = 0; i < static_cast<size_t>(data_num); i++) {
    sparse_values_vec[i] = static_cast<T>((sparse_values_data[i]));
  }

  std::vector<int64_t> sparse_shape(dimension);
  for (size_t i = 0; i < static_cast<size_t>(dimension); i++) {
    sparse_shape[i] = static_cast<int64_t>(sparse_shape_data[i]);
  }
  std::vector<int64_t> sparse_shape1(dimension);
  for (size_t j = 0; j < static_cast<size_t>(dimension); j++) {
    sparse_shape1[j] = static_cast<int64_t>(sparse_shape[j]);
  }

  BroadcastIterator broad_base_iter_1(sparse_shape, dense_shape_, sparse_shape1);
  std::vector<T> Dense(Sparse_numelements);
  broad_base_iter_1.SetPos(0);
  for (size_t i = 0; i < static_cast<size_t>(Sparse_numelements); i++) {
    Dense[i] = static_cast<T>(dense_data[broad_base_iter_1.GetInputPosB()]);
    broad_base_iter_1.GenNextPos();
  }
  for (size_t i = 0; i < static_cast<size_t>(value_nums); i++) {
    int index = 0;
    for (size_t j = 0; j < static_cast<size_t>(dimension - 1); j++) {
      int c = 1;
      for (size_t k = j + 1; k < static_cast<size_t>(dimension); k++) {
        c = static_cast<int>(c * sparse_shape_data[k]);
      }
      index += static_cast<int>(sparse_indices_data[j + i * static_cast<size_t>(dimension)] * c);
    }
    index += sparse_indices_data[(i + 1) * static_cast<size_t>(dimension) - 1];
    output_data[i] = static_cast<T>(sparse_values_vec[i]) + static_cast<T>(Dense[static_cast<size_t>(index)]);
  }
}

bool SparseDenseCwiseAddCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                             const std::vector<kernel::AddressPtr> &,
                                             const std::vector<AddressPtr> &outputs) {
  if (data_type_ == kNumberTypeInt8) {
    ComputeAdd<int8_t>(inputs, outputs);
  } else if (data_type_ == kNumberTypeInt16) {
    ComputeAdd<int16_t>(inputs, outputs);
  } else if (data_type_ == kNumberTypeInt32) {
    ComputeAdd<int32_t>(inputs, outputs);
  } else if (data_type_ == kNumberTypeInt64) {
    ComputeAdd<int64_t>(inputs, outputs);
  } else if (data_type_ == kNumberTypeUInt8) {
    ComputeAdd<uint8_t>(inputs, outputs);
  } else if (data_type_ == kNumberTypeUInt16) {
    ComputeAdd<uint16_t>(inputs, outputs);
  } else if (data_type_ == kNumberTypeUInt32) {
    ComputeAdd<uint32_t>(inputs, outputs);
  } else if (data_type_ == kNumberTypeUInt64) {
    ComputeAdd<uint64_t>(inputs, outputs);
  } else if (data_type_ == kNumberTypeFloat16) {
    ComputeAdd<float16>(inputs, outputs);
  } else if (data_type_ == kNumberTypeFloat32) {
    ComputeAdd<float>(inputs, outputs);
  } else if (data_type_ == kNumberTypeFloat64) {
    ComputeAdd<double>(inputs, outputs);
  }
  return true;
}

#define ADD_KERNEL(t1, t2, t3, t4, t5) \
  KernelAttr()                         \
    .AddInputAttr(kNumberType##t1)     \
    .AddInputAttr(kNumberType##t2)     \
    .AddInputAttr(kNumberType##t3)     \
    .AddInputAttr(kNumberType##t4)     \
    .AddOutputAttr(kNumberType##t5)

std::vector<KernelAttr> SparseDenseCwiseAddCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {
    ADD_KERNEL(Int64, Int8, Int64, Int8, Int8),          ADD_KERNEL(Int64, Int16, Int64, Int16, Int16),
    ADD_KERNEL(Int64, Int32, Int64, Int32, Int32),       ADD_KERNEL(Int64, Int64, Int64, Int64, Int64),
    ADD_KERNEL(Int64, UInt8, Int64, UInt8, UInt8),       ADD_KERNEL(Int64, UInt16, Int64, UInt16, UInt16),
    ADD_KERNEL(Int64, UInt32, Int64, UInt32, UInt32),    ADD_KERNEL(Int64, UInt64, Int64, UInt64, UInt64),
    ADD_KERNEL(Int64, Float16, Int64, Float16, Float16), ADD_KERNEL(Int64, Float32, Int64, Float32, Float32),
    ADD_KERNEL(Int64, Float64, Int64, Float64, Float64)};

  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseDenseCwiseAdd, SparseDenseCwiseAddCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
