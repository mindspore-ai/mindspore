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

#include "plugin/device/cpu/kernel/sparse_dense_cwise_mul_cpu_kernel.h"
#include <string>
#include <vector>
#include <algorithm>

namespace mindspore {
namespace kernel {
namespace {
constexpr int64_t kSparseDenseCwiseInputsNum = 4;
constexpr int64_t kSparseDenseCwiseOutputsNum = 1;
const size_t kIndex0 = 0;
const size_t kIndex1 = 1;
const size_t kIndex2 = 2;
const size_t kIndex3 = 3;
}  // namespace

void SparseDenseCwiseMulCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  indices_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kIndex0);
  values_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kIndex1);
  shape_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kIndex2);
  dense_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kIndex3);
  data_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kIndex3);
}

template <typename T>
void SparseDenseCwiseMulCpuKernelMod::ComputeMul(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &outputs) {
  auto indices_data = static_cast<int64_t *>(inputs[kIndex0]->addr);
  auto sparse_shape_data = static_cast<int64_t *>(inputs[kIndex2]->addr);
  int64_t index_num = indices_shape_[kIndex0];
  int64_t dimension = indices_shape_[kIndex1];
  int64_t dense_dims = static_cast<int64_t>(dense_shape_.size());

  for (int64_t i = 0; i < index_num; i++) {
    for (int64_t j = 0; j < dimension; j++) {
      if (indices_data[i * dimension + j] >= sparse_shape_data[j]) {
        MS_EXCEPTION(ValueError) << "For SparseDenseCwiseMul, the indices cannot go out of bounds.";
      }
    }
  }

  std::vector<int64_t> sparse_shape(dimension);
  for (size_t i = 0; i < static_cast<size_t>(dimension); i++) {
    sparse_shape[i] = static_cast<int64_t>(sparse_shape_data[i]);
  }
  int64_t dense_num = 1;
  for (size_t i = 0; i < static_cast<size_t>(dense_dims); i++) {
    dense_num *= static_cast<int64_t>(dense_shape_[i]);
  }

  bool isNeedBcast = (dense_shape_ == sparse_shape || dense_num == 1);
  if (isNeedBcast) {
    SparseDenseCwiseMulNoBcastCompute<T>(inputs, outputs);
  } else if (dense_dims <= dimension) {
    for (int64_t i = dense_dims - 1; i >= 0; --i) {
      if ((dense_shape_[static_cast<size_t>(i)] != 1) &&
          (dense_shape_[static_cast<size_t>(i)] != sparse_shape[static_cast<size_t>(i + dimension - dense_dims)])) {
        MS_EXCEPTION(ValueError) << "For SparseDenseCwiseMul, the shape of 'x2' can't broadcast to 'x1_shape'. "
                                 << "In order to broadcast, the size of the trailing axes for 'x2' and "
                                 << "sparse in an operation must either be the same size or size of the "
                                 << "trailing axes for 'x2' must be one.";
      }
    }
    SparseDenseCwiseMulBcastCompute<T>(inputs, outputs);
  } else {
    MS_EXCEPTION(ValueError) << "For SparseDenseCwiseMul, dims of 'x2' should be smaller or equal to Number of "
                             << "elements of 'x1_shape'. Because broadcast direction can only be from dense to sparse. "
                             << "But got dims of dense:" << dense_dims << "dims of sparse:" << dimension << ".";
  }
}

template <typename T>
void SparseDenseCwiseMulCpuKernelMod::SparseDenseCwiseMulNoBcastCompute(const std::vector<AddressPtr> &inputs,
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
    sparse_values_vec[i] = static_cast<T>(sparse_values_data[i]);
  }
  if (dimension == dense_dims) {
    for (size_t i = 0; i < static_cast<size_t>(value_nums); i++) {
      int index = 0;
      for (size_t j = 0; j < static_cast<size_t>(dimension - 1); j++) {
        int c = 1;
        for (size_t k = j + 1; k < static_cast<size_t>(dimension); k++) {
          c = static_cast<int>(c * sparse_shape_data[k]);
        }
        index += static_cast<int>(c * sparse_indices_data[j + i * static_cast<size_t>(dimension)]);
      }
      index += static_cast<int>(sparse_indices_data[(i + 1) * static_cast<size_t>(dimension) - 1]);
      output_data[i] = static_cast<T>(sparse_values_vec[i] * dense_data[static_cast<size_t>(index)]);
    }
  } else {
    for (size_t i = 0; i < static_cast<size_t>(value_nums); i++) {
      output_data[i] = static_cast<T>(*(dense_data)*sparse_values_data[i]);
    }
  }
}

template <typename T>
void SparseDenseCwiseMulCpuKernelMod::SparseDenseCwiseMulBcastCompute(const std::vector<AddressPtr> &inputs,
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
    sparse_values_vec[i] = static_cast<T>(sparse_values_data[i]);
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
      index += static_cast<int>(sparse_indices_data[j + i * static_cast<size_t>(dimension)]) * c;
    }
    index += static_cast<int>(sparse_indices_data[(i + 1) * static_cast<size_t>(dimension) - 1]);
    output_data[i] = static_cast<T>(sparse_values_vec[i] * Dense[static_cast<size_t>(index)]);
  }
}

bool SparseDenseCwiseMulCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                             const std::vector<kernel::AddressPtr> &,
                                             const std::vector<AddressPtr> &outputs) {
  if (data_type_ == kNumberTypeInt8) {
    ComputeMul<int8_t>(inputs, outputs);
  } else if (data_type_ == kNumberTypeInt16) {
    ComputeMul<int16_t>(inputs, outputs);
  } else if (data_type_ == kNumberTypeInt32) {
    ComputeMul<int32_t>(inputs, outputs);
  } else if (data_type_ == kNumberTypeInt64) {
    ComputeMul<int64_t>(inputs, outputs);
  } else if (data_type_ == kNumberTypeUInt8) {
    ComputeMul<uint8_t>(inputs, outputs);
  } else if (data_type_ == kNumberTypeUInt16) {
    ComputeMul<uint16_t>(inputs, outputs);
  } else if (data_type_ == kNumberTypeUInt32) {
    ComputeMul<uint32_t>(inputs, outputs);
  } else if (data_type_ == kNumberTypeUInt64) {
    ComputeMul<uint64_t>(inputs, outputs);
  } else if (data_type_ == kNumberTypeFloat16) {
    ComputeMul<float16>(inputs, outputs);
  } else if (data_type_ == kNumberTypeFloat32) {
    ComputeMul<float>(inputs, outputs);
  } else if (data_type_ == kNumberTypeFloat64) {
    ComputeMul<double>(inputs, outputs);
  }
  return true;
}

#define MUL_KERNEL(t1, t2, t3, t4, t5) \
  KernelAttr()                         \
    .AddInputAttr(kNumberType##t1)     \
    .AddInputAttr(kNumberType##t2)     \
    .AddInputAttr(kNumberType##t3)     \
    .AddInputAttr(kNumberType##t4)     \
    .AddOutputAttr(kNumberType##t5)

std::vector<KernelAttr> SparseDenseCwiseMulCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {
    MUL_KERNEL(Int64, Int8, Int64, Int8, Int8),          MUL_KERNEL(Int64, Int16, Int64, Int16, Int16),
    MUL_KERNEL(Int64, Int32, Int64, Int32, Int32),       MUL_KERNEL(Int64, Int64, Int64, Int64, Int64),
    MUL_KERNEL(Int64, UInt8, Int64, UInt8, UInt8),       MUL_KERNEL(Int64, UInt16, Int64, UInt16, UInt16),
    MUL_KERNEL(Int64, UInt32, Int64, UInt32, UInt32),    MUL_KERNEL(Int64, UInt64, Int64, UInt64, UInt64),
    MUL_KERNEL(Int64, Float16, Int64, Float16, Float16), MUL_KERNEL(Int64, Float32, Int64, Float32, Float32),
    MUL_KERNEL(Int64, Float64, Int64, Float64, Float64)};

  return kernel_attr_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseDenseCwiseMul, SparseDenseCwiseMulCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
