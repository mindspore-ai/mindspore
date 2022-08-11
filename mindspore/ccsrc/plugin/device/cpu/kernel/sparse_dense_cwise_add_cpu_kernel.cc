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

void SparseDenseCwiseAddCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  indices_shape = AnfAlgo::GetInputDeviceShape(kernel_node, kIndex0);
  values_shape = AnfAlgo::GetInputDeviceShape(kernel_node, kIndex1);
  shape_shape = AnfAlgo::GetInputDeviceShape(kernel_node, kIndex2);
  dense_shape = AnfAlgo::GetInputDeviceShape(kernel_node, kIndex3);
  data_type = AnfAlgo::GetInputDeviceDataType(kernel_node, kIndex3);
}

template <typename T>
void SparseDenseCwiseAddCpuKernelMod::ComputeAdd(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &outputs) {
  auto indices_data = reinterpret_cast<int64_t *>(inputs[kIndex0]->addr);
  auto sparse_shape_data = reinterpret_cast<int64_t *>(inputs[kIndex2]->addr);
  int64_t index_num = indices_shape[kIndex0];
  int64_t dimension = indices_shape[kIndex1];
  int64_t dense_dims = dense_shape.size();

  for (int64_t i = 0; i < index_num; i++) {
    for (int64_t j = 0; j < dimension; j++) {
      if (indices_data[i * dimension + j] >= sparse_shape_data[j]) {
        MS_EXCEPTION(ValueError) << "For SparseDenseCwiseAdd, the indices can't"
                                 << "proceed to cross the border the interview.";
      }
    }
  }

  std::vector<int64_t> sparse_shape(dimension);
  for (int64_t i = 0; i < dimension; i++) {
    sparse_shape[i] = sparse_shape_data[i];
  }
  int64_t dense_num = 1;
  for (int64_t i = 0; i < dense_dims; i++) {
    dense_num *= dense_shape[i];
  }

  bool isNeedBcast = (dense_shape == sparse_shape || dense_num == 1);
  if (isNeedBcast) {
    SparseDenseCwiseAddNoBcastCompute<T>(inputs, outputs);
  } else if (dense_dims <= dimension) {
    for (int i = dense_dims - 1; i >= 0; --i) {
      if ((dense_shape[i] != 1) && (dense_shape[i] != sparse_shape[i + dimension - dense_dims])) {
        MS_EXCEPTION(ValueError) << "For SparseDenseCwiseAdd, the shape of 'x2' can't broadcast to 'x1_shape'."
                                 << "In order to broadcast, the size of the trailing axes for 'x2' and"
                                 << "sparse in an operation must either be the same size or size of the"
                                 << "trailing axes for 'x2' must be one";
      }
    }
    SparseDenseCwiseAddBcastCompute<T>(inputs, outputs);
  } else {
    MS_EXCEPTION(ValueError) << "For SparseDenseCwiseAdd, dims of 'x2' should be smaller or equal to Number of"
                             << "elements of 'x1_shape'. Because broadcast direction can only be from dense to sparse."
                             << "but got dims of dense:" << dense_dims << "dims of sparse:" << dimension << ".";
  }
}

template <typename T>
void SparseDenseCwiseAddCpuKernelMod::SparseDenseCwiseAddNoBcastCompute(const std::vector<AddressPtr> &inputs,
                                                                        const std::vector<AddressPtr> &outputs) {
  auto sparse_indices_data = reinterpret_cast<int64_t *>(inputs[kIndex0]->addr);
  auto sparse_values_data = reinterpret_cast<T *>(inputs[kIndex1]->addr);
  auto sparse_shape_data = reinterpret_cast<int64_t *>(inputs[kIndex2]->addr);
  auto dense_data = reinterpret_cast<T *>(inputs[kIndex3]->addr);
  auto output_data = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  int64_t value_nums = indices_shape[kIndex0];
  int64_t dimension = indices_shape[kIndex1];
  int64_t data_num = values_shape[kIndex0];
  int64_t dense_dims = dense_shape.size();

  std::vector<T> sparse_values_vec(data_num);
  for (int64_t i = 0; i < data_num; i++) {
    sparse_values_vec[i] = (sparse_values_data[i]);
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
      output_data[i] = sparse_values_vec[i] + dense_data[index];
    }
  } else {
    for (int64_t i = 0; i < value_nums; i++) {
      output_data[i] = sparse_values_data[i] + *(dense_data);
    }
  }
}

template <typename T>
void SparseDenseCwiseAddCpuKernelMod::SparseDenseCwiseAddBcastCompute(const std::vector<AddressPtr> &inputs,
                                                                      const std::vector<AddressPtr> &outputs) {
  auto sparse_indices_data = reinterpret_cast<int64_t *>(inputs[kIndex0]->addr);
  auto sparse_values_data = reinterpret_cast<T *>(inputs[kIndex1]->addr);
  auto sparse_shape_data = reinterpret_cast<int64_t *>(inputs[kIndex2]->addr);
  auto dense_data = reinterpret_cast<T *>(inputs[kIndex3]->addr);
  auto output_data = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  int64_t value_nums = indices_shape[kIndex0];
  int64_t dimension = indices_shape[kIndex1];
  int64_t data_num = values_shape[kIndex0];
  int64_t dims = shape_shape[kIndex0];

  int64_t Sparse_numelements = 1;
  for (int64_t i = 0; i < dims; i++) {
    Sparse_numelements *= sparse_shape_data[i];
  }

  std::vector<T> sparse_values_vec(data_num);
  for (int64_t i = 0; i < data_num; i++) {
    sparse_values_vec[i] = (sparse_values_data[i]);
  }

  std::vector<int64_t> sparse_shape(dimension);
  for (int64_t i = 0; i < dimension; i++) {
    sparse_shape[i] = sparse_shape_data[i];
  }
  std::vector<int64_t> sparse_shape1(dimension);
  for (int64_t j = 0; j < dimension; j++) {
    sparse_shape1[j] = sparse_shape[j];
  }

  BroadcastIterator broad_base_iter_1(sparse_shape, dense_shape, sparse_shape1);
  std::vector<T> Dense(Sparse_numelements);
  broad_base_iter_1.SetPos(0);
  for (int64_t i = 0; i < Sparse_numelements; i++) {
    Dense[i] = dense_data[broad_base_iter_1.GetInputPosB()];
    broad_base_iter_1.GenNextPos();
  }
  for (int64_t i = 0; i < value_nums; i++) {
    int index = 0;
    for (int64_t j = 0; j < dimension - 1; j++) {
      int c = 1;
      for (int64_t k = j + 1; k < dimension; k++) {
        c = c * sparse_shape_data[k];
      }
      index += sparse_indices_data[j + i * dimension] * c;
    }
    index += sparse_indices_data[(i + 1) * dimension - 1];
    output_data[i] = sparse_values_vec[i] + Dense[index];
  }
}

bool SparseDenseCwiseAddCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                             const std::vector<kernel::AddressPtr> &,
                                             const std::vector<AddressPtr> &outputs) {
  if (data_type == kNumberTypeInt8) {
    ComputeAdd<int8_t>(inputs, outputs);
  } else if (data_type == kNumberTypeInt16) {
    ComputeAdd<int16_t>(inputs, outputs);
  } else if (data_type == kNumberTypeInt32) {
    ComputeAdd<int32_t>(inputs, outputs);
  } else if (data_type == kNumberTypeInt64) {
    ComputeAdd<int64_t>(inputs, outputs);
  } else if (data_type == kNumberTypeUInt8) {
    ComputeAdd<uint8_t>(inputs, outputs);
  } else if (data_type == kNumberTypeUInt16) {
    ComputeAdd<uint16_t>(inputs, outputs);
  } else if (data_type == kNumberTypeUInt32) {
    ComputeAdd<uint32_t>(inputs, outputs);
  } else if (data_type == kNumberTypeUInt64) {
    ComputeAdd<uint64_t>(inputs, outputs);
  } else if (data_type == kNumberTypeFloat16) {
    ComputeAdd<float16>(inputs, outputs);
  } else if (data_type == kNumberTypeFloat32) {
    ComputeAdd<float>(inputs, outputs);
  } else if (data_type == kNumberTypeFloat64) {
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
