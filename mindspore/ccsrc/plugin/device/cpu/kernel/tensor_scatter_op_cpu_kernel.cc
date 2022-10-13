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

#include "plugin/device/cpu/kernel/tensor_scatter_op_cpu_kernel.h"
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include "include/common/thread_pool.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"
#include "utils/profile.h"
#include "Eigen/Eigen"
#include "unsupported/Eigen/CXX11/Tensor"

namespace mindspore {
namespace kernel {
namespace {
template <typename T>
using MatrixXd = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
constexpr size_t kMinIndiceRank = 2;
enum class Op { ADD, SUB, MAX, MIN, MUL, DIV };
std::map<string, Op> OpMap{
  {prim::kPrimTensorScatterAdd->name(), Op::ADD}, {prim::kPrimTensorScatterSub->name(), Op::SUB},
  {prim::kPrimTensorScatterMax->name(), Op::MAX}, {prim::kPrimTensorScatterMin->name(), Op::MIN},
  {prim::kPrimTensorScatterMul->name(), Op::MUL}, {prim::kPrimTensorScatterDiv->name(), Op::DIV},
};
}  // namespace
bool TensorScatterOpCpuKernelMode::Init(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

int TensorScatterOpCpuKernelMode::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input_shape_.clear();
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(input_shape_), LongToSize);
  auto indices_shape = inputs.at(kIndex1)->GetShapeVector();
  auto updates_shape = inputs.at(kIndex2)->GetShapeVector();
  const auto indices_rank = indices_shape.size();
  const auto last_indices_value = LongToSize(indices_shape.back());
  const auto update_rank = updates_shape.size();
  constexpr size_t min_indices_rank = 2;
  slice_size_ = last_indices_value;
  batch_size_ = 1;
  inner_size_ = 1;

  if (indices_shape.size() < kMinIndiceRank) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the dimension of 'indices' must be at least 2, but got "
                             << indices_shape.size();
  }

  for (size_t i = 0; i < update_rank; ++i) {
    if (i <= indices_rank - min_indices_rank) {
      batch_size_ *= LongToSize(indices_shape[i]);
    } else {
      inner_size_ *= LongToSize(updates_shape[i]);
    }
  }
  batch_strides_.resize(slice_size_);
  // Since the quit condition(i >= 0) is about negative integer,
  // we convert iterated index from unsigned integer to signed integer.
  for (auto i = SizeToLong(slice_size_) - 1; i >= 0; --i) {
    auto dim = LongToSize(i);
    total_batch_size_ *= input_shape_[dim];
    if (dim == slice_size_ - 1) {
      batch_strides_[dim] = 1;
    } else {
      batch_strides_[dim] = batch_strides_[dim + 1] * input_shape_[dim + 1];
    }
  }
  return KRET_OK;
}

template <typename T>
inline void ComputeFunc(const string &kernel_name, MatrixXd<T> eigen_output, size_t output_index,
                        MatrixXd<T> eigen_update, size_t update_index) {
  auto out_index = SizeToLong(output_index);
  auto upd_index = SizeToLong(update_index);
  switch (OpMap[kernel_name]) {
    case Op::ADD:
      eigen_output.row(out_index) += eigen_update.row(upd_index);
      break;
    case Op::SUB:
      eigen_output.row(out_index) -= eigen_update.row(upd_index);
      break;
    case Op::MAX:
      eigen_output.row(out_index) = eigen_output.row(out_index).cwiseMax(eigen_update.row(upd_index));
      break;
    case Op::MIN:
      eigen_output.row(out_index) = eigen_output.row(out_index).cwiseMin(eigen_update.row(upd_index));
      break;
    case Op::MUL:
      eigen_output.row(out_index) = eigen_output.row(out_index).cwiseProduct(eigen_update.row(upd_index));
      break;
    case Op::DIV:
      eigen_output.row(out_index) = eigen_output.row(out_index).cwiseQuotient(eigen_update.row(upd_index));
      break;
    default:
      MS_LOG(EXCEPTION) << "Invalid kernel name: " << kernel_name;
  }
}

template <typename T, typename S>
bool TensorScatterOpCpuKernelMode::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                const std::vector<kernel::AddressPtr> &,
                                                const std::vector<kernel::AddressPtr> &outputs) {
  auto input = GetDeviceAddress<T>(inputs, kIndex0);
  auto indices = GetDeviceAddress<S>(inputs, kIndex1);
  auto updates = GetDeviceAddress<T>(inputs, kIndex2);
  auto output = GetDeviceAddress<T>(outputs, kIndex0);

  // ScatterNd* operations need to write input data and copy into output data,
  // while TensorScatter* operations need to copy input data and write into output data.
  auto ret = memcpy_s(output, outputs[kIndex0]->size, input, inputs[kIndex0]->size);
  if (ret == ERANGE) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', input data size[" << inputs[kIndex0]->size
                      << " bytes] is larger than memcpy_s cache limit[" << SECUREC_MEM_MAX_LEN
                      << " bytes]. Error no: " << ret;
  }
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it's memcpy_s function run error. Error no: " << ret;
  }
  int64_t invalid_index_pos = -1;
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_update(updates, batch_size_,
                                                                                             inner_size_);
  Eigen::Map<Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_indices(indices, batch_size_,
                                                                                              slice_size_);
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_output(output, total_batch_size_,
                                                                                             inner_size_);
  for (size_t i = 0; i < batch_size_; ++i) {
    size_t out_index = 0;
    for (size_t j = 0; j < slice_size_; ++j) {
      S idx_index = eigen_indices(SizeToLong(i), SizeToLong(j));
      out_index += batch_strides_[j] * static_cast<size_t>(idx_index);
      if (idx_index < 0 || idx_index >= static_cast<S>(input_shape_[j])) {
        invalid_index_pos = SizeToLong(i * slice_size_);
        break;
      }
    }
    if (invalid_index_pos != -1) {
      break;
    }
    ComputeFunc<T>(kernel_name_, eigen_output, out_index, eigen_update, i);
  }
  if (invalid_index_pos != -1) {
    std::stringstream indices_ss;
    std::stringstream input_shape_ss;
    for (size_t i = 0; i < slice_size_; i++) {
      if (i > 0) {
        indices_ss << ", ";
        input_shape_ss << ", ";
      }
      indices_ss << std::to_string(indices[LongToSize(invalid_index_pos) + i]);
      input_shape_ss << std::to_string(input_shape_[i]);
    }
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the " << invalid_index_pos << "-th value of 'indices'["
                      << indices_ss.str() << "] is out of range[" << input_shape_ss.str() << "].";
  }
  return true;
}

#define TENSOR_SCATTER_OP_CPU_REGISTER(IN_DT0, IN_DT1, IN_DT2, OUT_DT0, T, S)                         \
  KernelAttr().AddInputAttr(IN_DT0).AddInputAttr(IN_DT1).AddInputAttr(IN_DT2).AddOutputAttr(OUT_DT0), \
    &TensorScatterOpCpuKernelMode::LaunchKernel<T, S>

const TensorScatterOpCpuKernelMode::TensorScatterSupportListType &TensorScatterOpCpuKernelMode::GetFuncList() const {
  static const TensorScatterOpCpuKernelMode::TensorScatterSupportListType func_list = {
    {TENSOR_SCATTER_OP_CPU_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeFloat64, kNumberTypeFloat64,
                                    double, int64_t)},
    {TENSOR_SCATTER_OP_CPU_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeFloat32, kNumberTypeFloat32, float,
                                    int64_t)},
    {TENSOR_SCATTER_OP_CPU_REGISTER(kNumberTypeFloat16, kNumberTypeInt64, kNumberTypeFloat16, kNumberTypeFloat16,
                                    float16, int64_t)},
    {TENSOR_SCATTER_OP_CPU_REGISTER(kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, int64_t,
                                    int64_t)},
    {TENSOR_SCATTER_OP_CPU_REGISTER(kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt32, int32_t,
                                    int64_t)},
    {TENSOR_SCATTER_OP_CPU_REGISTER(kNumberTypeInt16, kNumberTypeInt64, kNumberTypeInt16, kNumberTypeInt16, int16_t,
                                    int64_t)},
    {TENSOR_SCATTER_OP_CPU_REGISTER(kNumberTypeInt8, kNumberTypeInt64, kNumberTypeInt8, kNumberTypeInt8, int8_t,
                                    int64_t)},
    {TENSOR_SCATTER_OP_CPU_REGISTER(kNumberTypeUInt64, kNumberTypeInt64, kNumberTypeUInt64, kNumberTypeUInt64, uint64_t,
                                    int64_t)},
    {TENSOR_SCATTER_OP_CPU_REGISTER(kNumberTypeUInt32, kNumberTypeInt64, kNumberTypeUInt32, kNumberTypeUInt32, uint32_t,
                                    int64_t)},
    {TENSOR_SCATTER_OP_CPU_REGISTER(kNumberTypeUInt16, kNumberTypeInt64, kNumberTypeUInt16, kNumberTypeUInt16, uint16_t,
                                    int64_t)},
    {TENSOR_SCATTER_OP_CPU_REGISTER(kNumberTypeUInt8, kNumberTypeInt64, kNumberTypeUInt8, kNumberTypeUInt8, uint8_t,
                                    int64_t)},
    {TENSOR_SCATTER_OP_CPU_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, kNumberTypeFloat64, kNumberTypeFloat64,
                                    double, int32_t)},
    {TENSOR_SCATTER_OP_CPU_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeFloat32, float,
                                    int32_t)},
    {TENSOR_SCATTER_OP_CPU_REGISTER(kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeFloat16, kNumberTypeFloat16,
                                    float16, int32_t)},
    {TENSOR_SCATTER_OP_CPU_REGISTER(kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt64, int64_t,
                                    int32_t)},
    {TENSOR_SCATTER_OP_CPU_REGISTER(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int32_t,
                                    int32_t)},
    {TENSOR_SCATTER_OP_CPU_REGISTER(kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt16, kNumberTypeInt16, int16_t,
                                    int32_t)},
    {TENSOR_SCATTER_OP_CPU_REGISTER(kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt8, kNumberTypeInt8, int8_t,
                                    int32_t)},
    {TENSOR_SCATTER_OP_CPU_REGISTER(kNumberTypeUInt64, kNumberTypeInt32, kNumberTypeUInt64, kNumberTypeUInt64, uint64_t,
                                    int32_t)},
    {TENSOR_SCATTER_OP_CPU_REGISTER(kNumberTypeUInt32, kNumberTypeInt32, kNumberTypeUInt32, kNumberTypeUInt32, uint32_t,
                                    int32_t)},
    {TENSOR_SCATTER_OP_CPU_REGISTER(kNumberTypeUInt16, kNumberTypeInt32, kNumberTypeUInt16, kNumberTypeUInt16, uint16_t,
                                    int32_t)},
    {TENSOR_SCATTER_OP_CPU_REGISTER(kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeUInt8, kNumberTypeUInt8, uint8_t,
                                    int32_t)},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorScatterAdd, TensorScatterOpCpuKernelMode);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorScatterSub, TensorScatterOpCpuKernelMode);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorScatterMax, TensorScatterOpCpuKernelMode);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorScatterMin, TensorScatterOpCpuKernelMode);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorScatterDiv, TensorScatterOpCpuKernelMode);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorScatterMul, TensorScatterOpCpuKernelMode);
}  // namespace kernel
}  // namespace mindspore
