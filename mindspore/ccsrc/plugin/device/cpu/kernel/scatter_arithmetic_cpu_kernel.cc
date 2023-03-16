/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/scatter_arithmetic_cpu_kernel.h"
#include <algorithm>
#include <map>
#include <memory>
#include <limits>
#include <string>
#include <utility>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "include/backend/distributed/embedding_cache/embedding_cache_utils.h"

namespace mindspore {
namespace kernel {
constexpr size_t kScatterArithmeticInputsNum = 3;
constexpr size_t kScatterArithmeticOutputsNum = 1;

bool ScatterArithmeticCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();

  if (base_operator->HasAttr(kAttrEnableEmbeddingStorage)) {
    enable_embedding_storage_ = GetValue<bool>(base_operator->GetAttr(kAttrEnableEmbeddingStorage));
  }
  if (base_operator->HasAttr(kAttrParameterKey)) {
    parameter_key_ = GetValue<int32_t>(base_operator->GetAttr(kAttrParameterKey));
  }

  return MatchKernelFunc(base_operator, inputs, outputs);
}

int ScatterArithmeticCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs[0]->GetShapeVector();
  auto indices_shape = inputs[1]->GetShapeVector();
  if (input_shape.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the shape of 'input_x' can not be empty.";
  }

  first_dim_size_ = LongToInt(input_shape[0]);
  int64_t size_tmp = 1;
  for (size_t i = 1; i < input_shape.size(); i++) {
    size_tmp *= input_shape[i];
  }
  inner_size_ = LongToSize(size_tmp);
  input_size_ = LongToSize(input_shape[0]) * inner_size_;
  size_tmp = 1;
  for (size_t i = 0; i < indices_shape.size(); i++) {
    size_tmp *= indices_shape[i];
  }
  indices_size_ = LongToSize(size_tmp);
  return KRET_OK;
}

template <typename T, typename S>
bool ScatterArithmeticCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                 const std::vector<kernel::AddressPtr> &,
                                                 const std::vector<kernel::AddressPtr> &outputs) {
  static const mindspore::HashMap<std::string, std::function<void(T & a, const T &b)>> scatter_arithmetic_func_map{
    {prim::kPrimScatterMul->name(), [](T &a, const T &b) { a *= b; }},
    {prim::kPrimScatterDiv->name(), [](T &a, const T &b) { a /= b; }},
    {prim::kPrimScatterAdd->name(), [](T &a, const T &b) { a += b; }},
    {prim::kPrimScatterSub->name(), [](T &a, const T &b) { a -= b; }},
    {prim::kPrimScatterMax->name(), [](T &a, const T &b) { a = a > b ? a : b; }},
    {prim::kPrimScatterMin->name(), [](T &a, const T &b) { a = a > b ? b : a; }},
    {prim::kPrimScatterUpdate->name(), [](T &a, const T &b) { a = b; }},
  };
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kScatterArithmeticInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kScatterArithmeticOutputsNum, kernel_name_);
  auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto *indices = reinterpret_cast<S *>(inputs[1]->addr);
  auto *updates = reinterpret_cast<T *>(inputs[2]->addr);
  auto *output = reinterpret_cast<T *>(outputs[0]->addr);
  auto func_iter = scatter_arithmetic_func_map.find(kernel_name_);
  if (func_iter == scatter_arithmetic_func_map.end()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the current operator does not support this operation.";
    return false;
  }
  if (kernel_name_ == "ScatterDiv") {
    for (size_t i = 0; i < indices_size_; i++) {
      auto base_index_updates = i * inner_size_;
      auto base_index_input = indices[i] * inner_size_;
      if (indices[i] < 0 || indices[i] >= first_dim_size_) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of indices should be in [0, " << first_dim_size_
                          << "), but got '" << indices[i] << "' in indices.";
      }
      for (size_t j = 0; j < inner_size_; j++) {
        if (std::equal_to<T>()(updates[base_index_updates + j], static_cast<T>(0))) {
          MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', updates must not contain 0";
        }
        func_iter->second(input[base_index_input + j], updates[base_index_updates + j]);
      }
    }
  } else {
    if (enable_embedding_storage_) {
      auto embedding_storage = embedding_storage_manager.Get(parameter_key_);
      MS_ERROR_IF_NULL(embedding_storage);
      if (!embedding_storage->Put({indices, inputs[1]->size}, {updates, inputs[2]->size})) {
        MS_LOG(ERROR) << "For '" << kernel_name_
                      << "', Update embedding storage failed, parameter key: " << parameter_key_;
        return false;
      }
      return true;
    }

    for (size_t i = 0; i < indices_size_; i++) {
      auto base_index_updates = i * inner_size_;
      auto base_index_input = indices[i] * inner_size_;
      if (indices[i] < 0 || indices[i] >= first_dim_size_) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of indices should be in [0, " << first_dim_size_
                          << "), but got '" << indices[i] << "' in indices.";
      }
      for (size_t j = 0; j < inner_size_; j++) {
        func_iter->second(input[base_index_input + j], updates[base_index_updates + j]);
      }
    }
  }

  // Scatter ops are registered as a ref type operator. The new runtime supports the ref mechanism with the same input
  // and output addresses, but the old runtime does not support the ref mechanism, and the input and output addresses
  // are different. Therefore, in order to adapt to the old runtime, the content of the input needs to be copied to
  // output. After removing the old runtime, the following copy logic code can be deleted.
  if (input != output) {
    auto bufferSize = outputs[0]->size;
    auto ret = memcpy_s(output, bufferSize, input, input_size_ * sizeof(T));
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memory copy failed. Error no: " << ret;
    }
  }
  return true;
}

#define SCATTER_ARITHMETIC_CPU_REGISTER(IN_DT0, IN_DT1, IN_DT2, OUT_DT0, T, S)                                       \
  KernelAttr().AddInputAttr(IN_DT0).AddInputAttr(IN_DT1).AddInputAttr(IN_DT2).AddOutputAttr(OUT_DT0).AddOutInRef(0,  \
                                                                                                                 0), \
    &ScatterArithmeticCpuKernelMod::LaunchKernel<T, S>

const ScatterArithmeticCpuKernelMod::ScatterSupportListType &ScatterArithmeticCpuKernelMod::GetFuncList() const {
  static const ScatterArithmeticCpuKernelMod::ScatterSupportListType func_list = {
    {SCATTER_ARITHMETIC_CPU_REGISTER(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int32_t,
                                     int)},
    {SCATTER_ARITHMETIC_CPU_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeFloat32,
                                     float, int)},
    {SCATTER_ARITHMETIC_CPU_REGISTER(kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt64, int64_t,
                                     int)},
    {SCATTER_ARITHMETIC_CPU_REGISTER(kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt8, kNumberTypeInt8, int8_t, int)},
    {SCATTER_ARITHMETIC_CPU_REGISTER(kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeUInt8, kNumberTypeUInt8, uint8_t,
                                     int)},
    {SCATTER_ARITHMETIC_CPU_REGISTER(kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeFloat16, kNumberTypeFloat16,
                                     float16, int)},

    {SCATTER_ARITHMETIC_CPU_REGISTER(kNumberTypeFloat16, kNumberTypeInt64, kNumberTypeFloat16, kNumberTypeFloat16,
                                     float16, int64_t)},
    {SCATTER_ARITHMETIC_CPU_REGISTER(kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt32, int32_t,
                                     int64_t)},
    {SCATTER_ARITHMETIC_CPU_REGISTER(kNumberTypeInt8, kNumberTypeInt64, kNumberTypeInt8, kNumberTypeInt8, int8_t,
                                     int64_t)},
    {SCATTER_ARITHMETIC_CPU_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeFloat32, kNumberTypeFloat32,
                                     float, int64_t)},
    {SCATTER_ARITHMETIC_CPU_REGISTER(kNumberTypeUInt8, kNumberTypeInt64, kNumberTypeUInt8, kNumberTypeUInt8, uint8_t,
                                     int64_t)},
    {SCATTER_ARITHMETIC_CPU_REGISTER(kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, int64_t,
                                     int64_t)},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ScatterAdd, ScatterArithmeticCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ScatterSub, ScatterArithmeticCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ScatterMul, ScatterArithmeticCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ScatterDiv, ScatterArithmeticCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ScatterMax, ScatterArithmeticCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ScatterMin, ScatterArithmeticCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ScatterUpdate, ScatterArithmeticCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
