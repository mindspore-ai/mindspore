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
#include "mindspore/ccsrc/distributed/embedding_cache/embedding_cache_utils.h"

namespace mindspore {
namespace kernel {
constexpr size_t kScatterArithmeticInputsNum = 3;
constexpr size_t kScatterArithmeticOutputsNum = 1;

bool ScatterArithmeticCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
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

  use_embedding_cache_ = GetValue<bool>(base_operator->GetAttr(kAttrUseEmbeddingStore));
  parameter_key_ = GetValue<int64_t>(base_operator->GetAttr(kAttrParameterKey));
  return KRET_OK;
}

template <typename T>
bool ScatterArithmeticCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                 const std::vector<kernel::AddressPtr> &,
                                                 const std::vector<kernel::AddressPtr> &outputs) {
  static const mindspore::HashMap<std::string, std::function<T(const T &a, const T &b)>> scatter_arithmetic_func_map{
    {prim::kPrimScatterMul->name(), [](const T &a, const T &b) { return a * b; }},
    {prim::kPrimScatterDiv->name(), [](const T &a, const T &b) { return a / b; }},
    {prim::kPrimScatterAdd->name(), [](const T &a, const T &b) { return a + b; }},
    {prim::kPrimScatterSub->name(), [](const T &a, const T &b) { return a - b; }},
    {prim::kPrimScatterMax->name(), [](const T &a, const T &b) { return a > b ? a : b; }},
    {prim::kPrimScatterMin->name(), [](const T &a, const T &b) { return a > b ? b : a; }},
    {prim::kPrimScatterUpdate->name(), [](const T &a, const T &b) { return b; }},
  };
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kScatterArithmeticInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kScatterArithmeticOutputsNum, kernel_name_);
  auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto *indices = reinterpret_cast<int *>(inputs[1]->addr);
  auto *updates = reinterpret_cast<T *>(inputs[2]->addr);
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
        if (std::equal_to<T>()(updates[base_index_updates + j], 0)) {
          MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', updates must not contain 0";
        }
        input[base_index_input + j] = func_iter->second(input[base_index_input + j], updates[base_index_updates + j]);
      }
    }
  } else {
    // A temporary solution to support external storage for embedding cache. Parameter Server use "ScatterUpdate" kernel
    // to update embedding cache at server. So we need use this kernel to update embedding cache by embedding store.
    // We will create a new kernel to support this feature in future.
    if (kernel_name_ == "ScatterUpdate" && use_embedding_cache_) {
      auto embedding_store = embedding_store_manager.Get(std::to_string(parameter_key_));
      MS_ERROR_IF_NULL(embedding_store);
      if (!embedding_store->Put(input, indices_size_, indices, updates)) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', for embedding cache failed for parameter: " << parameter_key_;
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
        input[base_index_input + j] = func_iter->second(input[base_index_input + j], updates[base_index_updates + j]);
      }
    }
  }
  return true;
}

#define SCATTER_ARITHMETIC_CPU_REGISTER(IN_DT0, IN_DT1, IN_DT2, OUT_DT0, T)                                          \
  KernelAttr().AddInputAttr(IN_DT0).AddInputAttr(IN_DT1).AddInputAttr(IN_DT2).AddOutputAttr(OUT_DT0).AddOutInRef(0,  \
                                                                                                                 0), \
    &ScatterArithmeticCpuKernelMod::LaunchKernel<T>

const ScatterArithmeticCpuKernelMod::ScatterSupportListType &ScatterArithmeticCpuKernelMod::GetFuncList() const {
  static const ScatterArithmeticCpuKernelMod::ScatterSupportListType func_list = {
    {SCATTER_ARITHMETIC_CPU_REGISTER(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int32_t)},
    {SCATTER_ARITHMETIC_CPU_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeFloat32,
                                     float)},
    {SCATTER_ARITHMETIC_CPU_REGISTER(kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt64, int64_t)},
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
