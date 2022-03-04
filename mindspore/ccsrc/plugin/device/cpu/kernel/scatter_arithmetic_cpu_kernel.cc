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
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kScatterArithmeticInputsNum = 3;
constexpr size_t kScatterArithmeticOutputsNum = 1;

template <typename T>
class ScatterArithmeticCpuKernelFunc : public CpuKernelFunc {
 public:
  ScatterArithmeticCpuKernelFunc() = default;
  ~ScatterArithmeticCpuKernelFunc() override = default;

  void InitFunc(const CNodePtr &kernel_node) override;

  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
               const std::vector<AddressPtr> &outputs) override;

 private:
  void InitComputeFunc();
  void ScatterAdd(T *input, const int *indices, const T *updates) const;
  void ScatterSub(T *input, const int *indices, const T *updates) const;
  void ScatterMul(T *input, const int *indices, const T *updates) const;
  void ScatterDiv(T *input, const int *indices, const T *updates) const;
  void ScatterMax(T *input, const int *indices, const T *updates) const;
  void ScatterMin(T *input, const int *indices, const T *updates) const;
  void ScatterUpdate(T *input, const int *indices, const T *updates) const;

  using TypeComputeFunc = std::function<void(ScatterArithmeticCpuKernelFunc *, T *, const int *, const T *)>;

  TypeComputeFunc compute_func_;
  int input_shape_0{0};
  size_t input_size_{0};
  size_t inner_size_{0};
  size_t indices_size_{0};
  const size_t INPUT_INDEX_{0};
  const size_t INDICES_INDEX_{1};
  const size_t UPDATES_INDEX_{2};
  const size_t OUTPUT_INDEX_{0};
  std::string kernel_name_;
};

template <typename T>
void ScatterArithmeticCpuKernelFunc<T>::InitComputeFunc() {
  static const std::map<std::string, TypeComputeFunc> scatterArithmeticFuncMap{
    {prim::kPrimScatterAdd->name(), &ScatterArithmeticCpuKernelFunc<T>::ScatterAdd},
    {prim::kPrimScatterSub->name(), &ScatterArithmeticCpuKernelFunc<T>::ScatterSub},
    {prim::kPrimScatterMul->name(), &ScatterArithmeticCpuKernelFunc<T>::ScatterMul},
    {prim::kPrimScatterDiv->name(), &ScatterArithmeticCpuKernelFunc<T>::ScatterDiv},
    {prim::kPrimScatterMax->name(), &ScatterArithmeticCpuKernelFunc<T>::ScatterMax},
    {prim::kPrimScatterMin->name(), &ScatterArithmeticCpuKernelFunc<T>::ScatterMin},
    {prim::kPrimScatterUpdate->name(), &ScatterArithmeticCpuKernelFunc<T>::ScatterUpdate}};
  if (scatterArithmeticFuncMap.find(kernel_name_) == scatterArithmeticFuncMap.end()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the current operator does not support this operation.";
  }
  compute_func_ = scatterArithmeticFuncMap.at(kernel_name_);
}

template <typename T>
void ScatterArithmeticCpuKernelFunc<T>::InitFunc(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (input_shape.size() < 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of 'input_x' should be greater than or equal to 1, but got "
                      << input_shape.size() << ".";
  }
  input_shape_0 = SizeToInt(input_shape[0]);
  input_size_ = 1;
  inner_size_ = 1;
  if (input_shape.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the shape of 'input_x' should be not empty.";
  }

  for (size_t i = 1; i < input_shape.size(); i++) {
    inner_size_ *= input_shape[i];
  }
  input_size_ = input_shape[0] * inner_size_;
  auto indices_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  indices_size_ = 1;
  for (size_t i = 0; i < indices_shape.size(); i++) {
    indices_size_ *= indices_shape[i];
  }
  InitComputeFunc();
}

template <typename T>
bool ScatterArithmeticCpuKernelFunc<T>::RunFunc(const std::vector<kernel::AddressPtr> &inputs,
                                                const std::vector<kernel::AddressPtr> &,
                                                const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kScatterArithmeticInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kScatterArithmeticOutputsNum, kernel_name_);
  auto *input = reinterpret_cast<T *>(inputs[INPUT_INDEX_]->addr);
  auto *indices = reinterpret_cast<int *>(inputs[INDICES_INDEX_]->addr);
  auto *updates = reinterpret_cast<T *>(inputs[UPDATES_INDEX_]->addr);
  auto *output = reinterpret_cast<T *>(outputs[OUTPUT_INDEX_]->addr);
  compute_func_(this, input, indices, updates);
  auto bufferSize = outputs[OUTPUT_INDEX_]->size;
  auto ret = memcpy_s(output, bufferSize, input, input_size_ * sizeof(T));
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memory copy failed. Error no: " << ret;
  }
  return true;
}

template <typename T>
void ScatterArithmeticCpuKernelFunc<T>::ScatterAdd(T *input, const int *indices, const T *updates) const {
  for (size_t i = 0; i < indices_size_; i++) {
    if (indices[i] >= input_shape_0) {
      continue;
    }
    auto base_index_updates = i * inner_size_;
    auto base_index_input = indices[i] * inner_size_;
    for (size_t j = 0; j < inner_size_; j++) {
      input[base_index_input + j] += updates[base_index_updates + j];
    }
  }
}

template <typename T>
void ScatterArithmeticCpuKernelFunc<T>::ScatterSub(T *input, const int *indices, const T *updates) const {
  for (size_t i = 0; i < indices_size_; i++) {
    if (indices[i] >= input_shape_0) {
      continue;
    }
    auto base_index_updates = i * inner_size_;
    auto base_index_input = indices[i] * inner_size_;
    for (size_t j = 0; j < inner_size_; j++) {
      input[base_index_input + j] -= updates[base_index_updates + j];
    }
  }
}

template <typename T>
void ScatterArithmeticCpuKernelFunc<T>::ScatterMul(T *input, const int *indices, const T *updates) const {
  for (size_t i = 0; i < indices_size_; i++) {
    auto base_index_updates = i * inner_size_;
    auto base_index_input = indices[i] * inner_size_;
    for (size_t j = 0; j < inner_size_; j++) {
      input[base_index_input + j] *= updates[base_index_updates + j];
    }
  }
}

template <typename T>
void ScatterArithmeticCpuKernelFunc<T>::ScatterDiv(T *input, const int *indices, const T *updates) const {
  for (size_t i = 0; i < indices_size_; i++) {
    for (size_t j = 0; j < inner_size_; j++) {
      auto dividend = input[indices[i] * inner_size_ + j];
      auto divisor = updates[i * inner_size_ + j];
      if (divisor != 0) {
        input[indices[i] * inner_size_ + j] = dividend / divisor;
        continue;
      }
      if (dividend == 0) {
        input[indices[i] * inner_size_ + j] = std::numeric_limits<T>::quiet_NaN();
        continue;
      }
      if (std::numeric_limits<T>::has_infinity) {
        input[indices[i] * inner_size_ + j] =
          dividend > 0 ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      } else {
        input[indices[i] * inner_size_ + j] =
          dividend > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
      }
    }
  }
}

template <typename T>
void ScatterArithmeticCpuKernelFunc<T>::ScatterMax(T *input, const int *indices, const T *updates) const {
  for (size_t i = 0; i < indices_size_; i++) {
    auto base_index_updates = i * inner_size_;
    auto base_index_input = indices[i] * inner_size_;
    for (size_t j = 0; j < inner_size_; j++) {
      input[base_index_input + j] = input[base_index_input + j] > updates[base_index_updates + j]
                                      ? input[base_index_input + j]
                                      : updates[base_index_updates + j];
    }
  }
}

template <typename T>
void ScatterArithmeticCpuKernelFunc<T>::ScatterMin(T *input, const int *indices, const T *updates) const {
  for (size_t i = 0; i < indices_size_; i++) {
    auto base_index_updates = i * inner_size_;
    auto base_index_input = indices[i] * inner_size_;
    for (size_t j = 0; j < inner_size_; j++) {
      input[base_index_input + j] = input[base_index_input + j] < updates[base_index_updates + j]
                                      ? input[base_index_input + j]
                                      : updates[base_index_updates + j];
    }
  }
}

template <typename T>
void ScatterArithmeticCpuKernelFunc<T>::ScatterUpdate(T *input, const int *indices, const T *updates) const {
  for (size_t i = 0; i < indices_size_; i++) {
    auto base_index_updates = i * inner_size_;
    auto base_index_input = indices[i] * inner_size_;
    for (size_t j = 0; j < inner_size_; j++) {
      input[base_index_input + j] = updates[base_index_updates + j];
    }
  }
}

template <typename T>
std::shared_ptr<CpuKernelFunc> SpecializeScatterArithFunc() {
  return std::make_shared<ScatterArithmeticCpuKernelFunc<T>>();
}
using SpecializeScatterArithFuncCreator = std::function<std::shared_ptr<CpuKernelFunc>()>;
static std::map<std::string, std::vector<std::pair<KernelAttr, SpecializeScatterArithFuncCreator>>>
  func_class_list_map = {{kScatterAdd,
                          {{KernelAttr()
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddOutputAttr(kNumberTypeInt32),
                            SpecializeScatterArithFunc<int32_t>},
                           {KernelAttr()
                              .AddInputAttr(kNumberTypeFloat32)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeFloat32)
                              .AddOutputAttr(kNumberTypeFloat32),
                            SpecializeScatterArithFunc<float>},
                           {KernelAttr()
                              .AddInputAttr(kNumberTypeInt64)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeInt64)
                              .AddOutputAttr(kNumberTypeInt64),
                            SpecializeScatterArithFunc<int64_t>}}},
                         {kScatterSub,
                          {{KernelAttr()
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddOutputAttr(kNumberTypeInt32),
                            SpecializeScatterArithFunc<int32_t>},
                           {KernelAttr()
                              .AddInputAttr(kNumberTypeFloat32)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeFloat32)
                              .AddOutputAttr(kNumberTypeFloat32),
                            SpecializeScatterArithFunc<float>},
                           {KernelAttr()
                              .AddInputAttr(kNumberTypeInt64)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeInt64)
                              .AddOutputAttr(kNumberTypeInt64),
                            SpecializeScatterArithFunc<int64_t>}}},
                         {kScatterMul,
                          {{KernelAttr()
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddOutputAttr(kNumberTypeInt32),
                            SpecializeScatterArithFunc<int32_t>},
                           {KernelAttr()
                              .AddInputAttr(kNumberTypeFloat32)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeFloat32)
                              .AddOutputAttr(kNumberTypeFloat32),
                            SpecializeScatterArithFunc<float>},
                           {KernelAttr()
                              .AddInputAttr(kNumberTypeInt64)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeInt64)
                              .AddOutputAttr(kNumberTypeInt64),
                            SpecializeScatterArithFunc<int64_t>}}},
                         {kScatterDiv,
                          {{KernelAttr()
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddOutputAttr(kNumberTypeInt32),
                            SpecializeScatterArithFunc<int32_t>},
                           {KernelAttr()
                              .AddInputAttr(kNumberTypeFloat32)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeFloat32)
                              .AddOutputAttr(kNumberTypeFloat32),
                            SpecializeScatterArithFunc<float>},
                           {KernelAttr()
                              .AddInputAttr(kNumberTypeInt64)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeInt64)
                              .AddOutputAttr(kNumberTypeInt64),
                            SpecializeScatterArithFunc<int64_t>}}},
                         {kScatterMax,
                          {{KernelAttr()
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddOutputAttr(kNumberTypeInt32),
                            SpecializeScatterArithFunc<int32_t>},
                           {KernelAttr()
                              .AddInputAttr(kNumberTypeFloat32)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeFloat32)
                              .AddOutputAttr(kNumberTypeFloat32),
                            SpecializeScatterArithFunc<float>},
                           {KernelAttr()
                              .AddInputAttr(kNumberTypeInt64)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeInt64)
                              .AddOutputAttr(kNumberTypeInt64),
                            SpecializeScatterArithFunc<int64_t>}}},
                         {kScatterMin,
                          {{KernelAttr()
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddOutputAttr(kNumberTypeInt32),
                            SpecializeScatterArithFunc<int32_t>},
                           {KernelAttr()
                              .AddInputAttr(kNumberTypeFloat32)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeFloat32)
                              .AddOutputAttr(kNumberTypeFloat32),
                            SpecializeScatterArithFunc<float>},
                           {KernelAttr()
                              .AddInputAttr(kNumberTypeInt64)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeInt64)
                              .AddOutputAttr(kNumberTypeInt64),
                            SpecializeScatterArithFunc<int64_t>}}},
                         {kScatterUpdate,
                          {{KernelAttr()
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddOutputAttr(kNumberTypeInt32),
                            SpecializeScatterArithFunc<int32_t>},
                           {KernelAttr()
                              .AddInputAttr(kNumberTypeFloat32)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeFloat32)
                              .AddOutputAttr(kNumberTypeFloat32),
                            SpecializeScatterArithFunc<float>},
                           {KernelAttr()
                              .AddInputAttr(kNumberTypeInt64)
                              .AddInputAttr(kNumberTypeInt32)
                              .AddInputAttr(kNumberTypeInt64)
                              .AddOutputAttr(kNumberTypeInt64),
                            SpecializeScatterArithFunc<int64_t>}}}};
}  // namespace

void ScatterArithmeticCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  if (kernel_name_ != kernel_type_) {
    MS_LOG(EXCEPTION) << "Need to be " << kernel_type_ << " but got kernel name as " << kernel_name_;
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "ScatterArithmetic does not support this kernel data type: " << kernel_attr;
  }

  func_obj_ = func_class_list_map[kernel_name_][index].second();
  func_obj_->InitFunc(kernel_node);
}

std::vector<KernelAttr> ScatterArithmeticCpuKernelMod::GetOpSupport() {
  auto iter = func_class_list_map.find(kernel_type_);
  if (iter == func_class_list_map.end()) {
    MS_LOG(EXCEPTION) << "ScatterArithmetic cpu does not support " << kernel_type_;
  }

  std::vector<KernelAttr> support_list;
  std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                 [](const std::pair<KernelAttr, SpecializeScatterArithFuncCreator> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ScatterAdd,
                                 []() { return std::make_shared<ScatterArithmeticCpuKernelMod>(kScatterAdd); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ScatterSub,
                                 []() { return std::make_shared<ScatterArithmeticCpuKernelMod>(kScatterSub); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ScatterMul,
                                 []() { return std::make_shared<ScatterArithmeticCpuKernelMod>(kScatterMul); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ScatterDiv,
                                 []() { return std::make_shared<ScatterArithmeticCpuKernelMod>(kScatterDiv); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ScatterMax,
                                 []() { return std::make_shared<ScatterArithmeticCpuKernelMod>(kScatterMax); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ScatterMin,
                                 []() { return std::make_shared<ScatterArithmeticCpuKernelMod>(kScatterMin); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ScatterUpdate,
                                 []() { return std::make_shared<ScatterArithmeticCpuKernelMod>(kScatterUpdate); });
}  // namespace kernel
}  // namespace mindspore
