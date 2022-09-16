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

#include "plugin/device/cpu/kernel/unary_op_cpu_kernel.h"
#include <map>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <memory>
#include "nnacl/fp32/arithmetic_self_fp32.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kReal = "Real";
constexpr auto kImag = "Imag";
constexpr auto kConj = "Conj";
constexpr auto kCeil = "Ceil";

template <typename T, typename S>
class UnaryOpCpuKernelFunc : public DeprecatedCpuKernelFunc {
 public:
  UnaryOpCpuKernelFunc() = default;
  ~UnaryOpCpuKernelFunc() override = default;
  using UnaryOpFunc = std::function<void(const T *, S *, size_t, size_t)>;
  void InitFunc(const CNodePtr &kernel_node) override;
  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
               const std::vector<AddressPtr> &outputs) override;

 private:
  void GetUnaryOpFunc();
  UnaryOpFunc unary_op_func_{nullptr};
  std::string kernel_name_;
};

void Ceil(const float *input, float *output, size_t start, size_t end) {
  auto input_s = input + start;
  auto output_s = output + start;
  ElementCeil(input_s, output_s, (end - start));
}

template <typename T, typename S>
void Real(const T *input, S *output, size_t start, size_t end) {
  for (size_t i = start; i < end; ++i) {
    output[i] = static_cast<S>(std::real(input[i]));
  }
}

template <typename T, typename S>
void Imag(const T *input, S *output, size_t start, size_t end) {
  for (size_t i = start; i < end; ++i) {
    output[i] = static_cast<S>(std::imag(input[i]));
  }
}

template <typename T, typename S>
void Conj(const T *input, S *output, size_t start, size_t end) {
  if constexpr (std::is_same<T, S>::value) {
    if constexpr ((std::is_same<T, complex64>::value || std::is_same<T, complex128>::value)) {
      for (size_t i = start; i < end; ++i) {
        output[i] = static_cast<S>(std::conj(input[i]));
      }
    } else {
      for (size_t i = start; i < end; ++i) {
        output[i] = static_cast<S>(input[i]);
      }
    }
  } else {
    MS_LOG(EXCEPTION) << "For Conj, it's output data type not equal to input data type.";
  }
}

template <typename T, typename S>
void UnaryOpCpuKernelFunc<T, S>::GetUnaryOpFunc() {
  const std::map<std::string, UnaryOpFunc> kCommonSupportedMap = {{prim::kPrimReal->name(), &Real<T, S>},
                                                                  {prim::kPrimImag->name(), &Imag<T, S>},
                                                                  {prim::kPrimConj->name(), &Conj<T, S>}};
  if constexpr (std::is_same<T, float>::value) {
    if (kernel_name_ == prim::kPrimCeil->name()) {
      unary_op_func_ = &Ceil;
      return;
    }
  }
  auto iter = kCommonSupportedMap.find(kernel_name_);
  if (iter != kCommonSupportedMap.end()) {
    unary_op_func_ = iter->second;
  }
}

template <typename T, typename S>
void UnaryOpCpuKernelFunc<T, S>::InitFunc(const CNodePtr &kernel_node) {
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  GetUnaryOpFunc();
}

template <typename T, typename S>
bool UnaryOpCpuKernelFunc<T, S>::RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                         const std::vector<AddressPtr> &outputs) {
  auto input = inputs.front();
  auto output = outputs.front();
  const auto input_addr = reinterpret_cast<T *>(input->addr);
  auto output_addr = reinterpret_cast<S *>(output->addr);
  if (unary_op_func_ == nullptr) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it has no cpu backend implements.";
  }
  ParallelLaunchAutoSearch(
    std::bind(unary_op_func_, input_addr, output_addr, std::placeholders::_1, std::placeholders::_2),
    output->size / sizeof(S), this, &parallel_search_info_);
  return true;
}

template <typename T, typename S>
std::shared_ptr<DeprecatedCpuKernelFunc> SpecializeUnaryFunc() {
  return std::make_shared<UnaryOpCpuKernelFunc<T, S>>();
}
using UnaryOpCpuFuncCreator = std::function<std::shared_ptr<DeprecatedCpuKernelFunc>()>;
std::map<std::string, std::vector<std::pair<KernelAttr, UnaryOpCpuFuncCreator>>> kernel_attr_list = {
  {kReal,
   {{KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeFloat64),
     SpecializeUnaryFunc<complex128, double>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeFloat32),
     SpecializeUnaryFunc<complex64, float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8), SpecializeUnaryFunc<char, char>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     SpecializeUnaryFunc<int16_t, int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     SpecializeUnaryFunc<int32_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     SpecializeUnaryFunc<int64_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     SpecializeUnaryFunc<uint16_t, uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     SpecializeUnaryFunc<uint32_t, uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     SpecializeUnaryFunc<uint64_t, uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     SpecializeUnaryFunc<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     SpecializeUnaryFunc<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool), SpecializeUnaryFunc<bool, bool>}}},
  {kCeil,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     SpecializeUnaryFunc<float, float>}}},
  {kImag,
   {{KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeFloat64),
     SpecializeUnaryFunc<complex128, double>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeFloat32),
     SpecializeUnaryFunc<complex64, float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8), SpecializeUnaryFunc<char, char>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     SpecializeUnaryFunc<int16_t, int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     SpecializeUnaryFunc<int32_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     SpecializeUnaryFunc<int64_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     SpecializeUnaryFunc<uint16_t, uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     SpecializeUnaryFunc<uint32_t, uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     SpecializeUnaryFunc<uint64_t, uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     SpecializeUnaryFunc<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     SpecializeUnaryFunc<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool), SpecializeUnaryFunc<bool, bool>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     SpecializeUnaryFunc<uint8_t, uint8_t>}}},
  {kConj,
   {{KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     SpecializeUnaryFunc<complex128, complex128>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     SpecializeUnaryFunc<complex64, complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8), SpecializeUnaryFunc<char, char>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     SpecializeUnaryFunc<int16_t, int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     SpecializeUnaryFunc<int32_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     SpecializeUnaryFunc<int64_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     SpecializeUnaryFunc<uint16_t, uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     SpecializeUnaryFunc<uint32_t, uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     SpecializeUnaryFunc<uint64_t, uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     SpecializeUnaryFunc<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     SpecializeUnaryFunc<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool), SpecializeUnaryFunc<bool, bool>}}}};
}  // namespace

void UnaryOpCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  if (kernel_name_ != kernel_type_) {
    MS_LOG(EXCEPTION) << "Need to be " << kernel_type_ << " but got kernel name as " << kernel_name_;
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_type_ << " does not support this kernel data type: " << kernel_attr;
  }

  func_obj_ = kernel_attr_list[kernel_type_][index].second();
  func_obj_->InitFunc(kernel_node);
}

std::vector<KernelAttr> UnaryOpCpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_list.find(kernel_type_);
  if (iter == kernel_attr_list.end()) {
    MS_LOG(EXCEPTION) << "UnaryOp cpu does not support " << kernel_type_;
  }

  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, UnaryOpCpuFuncCreator> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Real,
                                 []() { return std::make_shared<UnaryOpCpuKernelMod>(kReal); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Imag,
                                 []() { return std::make_shared<UnaryOpCpuKernelMod>(kImag); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Conj,
                                 []() { return std::make_shared<UnaryOpCpuKernelMod>(kConj); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Ceil,
                                 []() { return std::make_shared<UnaryOpCpuKernelMod>(kCeil); });
}  // namespace kernel
}  // namespace mindspore
