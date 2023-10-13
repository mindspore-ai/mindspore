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

#include "plugin/device/cpu/kernel/matmul_cpu_kernel.h"
#include "plugin/device/cpu/kernel/eigen/matmul_double_cpu_kernel_func.h"
#include "plugin/device/cpu/kernel/mkldnn/matmul_cpu_kernel_func.h"
#include <utility>
#include <algorithm>
#include <functional>
#include <map>

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kMatMul = "MatMul";
constexpr auto kBatchMatMul = "BatchMatMul";
constexpr auto kMatMulBiasAdd = "FusedMatMulBiasAdd";
constexpr auto kMatMulBiasAddRelu = "MatMulBiasAddReluFusion";

using MatMulFuncCreator = std::function<std::shared_ptr<CpuKernelFunc>()>;
static std::map<std::string, std::vector<std::pair<KernelAttr, MatMulFuncCreator>>> support_list_map = {
  {kMatMul,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     []() { return std::make_shared<MatMulCpuKernelFunc>(); }},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     []() { return std::make_shared<MatMulCpuKernelFunc>(); }},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     []() { return std::make_shared<MatmulDoubleCpuKernelFunc>(); }},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     []() { return std::make_shared<MatmulDoubleCpuKernelFunc>(); }},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     []() { return std::make_shared<MatmulDoubleCpuKernelFunc>(); }},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     []() { return std::make_shared<MatmulDoubleCpuKernelFunc>(); }},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     []() { return std::make_shared<MatmulDoubleCpuKernelFunc>(); }},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     []() { return std::make_shared<MatmulDoubleCpuKernelFunc>(); }},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     []() { return std::make_shared<MatmulDoubleCpuKernelFunc>(); }},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     []() { return std::make_shared<MatmulDoubleCpuKernelFunc>(); }}}},
  {kMatMulBiasAdd,
   {{KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     []() { return std::make_shared<MatMulCpuKernelFunc>(); }}}},
  {kMatMulBiasAddRelu,
   {{KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     []() { return std::make_shared<MatMulCpuKernelFunc>(); }}}},
  {kBatchMatMul,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     []() { return std::make_shared<MatMulCpuKernelFunc>(); }},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     []() { return std::make_shared<MatmulDoubleCpuKernelFunc>(); }},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     []() { return std::make_shared<MatmulDoubleCpuKernelFunc>(); }},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     []() { return std::make_shared<MatmulDoubleCpuKernelFunc>(); }},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     []() { return std::make_shared<MatmulDoubleCpuKernelFunc>(); }},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     []() { return std::make_shared<MatmulDoubleCpuKernelFunc>(); }},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     []() { return std::make_shared<MatmulDoubleCpuKernelFunc>(); }},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     []() { return std::make_shared<MatmulDoubleCpuKernelFunc>(); }},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     []() { return std::make_shared<MatmulDoubleCpuKernelFunc>(); }},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     []() { return std::make_shared<MatmulDoubleCpuKernelFunc>(); }},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     []() { return std::make_shared<MatmulDoubleCpuKernelFunc>(); }},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     []() { return std::make_shared<MatmulDoubleCpuKernelFunc>(); }}}}};

template <typename T>
void LaunchEmptyTensor(const std::vector<AddressPtr> &outputs) {
  auto output = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  output[kIndex0] = 0;
}

static std::map<int, LaunchEmptyTensorFunc> empty_tensor_map_ = {
  {kNumberTypeFloat32, LaunchEmptyTensor<float>},       {kNumberTypeFloat64, LaunchEmptyTensor<double>},
  {kNumberTypeInt8, LaunchEmptyTensor<int8_t>},         {kNumberTypeInt16, LaunchEmptyTensor<int16_t>},
  {kNumberTypeInt32, LaunchEmptyTensor<int32_t>},       {kNumberTypeInt64, LaunchEmptyTensor<int64_t>},
  {kNumberTypeUInt8, LaunchEmptyTensor<uint8_t>},       {kNumberTypeUInt16, LaunchEmptyTensor<uint16_t>},
  {kNumberTypeUInt32, LaunchEmptyTensor<uint32_t>},     {kNumberTypeUInt64, LaunchEmptyTensor<uint64_t>},
  {kNumberTypeComplex64, LaunchEmptyTensor<complex64>}, {kNumberTypeComplex128, LaunchEmptyTensor<complex128>},
};
}  // namespace

std::vector<KernelAttr> MatMulCpuKernelMod::GetOpSupport() {
  auto iter = support_list_map.find(kernel_type_);
  if (iter == support_list_map.end()) {
    MS_LOG(EXCEPTION) << "Does not support " << kernel_type_ << "!";
  }

  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MatMulFuncCreator> &pair) { return pair.first; });
  return support_list;
}

bool MatMulCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (kernel_name_ != kernel_type_) {
    MS_LOG(EXCEPTION) << "Suppose to be " << kernel_type_ << " but got " << kernel_name_;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "MatMul does not support this kernel data type: " << kernel_attr;
  }

  func_obj_ = support_list_map[kernel_type_][index].second();
  func_obj_->InitFunc(base_operator, inputs, outputs);
  return true;
}

int MatMulCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = 0;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs)) != 0) {
    return ret;
  }
  auto shape0 = inputs[kIndex0]->GetShapeVector();
  is_empty_tensor_ = std::any_of(shape0.begin(), shape0.end(), [](const int64_t shape) { return shape == 0; });
  if (is_empty_tensor_) {
    auto dtype = inputs[kIndex0]->GetDtype();
    auto iter = empty_tensor_map_.find(dtype);
    if (iter == empty_tensor_map_.end()) {
      MS_LOG(EXCEPTION) << "Does not support " << TypeIdLabel(dtype) << "!";
    }
    launch_empty_tensor_func_ = empty_tensor_map_[dtype];
    return ret;
  }
  return func_obj_->Resize(base_operator, inputs, outputs, inputsOnHost);
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, MatMul,
                                 []() { return std::make_shared<MatMulCpuKernelMod>(kMatMul); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, BatchMatMul,
                                 []() { return std::make_shared<MatMulCpuKernelMod>(kBatchMatMul); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, FusedMatMulBiasAdd,
                                 []() { return std::make_shared<MatMulCpuKernelMod>(kMatMulBiasAdd); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, MatMulBiasAddReluFusion,
                                 []() { return std::make_shared<MatMulCpuKernelMod>(kMatMulBiasAddRelu); });
}  // namespace kernel
}  // namespace mindspore
