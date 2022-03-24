/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/math/unary_op_gpu_kernel.h"
#include <map>
#include <functional>
#include <utility>
#include <algorithm>

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kAbs = "Abs";
constexpr auto kACos = "ACos";
constexpr auto kAcosh = "Acosh";
constexpr auto kAsin = "Asin";
constexpr auto kAsinh = "Asinh";
constexpr auto kAtan = "Atan";
constexpr auto kCos = "Cos";
constexpr auto kErf = "Erf";
constexpr auto kErfc = "Erfc";
constexpr auto kExp = "Exp";
constexpr auto kExpm1 = "Expm1";
constexpr auto kFloor = "Floor";
constexpr auto kLog = "Log";
constexpr auto kLog1p = "Log1p";
constexpr auto kNeg = "Neg";
constexpr auto kReciprocal = "Reciprocal";
constexpr auto kRint = "Rint";
constexpr auto kRound = "Round";
constexpr auto kRsqrt = "Rsqrt";
constexpr auto kSign = "Sign";
constexpr auto kSin = "Sin";
constexpr auto kSqrt = "Sqrt";
constexpr auto kSquare = "Square";

template <typename T>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateUnaryKernelPtr(const std::string &kernel_name) {
  return std::make_unique<cukernel::UnaryHelperGpuKernel<T>>(kernel_name);
}
using UnaryPtrCreatorFunc = std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &)>;

const std::map<std::string, std::vector<std::pair<KernelAttr, UnaryPtrCreatorFunc>>> kernel_attr_map = {
  {kExp,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateUnaryKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateUnaryKernelPtr<half>}}},
  {kExpm1,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateUnaryKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateUnaryKernelPtr<half>}}},
  {kLog,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateUnaryKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateUnaryKernelPtr<half>}}},
  {kLog1p,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateUnaryKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateUnaryKernelPtr<half>}}},
  {kErf,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateUnaryKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateUnaryKernelPtr<half>}}},
  {kErfc,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateUnaryKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateUnaryKernelPtr<half>}}},
  {kNeg,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateUnaryKernelPtr<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateUnaryKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateUnaryKernelPtr<half>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8), CreateUnaryKernelPtr<char>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8), CreateUnaryKernelPtr<uchar>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), CreateUnaryKernelPtr<int>}}},
  {kReciprocal,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateUnaryKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateUnaryKernelPtr<half>}}},
  {kSquare,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateUnaryKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateUnaryKernelPtr<half>}}},
  {kSqrt,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateUnaryKernelPtr<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateUnaryKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateUnaryKernelPtr<half>}}},
  {kRsqrt,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateUnaryKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateUnaryKernelPtr<half>}}},
  {kSin,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateUnaryKernelPtr<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateUnaryKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateUnaryKernelPtr<half>}}},
  {kAsin,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateUnaryKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateUnaryKernelPtr<half>}}},
  {kAsinh,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateUnaryKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateUnaryKernelPtr<half>}}},
  {kCos,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateUnaryKernelPtr<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateUnaryKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateUnaryKernelPtr<half>}}},
  {kACos,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateUnaryKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateUnaryKernelPtr<half>}}},
  {kAcosh,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateUnaryKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateUnaryKernelPtr<half>}}},
  {kAtan,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateUnaryKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateUnaryKernelPtr<half>}}},
  {kAbs,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateUnaryKernelPtr<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateUnaryKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateUnaryKernelPtr<half>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), CreateUnaryKernelPtr<int>}}},
  {kFloor,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateUnaryKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateUnaryKernelPtr<half>}}},
  {kRint,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateUnaryKernelPtr<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateUnaryKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateUnaryKernelPtr<half>}}},
  {kRound,
   {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), CreateUnaryKernelPtr<int>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateUnaryKernelPtr<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateUnaryKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateUnaryKernelPtr<half>}}},
  {kSign,
   {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), CreateUnaryKernelPtr<int>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateUnaryKernelPtr<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateUnaryKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateUnaryKernelPtr<half>}}}};
}  // namespace

bool UnaryOpGpuKernelMod::Init(const CNodePtr &kernel_node) {
  kernel_node_ = kernel_node;
  std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
  auto index = GetMatchKernelAttrIdxWithException(kernel_node, GetOpSupport());
  helper_ptr_ = std::move(kernel_attr_map.at(kernel_type_)[index].second(kernel_name));
  helper_ptr_->ResetResource();
  std::vector<std::vector<size_t>> input_shapes;
  std::vector<std::vector<size_t>> output_shapes;
  auto input_shape = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 0);
  auto output_shape = AnfAlgo::GetOutputDeviceShapeAdaptively(kernel_node, 0);
  is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name, "input");
  if (is_null_input_) {
    input_size_list_.emplace_back(0);
    output_size_list_.emplace_back(0);
    return true;
  }
  input_shapes.emplace_back(input_shape);
  output_shapes.emplace_back(output_shape);
  int flag = helper_ptr_->CalMemSize(input_shapes, output_shapes);
  if (flag != 0) {
    return false;
  }
  InitSizeLists();
  return true;
}

std::vector<KernelAttr> UnaryOpGpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_map.find(kernel_type_);
  if (iter == kernel_attr_map.end()) {
    MS_LOG(EXCEPTION) << "Unary gpu do not support " << kernel_type_;
  }

  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, UnaryPtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Abs, []() { return std::make_shared<UnaryOpGpuKernelMod>(kAbs); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, ACos,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kACos); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Acosh,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kAcosh); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Asin,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kAsin); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Asinh,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kAsinh); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Atan,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kAtan); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Cos, []() { return std::make_shared<UnaryOpGpuKernelMod>(kCos); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Erf, []() { return std::make_shared<UnaryOpGpuKernelMod>(kErf); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Erfc,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kErfc); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Exp, []() { return std::make_shared<UnaryOpGpuKernelMod>(kExp); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Expm1,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kExpm1); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Floor,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kFloor); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Log, []() { return std::make_shared<UnaryOpGpuKernelMod>(kLog); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Log1p,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kLog1p); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Neg, []() { return std::make_shared<UnaryOpGpuKernelMod>(kNeg); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Reciprocal,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kReciprocal); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Rint,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kRint); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Round,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kRound); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Rsqrt,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kRsqrt); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Sign,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kSign); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Sin, []() { return std::make_shared<UnaryOpGpuKernelMod>(kSin); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Sqrt,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kSqrt); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Square,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kSquare); });
}  // namespace kernel
}  // namespace mindspore
