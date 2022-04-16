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

#include <functional>
#include <map>
#include <utility>
#include <algorithm>
#include "plugin/device/gpu/kernel/math/unary_op_grad_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t INPUT_NUM = 2;
constexpr size_t OUTPUT_NUM = 1;
constexpr auto kSqrtGrad = "SqrtGrad";
constexpr auto kRsqrtGrad = "RsqrtGrad";
constexpr auto kAsinGrad = "AsinGrad";
constexpr auto kACosGrad = "ACosGrad";
constexpr auto kAtanGrad = "AtanGrad";
constexpr auto kAsinhGrad = "AsinhGrad";
constexpr auto kAcoshGrad = "AcoshGrad";
constexpr auto kReciprocalGrad = "ReciprocalGrad";
constexpr auto kInvGrad = "InvGrad";

template <typename T>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateUnaryGradKernelPtr(const std::string &kernel_name,
                                                                        const uint32_t &device_id) {
  return std::make_unique<cukernel::UnaryGradHelperGpuKernel<T>>(kernel_name, device_id);
}
using UnaryGradPtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;

const std::map<std::string, std::vector<std::pair<KernelAttr, UnaryGradPtrCreatorFunc>>> kernel_attr_map = {
  {kSqrtGrad,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     CreateUnaryGradKernelPtr<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     CreateUnaryGradKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     CreateUnaryGradKernelPtr<half>}}},
  {kRsqrtGrad,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     CreateUnaryGradKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     CreateUnaryGradKernelPtr<half>}}},
  {kAsinGrad,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     CreateUnaryGradKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     CreateUnaryGradKernelPtr<half>}}},
  {kACosGrad,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     CreateUnaryGradKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     CreateUnaryGradKernelPtr<half>}}},
  {kAtanGrad,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     CreateUnaryGradKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     CreateUnaryGradKernelPtr<half>}}},
  {kAsinhGrad,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     CreateUnaryGradKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     CreateUnaryGradKernelPtr<half>}}},
  {kAcoshGrad,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     CreateUnaryGradKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     CreateUnaryGradKernelPtr<half>}}},
  {kReciprocalGrad,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     CreateUnaryGradKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     CreateUnaryGradKernelPtr<half>}}},
  {kInvGrad,
   {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     CreateUnaryGradKernelPtr<int>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     CreateUnaryGradKernelPtr<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     CreateUnaryGradKernelPtr<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     CreateUnaryGradKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     CreateUnaryGradKernelPtr<half>}}}};
}  // namespace

bool UnaryGradOpGpuKernelMod::Init(const CNodePtr &kernel_node) {
  kernel_node_ = kernel_node;
  std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != INPUT_NUM) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 2, but got " << input_num;
  }
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != OUTPUT_NUM) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of outputs should be 1, but got " << output_num;
  }
  auto index = GetMatchKernelAttrIdxWithException(kernel_node, GetOpSupport());
  helper_ptr_ = std::move(kernel_attr_map.at(kernel_type_)[index].second(kernel_name, deprecated_deviced_id_));
  auto input_shape = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 0);
  auto dx_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  auto output_shape = AnfAlgo::GetOutputDeviceShapeAdaptively(kernel_node, 0);
  is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name, "input");
  if (is_null_input_) {
    input_size_list_.emplace_back(0);
    output_size_list_.emplace_back(0);
    return true;
  }
  is_null_input_ = CHECK_SHAPE_NULL(dx_shape, kernel_name, "input");
  if (is_null_input_) {
    input_size_list_.emplace_back(0);
    output_size_list_.emplace_back(0);
    return true;
  }
  std::vector<int64_t> int64_inp_shape;
  std::vector<int64_t> int64_dx_shape;
  std::vector<int64_t> int64_out_shape;
  std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(int64_inp_shape), SizeToLong);
  std::transform(dx_shape.begin(), dx_shape.end(), std::back_inserter(int64_dx_shape), SizeToLong);
  std::transform(output_shape.begin(), output_shape.end(), std::back_inserter(int64_out_shape), SizeToLong);
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  input_shapes.emplace_back(int64_inp_shape);
  input_shapes.emplace_back(int64_dx_shape);
  output_shapes.emplace_back(int64_out_shape);
  int flag = helper_ptr_->CalMemSize(input_shapes, output_shapes);
  if (flag != 0) {
    return false;
  }
  InitSizeLists();
  return true;
}

std::vector<KernelAttr> UnaryGradOpGpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_map.find(kernel_type_);
  if (iter == kernel_attr_map.end()) {
    MS_LOG(EXCEPTION) << "UnaryGrad gpu do not support " << kernel_type_;
  }

  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, UnaryGradPtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, SqrtGrad,
                                 []() { return std::make_shared<UnaryGradOpGpuKernelMod>(kSqrtGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, RsqrtGrad,
                                 []() { return std::make_shared<UnaryGradOpGpuKernelMod>(kRsqrtGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, AsinGrad,
                                 []() { return std::make_shared<UnaryGradOpGpuKernelMod>(kAsinGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, ACosGrad,
                                 []() { return std::make_shared<UnaryGradOpGpuKernelMod>(kACosGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, AtanGrad,
                                 []() { return std::make_shared<UnaryGradOpGpuKernelMod>(kAtanGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, AsinhGrad,
                                 []() { return std::make_shared<UnaryGradOpGpuKernelMod>(kAsinhGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, AcoshGrad,
                                 []() { return std::make_shared<UnaryGradOpGpuKernelMod>(kAcoshGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, ReciprocalGrad,
                                 []() { return std::make_shared<UnaryGradOpGpuKernelMod>(kReciprocalGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, InvGrad,
                                 []() { return std::make_shared<UnaryGradOpGpuKernelMod>(kInvGrad); });
}  // namespace kernel
}  // namespace mindspore
