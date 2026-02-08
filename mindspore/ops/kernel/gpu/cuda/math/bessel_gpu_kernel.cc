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

#include <algorithm>
#include <utility>
#include "kernel/gpu/cuda/math/bessel_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kBesselJ0 = "BesselJ0";
constexpr auto kBesselJ1 = "BesselJ1";
constexpr auto kBesselK0 = "BesselK0";
constexpr auto kBesselK0e = "BesselK0e";
constexpr auto kBesselK1 = "BesselK1";
constexpr auto kBesselK1e = "BesselK1e";
constexpr auto kBesselI0 = "BesselI0";
constexpr auto kBesselI0e = "BesselI0e";
constexpr auto kBesselI1 = "BesselI1";
constexpr auto kBesselI1e = "BesselI1e";
constexpr auto kBesselY0 = "BesselY0";
constexpr auto kBesselY1 = "BesselY1";

template <typename T>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateBesselKernelPtr(const std::string &kernel_name,
                                                                     const uint32_t &device_id) {
  return std::make_unique<cukernel::BesselHelperGpuKernel<T>>(kernel_name, device_id);
}
using BesselPtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;

const std::map<std::string, std::vector<std::pair<KernelAttr, BesselPtrCreatorFunc>>> kernel_attr_map = {
  {kBesselI0,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateBesselKernelPtr<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateBesselKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateBesselKernelPtr<half>}}},
  {kBesselI0e,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateBesselKernelPtr<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateBesselKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateBesselKernelPtr<half>}}},
  {kBesselI1,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateBesselKernelPtr<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateBesselKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateBesselKernelPtr<half>}}},
  {kBesselI1e,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateBesselKernelPtr<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateBesselKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateBesselKernelPtr<half>}}},
  {kBesselJ0,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateBesselKernelPtr<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateBesselKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateBesselKernelPtr<half>}}},
  {kBesselJ1,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateBesselKernelPtr<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateBesselKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateBesselKernelPtr<half>}}},
  {kBesselK0,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateBesselKernelPtr<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateBesselKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateBesselKernelPtr<half>}}},
  {kBesselK0e,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateBesselKernelPtr<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateBesselKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateBesselKernelPtr<half>}}},
  {kBesselK1,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateBesselKernelPtr<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateBesselKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateBesselKernelPtr<half>}}},
  {kBesselK1e,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateBesselKernelPtr<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateBesselKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateBesselKernelPtr<half>}}},
  {kBesselY0,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateBesselKernelPtr<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateBesselKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateBesselKernelPtr<half>}}},
  {kBesselY1,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateBesselKernelPtr<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateBesselKernelPtr<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateBesselKernelPtr<half>}}}};
}  // namespace

bool BesselGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                                const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  std::vector<void *> input_ptrs = ConvertPtrs(inputs);
  std::vector<void *> work_ptrs = ConvertPtrs(workspace);
  std::vector<void *> output_ptrs = ConvertPtrs(outputs);
  if (helper_ptr_->Process(input_ptrs, output_ptrs, work_ptrs, stream_ptr) != 0) {
    return false;
  }
  return true;
}

bool BesselGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(tensor_attr, GetOpSupport());
  if (!is_match) {
    return false;
  }
  helper_ptr_ = std::move(kernel_attr_map.at(kernel_type_)[index].second(kernel_name_, device_id_));
  Resize(inputs, outputs);
  return true;
}

int BesselGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<int64_t> inp_shape = inputs[0]->GetShapeVector();
  std::vector<int64_t> out_shape = outputs[0]->GetShapeVector();
  input_shapes.emplace_back(inp_shape);
  output_shapes.emplace_back(out_shape);
  if (helper_ptr_->CalMemSize(input_shapes, output_shapes) == -1) {
    return KRET_RESIZE_FAILED;
  }
  output_size_list_ = helper_ptr_->GetOutputSizeList();
  workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  return KRET_OK;
}

std::vector<KernelAttr> BesselGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  auto iter = kernel_attr_map.find(kernel_type_);
  if (iter == kernel_attr_map.end()) {
    MS_LOG(ERROR) << "For 'BesselOp', only support these types: "
                  << kernel::Map2Str<std::map, std::vector<std::pair<KernelAttr, BesselPtrCreatorFunc>>>(
                       kernel_attr_map)
                  << " currently, but got " << kernel_name_;
    return support_list;
  }
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, BesselPtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, BesselJ0,
                                 []() { return std::make_shared<BesselGpuKernelMod>(kBesselJ0); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, BesselJ1,
                                 []() { return std::make_shared<BesselGpuKernelMod>(kBesselJ1); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, BesselK0,
                                 []() { return std::make_shared<BesselGpuKernelMod>(kBesselK0); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, BesselK0e,
                                 []() { return std::make_shared<BesselGpuKernelMod>(kBesselK0e); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, BesselK1,
                                 []() { return std::make_shared<BesselGpuKernelMod>(kBesselK1); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, BesselK1e,
                                 []() { return std::make_shared<BesselGpuKernelMod>(kBesselK1e); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, BesselI0,
                                 []() { return std::make_shared<BesselGpuKernelMod>(kBesselI0); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, BesselI0e,
                                 []() { return std::make_shared<BesselGpuKernelMod>(kBesselI0e); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, BesselI1,
                                 []() { return std::make_shared<BesselGpuKernelMod>(kBesselI1); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, BesselI1e,
                                 []() { return std::make_shared<BesselGpuKernelMod>(kBesselI1e); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, BesselY0,
                                 []() { return std::make_shared<BesselGpuKernelMod>(kBesselY0); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, BesselY1,
                                 []() { return std::make_shared<BesselGpuKernelMod>(kBesselY1); });
}  // namespace kernel
}  // namespace mindspore
