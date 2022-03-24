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

#include "plugin/device/gpu/kernel/arrays/batchtospace_gpu_kernel.h"
#include <algorithm>
#include <utility>

namespace mindspore {
namespace kernel {
namespace {
template <typename T>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateBTSKernelPtr(const std::string &kernel_name) {
  return std::make_unique<cukernel::BatchToSpaceHelperGpuKernel<T>>(kernel_name);
}
using BTSPtrCreatorFunc = std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &)>;

const std::vector<std::pair<KernelAttr, BTSPtrCreatorFunc>> kernel_attr = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateBTSKernelPtr<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateBTSKernelPtr<half>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), CreateBTSKernelPtr<int>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64), CreateBTSKernelPtr<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16), CreateBTSKernelPtr<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8), CreateBTSKernelPtr<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8), CreateBTSKernelPtr<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16), CreateBTSKernelPtr<uint16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32), CreateBTSKernelPtr<uint32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64), CreateBTSKernelPtr<uint64_t>}};
}  // namespace

bool BatchToSpaceGpuKernelMod::Init(const CNodePtr &kernel_node) {
  kernel_node_ = kernel_node;
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);

  auto index = GetMatchKernelAttrIdxWithException(kernel_node, GetOpSupport());
  helper_ptr_ = std::move(kernel_attr[index].second(kernel_name_));
  helper_ptr_->ResetResource();

  std::vector<std::vector<size_t>> input_shapes;
  std::vector<std::vector<size_t>> output_shapes;
  auto input_shape = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 0);
  auto output_shape = AnfAlgo::GetOutputDeviceShapeAdaptively(kernel_node, 0);
  input_shapes.emplace_back(input_shape);
  output_shapes.emplace_back(output_shape);
  attr_.block_size = GetAttr<int64_t>(kernel_node, "block_size");
  attr_.crops = GetAttr<std::vector<std::vector<int64_t>>>(kernel_node, "crops");
  attr_.input_shape = input_shape;
  int flag = helper_ptr_->CheckKernelParam(&attr_);
  if (flag != 0) {
    return false;
  }

  flag = helper_ptr_->CalMemSize(input_shapes, output_shapes);
  if (flag != 0) {
    return false;
  }
  InitSizeLists();
  return true;
}

std::vector<KernelAttr> BatchToSpaceGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, BTSPtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, BatchToSpace, BatchToSpaceGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
