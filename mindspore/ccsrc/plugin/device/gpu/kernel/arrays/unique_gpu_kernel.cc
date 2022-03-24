/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/arrays/unique_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>

namespace mindspore {
namespace kernel {
namespace {
template <typename T, typename S>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateUniqueKernelPtr(const std::string &kernel_name) {
  return std::make_unique<cukernel::UniqueHelperGpuKernel<T, S>>(kernel_name);
}
using UniquePtrCreatorFunc = std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &)>;

const std::vector<std::pair<KernelAttr, UniquePtrCreatorFunc>> kernel_attr = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
   CreateUniqueKernelPtr<float, int>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
   CreateUniqueKernelPtr<half, int>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   CreateUniqueKernelPtr<int, int>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   CreateUniqueKernelPtr<int64_t, int64_t>}};
}  // namespace

bool UniqueGpuKernelMod::Init(const CNodePtr &kernel_node) {
  kernel_node_ = kernel_node;
  auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
  auto index = GetMatchKernelAttrIdxWithException(kernel_node, GetOpSupport());
  helper_ptr_ = std::move(kernel_attr[index].second(kernel_name));
  helper_ptr_->ResetResource();
  std::vector<std::vector<size_t>> input_shapes;
  std::vector<std::vector<size_t>> output_shapes;
  std::vector<size_t> shape = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 0);
  is_null_input_ = CHECK_SHAPE_NULL(shape, kernel_name, "input");
  if (is_null_input_) {
    InitSizeLists();
    return true;
  }
  input_shapes.emplace_back(shape);
  helper_ptr_->CalMemSize(input_shapes, output_shapes);
  InitSizeLists();
  is_need_updateop_ = true;
  return true;
}

std::vector<KernelAttr> UniqueGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, UniquePtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Unique, UniqueGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
