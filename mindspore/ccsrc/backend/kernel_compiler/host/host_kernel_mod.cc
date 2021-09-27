/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/host/host_kernel_mod.h"

#include "runtime/mem.h"
#include "utils/ms_context.h"
#include "backend/kernel_compiler/common_utils.h"
#include "runtime/device/kernel_runtime.h"
#include "runtime/device/ascend/executor/host_dynamic_kernel.h"

namespace mindspore {
namespace kernel {
void HostKernelFactory::Register(const std::string &name, HostKernelCreater &&fun) {
  hostKernelMap_.emplace(name, std::move(fun));
}

std::shared_ptr<HostKernelMod> HostKernelFactory::Get(const std::string &name) {
  const auto &map = Get().hostKernelMap_;
  auto it = map.find(name);
  if (it != map.end() && it->second) {
    return (it->second)();
  }
  return nullptr;
}

HostKernelFactory &HostKernelFactory::Get() {
  static HostKernelFactory instance{};
  return instance;
}

const std::vector<size_t> &HostKernelMod::GetInputSizeList() const { return input_size_list_; }
const std::vector<size_t> &HostKernelMod::GetOutputSizeList() const { return output_size_list_; }
const std::vector<size_t> &HostKernelMod::GetWorkspaceSizeList() const { return workspace_size_list_; }
bool HostKernelMod::Init(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  size_t input_num = AnfAlgo::GetInputTensorNum(anf_node);
  size_t output_num = AnfAlgo::GetOutputTensorNum(anf_node);

  for (size_t i = 0; i < input_num; i++) {
    std::vector<size_t> shape_i = AnfAlgo::GetInputDeviceShape(anf_node, i);
    TypePtr type_ptr = TypeIdToType(AnfAlgo::GetInputDeviceDataType(anf_node, i));
    int64_t size_i = 1;
    if (!GetShapeSize(shape_i, type_ptr, &size_i)) {
      return false;
    }
    input_size_list_.push_back(LongToSize(size_i));
  }

  for (size_t i = 0; i < output_num; i++) {
    std::vector<size_t> shape_i = AnfAlgo::GetOutputDeviceShape(anf_node, i);
    TypePtr type_ptr = TypeIdToType(AnfAlgo::GetOutputDeviceDataType(anf_node, i));
    MS_EXCEPTION_IF_NULL(type_ptr);
    int64_t size_i = 1;
    if (!GetShapeSize(shape_i, type_ptr, &size_i)) {
      return false;
    }
    output_size_list_.push_back(LongToSize(size_i));
  }
  return true;
}
bool HostKernelMod::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                           const std::vector<AddressPtr> &, void *) {
  return true;
}
std::vector<TaskInfoPtr> HostKernelMod::GenTask(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                                const std::vector<AddressPtr> &, uint32_t) {
  return {};
}
}  // namespace kernel
}  // namespace mindspore
