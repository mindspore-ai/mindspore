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

#include "plugin/device/ascend/kernel/host/host_kernel_mod.h"

#include "runtime/mem.h"
#include "utils/ms_context.h"
#include "kernel/common_utils.h"
#include "runtime/device/kernel_runtime.h"

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

bool HostKernelMod::Init(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  input_size_list_.clear();
  output_size_list_.clear();

  size_t input_num = common::AnfAlgo::GetInputTensorNum(anf_node);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(anf_node);

  for (size_t i = 0; i < input_num; i++) {
    auto shape_i = AnfAlgo::GetInputDeviceShape(anf_node, i);
    TypePtr type_ptr = TypeIdToType(AnfAlgo::GetInputDeviceDataType(anf_node, i));
    int64_t size_i = 1;
    if (!GetShapeSize(shape_i, type_ptr, &size_i)) {
      return false;
    }
    input_size_list_.push_back(LongToSize(size_i));
  }

  for (size_t i = 0; i < output_num; i++) {
    auto shape_i = AnfAlgo::GetOutputDeviceShape(anf_node, i);
    TypePtr type_ptr = TypeIdToType(AnfAlgo::GetOutputDeviceDataType(anf_node, i));
    MS_EXCEPTION_IF_NULL(type_ptr);
    int64_t size_i = 1;
    if (!GetShapeSize(shape_i, type_ptr, &size_i)) {
      return false;
    }
    output_size_list_.push_back(LongToSize(size_i));
  }
  anf_node_ = anf_node;
  return true;
}

bool HostKernelMod::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                           const std::vector<AddressPtr> &, void *) {
  return true;
}

int HostKernelMod::Resize(const BaseOperatorPtr &, const std::vector<KernelTensorPtr> &,
                          const std::vector<KernelTensorPtr> &, const std::map<uint32_t, tensor::TensorPtr> &) {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  if (!Init(cnode)) {
    MS_LOG(EXCEPTION) << "Init failed, node:" << cnode->fullname_with_scope();
  }
  return 0;
}

std::vector<TaskInfoPtr> HostKernelMod::GenTask(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                                const std::vector<AddressPtr> &, uint32_t) {
  return {};
}
}  // namespace kernel
}  // namespace mindspore
