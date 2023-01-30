/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/rts/rt_kernel_info.h"
#include <unordered_map>
#include <algorithm>
#include "include/common/utils/convert_utils.h"
#include "include/common/utils/utils.h"
#include "utils/ms_utils.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace kernel {
void RtKerDescFactory::Register(const std::string &name, RtKerDescCreater &&fun) {
  MS_EXCEPTION_IF_NULL(fun);
  if (fmap_.find(name) == fmap_.end()) {
    (void)fmap_.emplace(name, std::move(fun));
  }
}

std::shared_ptr<RtKerDesc> RtKerDescFactory::Create(const std::string &name) {
  const auto &map = Get().fmap_;
  auto it = map.find(name);
  if (it != map.end() && it->second) {
    return (it->second)();
  }
  return nullptr;
}

RtKerDescFactory &RtKerDescFactory::Get() {
  static RtKerDescFactory _this{};
  return _this;
}

static bool IsDefaultKernelInfo(const std::string &name) {
  static const std::set<std::string> white_list = {kStreamSwitchOpName, kStreamActiveOpName, kLabelSetOpName,
                                                   kLabelGotoOpName};
  return white_list.find(name) != white_list.end();
}

void GetRtKelInfo(const CNodePtr &kernel_node,
                  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list) {
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::string opNameLower = common::AnfAlgo::GetCNodeName(kernel_node);
  (void)std::transform(opNameLower.begin(), opNameLower.end(), opNameLower.begin(), ::tolower);

  auto ker_desc_ptr = RtKerDescFactory::Create(opNameLower);
  if (ker_desc_ptr != nullptr && !ker_desc_ptr->GetKernelInfo(kernel_node).empty()) {
    *kernel_info_list = ker_desc_ptr->GetKernelInfo(kernel_node);
    return;
  }
  // if can't find kernel info in kernel info database, use the default kernel info
  auto node_name = common::AnfAlgo::GetCNodeName(kernel_node);
  if (IsDefaultKernelInfo(node_name)) {
    auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    // set input infos
    auto input_num = AnfAlgo::GetInputElementNum(kernel_node);
    MS_EXCEPTION_IF_NULL(kernel_build_info_builder);
    kernel_build_info_builder->SetInputsFormat(std::vector<std::string>(input_num, kOpFormat_DEFAULT));
    kernel_build_info_builder->SetInputsKernelObjectType(
      std::vector<KernelObjectType>(input_num, KernelObjectType::TENSOR));
    std::vector<TypeId> input_types = {};
    for (size_t i = 0; i < input_num; i++) {
      input_types.push_back(common::AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, i));
    }
    kernel_build_info_builder->SetInputsDeviceType(input_types);
    // Kernel ops in while-list such as 'LabelSet' always return UMonad.
    kernel_build_info_builder->SetOutputsFormat({kOpFormat_DEFAULT});
    kernel_build_info_builder->SetOutputsDeviceType({TypeId::kObjectTypeUMonad});
    kernel_build_info_builder->SetOutputsKernelObjectType({KernelObjectType::TENSOR});
    // set other info
    kernel_build_info_builder->SetFusionType(kernel::kPatternOpaque);
    kernel_build_info_builder->SetProcessor(kernel::Processor::AICORE);
    kernel_build_info_builder->SetKernelType(KernelType::RT_KERNEL);
    kernel_info_list->push_back(kernel_build_info_builder->Build());
    return;
  }
  MS_LOG(DEBUG) << "Rt dose not have op [" << opNameLower << "].";
}
}  // namespace kernel
}  // namespace mindspore
