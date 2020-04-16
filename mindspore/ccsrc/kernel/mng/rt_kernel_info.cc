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

#include "kernel/mng/rt_kernel_info.h"
#include <unordered_map>
#include <algorithm>
#include "utils/convert_utils.h"
#include "utils/utils.h"
#include "common/utils.h"
#include "session/anf_runtime_algorithm.h"

namespace mindspore {
namespace kernel {
void RtKerDescFactory::Register(const std::string &name, RtKerDescCreater &&fun) {
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
  static RtKerDescFactory _this;
  return _this;
}

void GetRtKelInfo(const CNodePtr &kernel_node,
                  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list) {
  MS_LOG(INFO) << "Mng kernel Info.";
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::string opNameLower = AnfAlgo::GetCNodeName(kernel_node);
  (void)std::transform(opNameLower.begin(), opNameLower.end(), opNameLower.begin(), ::tolower);

  auto ker_desc_ptr = RtKerDescFactory::Create(opNameLower);
  if (ker_desc_ptr == nullptr) {
    MS_LOG(DEBUG) << "Mng can't find op [" << opNameLower << "].";
    return;
  }
  MS_EXCEPTION_IF_NULL(ker_desc_ptr);
  auto kernel_info = ker_desc_ptr->GetKernelInfo();
  if (kernel_info.empty()) {
    MS_LOG(DEBUG) << "Rt dose not have op [" << opNameLower << "].";
    return;
  }
  *kernel_info_list = kernel_info;
}
}  // namespace kernel
}  // namespace mindspore
