/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "tools/graph_kernel/converter/update_kernel_info.h"
#include "tools/graph_kernel/common/utils.h"

namespace mindspore::graphkernel {
bool UpdateKernelInfo::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  bool changed = false;
  const auto &params = func_graph->parameters();
  for (const auto &param : params) {
    if (param == nullptr) {
      continue;
    }
    auto build_info = GetKernelInfo(param);
    if (build_info == nullptr) {
      continue;
    }
    auto out_types = build_info->GetAllOutputDeviceTypes();
    auto out_formats = build_info->GetAllOutputFormats();
    if (out_types.size() != out_formats.size()) {
      MS_LOG(INFO) << "Clear kernel info for node: " << param->fullname_with_scope();
      param->set_kernel_info(nullptr);
      changed = true;
    }
  }
  return changed;
}
}  // namespace mindspore::graphkernel
