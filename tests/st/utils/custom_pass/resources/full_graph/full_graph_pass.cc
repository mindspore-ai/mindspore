/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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
#include "full_graph/full_graph_pass.h"
#include "mindspore/core/include/utils/log_adapter.h"

namespace mindspore {
namespace opt {
bool FullGraphPass::Run(const FuncGraphPtr &func_graph) {
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  MS_LOG(INFO) << "FullGraphPass Run successfully";
  return true;
}

}  // namespace opt
}  // namespace mindspore
