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

#include "src/control_flow/control_subgraph_creator.h"
#ifndef CONTROLFLOW_TENSORLIST_CLIP
#include "src/control_flow/kernel/entrance_subgraph_kernel.h"
#include "src/control_flow/kernel/exit_subgraph_kernel.h"
#endif

namespace mindspore::lite {
#ifndef CONTROLFLOW_TENSORLIST_CLIP
kernel::SubGraphKernel *CreateControlSubgraph(const kernel::SubGraphType &type, kernel::MSKernel *kernel) {
  kernel::SubGraphKernel *sub_graph = nullptr;
  switch (type) {
    case kernel::kEntranceSubGraph: {
      sub_graph = kernel::EntranceSubGraphKernel::Create(kernel);
    } break;
    case kernel::kExitSubGraph: {
      sub_graph = kernel::ExitSubGraphKernel::Create(kernel);
    } break;
    default: {
      MS_LOG(ERROR) << "not support subgraph type: " << type;
    }
  }
  return sub_graph;
}
#else
kernel::SubGraphKernel *CreateControlSubgraph(const kernel::SubGraphType &type, kernel::MSKernel *kernel) {
  return nullptr;
}
#endif
}  // namespace mindspore::lite
