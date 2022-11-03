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

#include "src/litert/pass/delete_isolated_kernel.h"
#include <set>
#include <queue>
#include "src/litert/kernel_exec_util.h"

namespace mindspore::lite::pass {
int DeleteIsolatedKernel::Run(kernel::SubGraphKernel *subgraph, std::vector<Tensor *> *) {
  subgraph->SetInNodes(kernel::KernelExecUtil::SubgraphInputNodes(subgraph->nodes()));

  std::set<kernel::KernelExec *> visited;  // record the kernel that will be executed
  std::queue<kernel::KernelExec *> kernel_queue;

  for (auto in_kernel : subgraph->in_nodes()) {
    kernel_queue.push(in_kernel);
    (void)visited.insert(in_kernel);
  }
  while (!kernel_queue.empty()) {
    auto kernel = kernel_queue.front();
    kernel_queue.pop();

    for (auto out_kernel : kernel->out_kernels()) {
      if (visited.find(out_kernel) != visited.end()) {
        continue;
      }
      kernel_queue.push(out_kernel);
      (void)visited.insert(out_kernel);
    }
  }

  auto kernels = subgraph->nodes();
  for (auto kernel : kernels) {
    if (visited.find(kernel) == visited.end()) {
      subgraph->DropNode(kernel);
      delete kernel;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite::pass
