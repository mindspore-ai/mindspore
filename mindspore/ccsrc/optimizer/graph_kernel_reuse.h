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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_GRAPH_KERNEL_OP_REUSE_H
#define MINDSPORE_CCSRC_OPTIMIZER_GRAPH_KERNEL_OP_REUSE_H

#include <mindspore/ccsrc/session/anf_runtime_algorithm.h>
#include <unordered_map>
#include <string>
#include <vector>

#include "optimizer/optimizer.h"

namespace mindspore {
namespace opt {

// Common subexpression elimination.
class GraphKernelReuse {
 public:
  GraphKernelReuse() : count(0) {}
  virtual ~GraphKernelReuse() = default;

  bool operator()(const FuncGraphPtr &root, const OptimizerPtr &optimizer) {
    bool chg = ReuseGraphKernel(root, optimizer->resource()->manager());
    return chg;
  }

  bool CompareNode(const AnfNodePtr a, const AnfNodePtr other);
  bool DoReplace(const FuncGraphManagerPtr manager);

  bool ReuseGraphKernel(const FuncGraphPtr root, const FuncGraphManagerPtr manager);

 private:
  std::unordered_map<std::string, std::vector<FuncGraphPtr>> graph_kernel_ops;
  int count;
};

}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_OPTIMIZER_GRAPH_KERNEL_OP_REUSE_H
