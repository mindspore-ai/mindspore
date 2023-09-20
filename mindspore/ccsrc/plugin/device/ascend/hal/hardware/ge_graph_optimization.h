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

#ifndef MINDSPORE_GE_GRAPH_OPTIMIZATION_H
#define MINDSPORE_GE_GRAPH_OPTIMIZATION_H
#include <vector>
#include <set>
#include <memory>
#include "include/backend/kernel_graph.h"
#include "include/backend/optimizer/graph_optimizer.h"
namespace mindspore {
namespace device {
namespace ascend {
class GEGraphOptimization {
 public:
  static GEGraphOptimization &GetInstance() {
    static GEGraphOptimization instance;
    return instance;
  }
  void OptimizeGEGraph(const KernelGraphPtr &graph);
  void OptimizeACLGraph(const KernelGraphPtr &graph);
  void OptimizeACLGraphAfterKernelSelect(const KernelGraphPtr &graph);
  void UnifyMindIR(const KernelGraphPtr &graph);
  void GEMindIRPass(const KernelGraphPtr &graph) const;

 private:
  GEGraphOptimization() {}
  ~GEGraphOptimization() = default;
  GEGraphOptimization(const GEGraphOptimization &) = delete;
  GEGraphOptimization &operator=(const GEGraphOptimization &) = delete;
};

}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_GE_GRAPH_OPTIMIZATION_H
