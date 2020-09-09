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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_COMPOSITE_OPS_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_COMPOSITE_OPS_FUSION_H_

#include <set>
#include <string>
#include <vector>
#include <memory>
#include <limits>
#include "backend/optimizer/common/optimizer.h"
#include "backend/session/kernel_graph.h"

namespace mindspore {
namespace opt {
enum GraphKernelType {
  ELEWISE = 0,  // only contain elewise basic ops
  REDUCE,       // contain reduce ops
  CUBE,         // contain cube ops
};

struct GraphKernelInfo {
  GraphKernelType op_type = ELEWISE;
  bool is_before_kernel_select = false;
  int reduce_op_num = 0;
  int cal_step = 0;
  std::string origin_composite_name = "";
};

// when composite fuse composite the cal step is greate than this number, not fuse
#if ENABLE_D
const int MAX_REDUCE_OP_FUSION_CAL_STEP = 5;
const int MAX_REDUCE_OP_FUSION_REDUCE_NUM = 2;
#endif
const std::set<std::string> graph_kernel_black_list = {"BNTrainingUpdateSum", "ApplyMomentum", "LayerNormForward",
                                                       "LambNextMV", "LambUpdateWithLR"};

std::vector<AnfNodePtr> RemoveCircle(const std::vector<AnfNodePtr> &fused_op, bool is_backward = true);

void TopoSortForNodeList(std::vector<AnfNodePtr> *lst);

bool FuseCompositeOps(const std::shared_ptr<session::KernelGraph> &kernel_graph, bool is_before_kernel_select = false);

class CompositeOpsFusion : public Pass {
 public:
  CompositeOpsFusion() : Pass("composite_ops_fusion") {}
  ~CompositeOpsFusion() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};
using FuseGraphKernelPassPtr = std::shared_ptr<CompositeOpsFusion>;
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_COMPOSITE_OPS_FUSION_H_
