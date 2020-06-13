
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
#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_PASS_FUSE_COMPOSITE_H_
#define MINDSPORE_CCSRC_PRE_ACTIVATE_PASS_FUSE_COMPOSITE_H_

#include <set>
#include <string>
#include <vector>
#include <memory>
#include "pre_activate/common/optimizer.h"
#include "session/kernel_graph.h"

namespace mindspore {
namespace opt {
enum CompositeType {
  ELEWISE = 0,  // only contain elewise basic ops
  REDUCE,       // contain reduce ops
  CUBE,         // contain cube ops
};
struct CompositeInfo {
  CompositeType op_type = ELEWISE;
  bool is_before_kernel_select = false;
  int reduce_op_num = 0;
  int cal_step = 0;
};

// when reduce composite's cal step is greater than this number, not fuse
const int MAX_REDUCE_OP_FUSION_CAL_STEP = 5;
// when reduce composite contain reduce op num is greater than this number, not fuse
const int MAX_REDUCE_OP_FUSION_REDUCE_NUM = 2;

const std::set<std::string> composite_black_list = {"BNTrainingUpdateSum", "ApplyMomentum", "LayerNormForward",
                                                    "LambNextMV", "LambUpdateWithLR"};

std::vector<AnfNodePtr> RemoveCircle(const std::vector<AnfNodePtr> &fused_op, bool is_backward = true);

void TopoSortForNodeList(std::vector<AnfNodePtr> *lst);

AnfNodePtr CreateNewFuseCNode(const std::shared_ptr<session::KernelGraph> &kernel_graph, const FuncGraphPtr &fg,
                              const AnfNodePtrList &inputs, const AnfNodePtrList &outputs,
                              bool is_before_kernel_select);

void ReplaceNewFuseCNode(const std::shared_ptr<session::KernelGraph> &kernel_graph, const AnfNodePtr &new_fuse_cnode,
                         const AnfNodePtrList &outputs);

void FuseComposite(const std::shared_ptr<session::KernelGraph> &kernel_graph, bool is_before_kernel_select = false);
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_PASS_FUSE_COMPOSITE_H_
