/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ALLTOALL_ALLGATHER_BATCH_MATMUL_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ALLTOALL_ALLGATHER_BATCH_MATMUL_FUSION_H_

#include <memory>
#include <utility>
#include <string>
#include <vector>
#include "include/backend/common/pass_manager/node_pass.h"
#include "primitive/other_ops.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace opt {
using AnfNodeIndex = std::pair<AnfNodePtr, int>;
using AnfNodeIndexList = std::vector<AnfNodeIndex>;
class BACKEND_COMMON_EXPORT AllToAllAllGatherBatchMatMulFusion : public NodePass {
 public:
  explicit AllToAllAllGatherBatchMatMulFusion(const std::string &name = "alltoall_allgather_batch_matmul_fusion")
      : NodePass(name) {}
  ~AllToAllAllGatherBatchMatMulFusion() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
  AnfNodePtr Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) override;
  std::vector<std::string> MustExistPrimitiveName() const {
    return {prim::kPrimAlltoAll->name(), prim::kPrimAlltoAllV->name()};
  }

 private:
  void InitAttr();
  CNodePtr PreProcessAndCreateAllToAllAllGatherBatchMatMulCNode(const FuncGraphPtr func_graph, bool output_y2_flag,
                                                                bool output_y3_flag);
  void SetOutputTypeAndShapeForAllToAllAllGatherBatchMatMul(const AnfNodePtr &alltoall_allgather_batch_matmul_node,
                                                            bool output_y2_flag, bool output_y3_flag);
  void InferInputDimByAllToAllCNode();
  CNodePtr FindReshapeBeforeBatchMatMul(const FuncGraphPtr &func_graph, const AnfNodePtr &node);
  bool IsValidAllToAll(const AnfNodePtr &node);
  void ReplaceGraph(const FuncGraphPtr &func_graph, const CNodePtr &fusion_cnode,
                    const AnfNodeIndexList &allgather_last_node_other_users,
                    const AnfNodeIndexList &bias_add_other_users, bool output_y2_flag, bool output_y3_flag);
  void FindFirstNonReshapeAllToAllUser(const AnfNodePtr &node, bool *is_grad_);
  CNodePtr reshape_before_batch_matmul_cnode_;
  CNodePtr alltoall_cnode_;
  CNodePtr allgather_cnode_;
  CNodePtr batch_matmul_cnode_;
  CNodePtr bias_add_cnode_;
  CNodePtr act_cnode_;
  CNodePtr allgather_last_cnode_;
  CNodePtr last_cnode_;
  bool with_bias_add_;
  bool with_act_calc_;
  bool is_grad_;
  int64_t allgather_gather_dim_;
  int64_t split_dim_;
  int64_t concat_dim_;
  int64_t expert_size_;
  int64_t capacity_size_;
  int64_t hidden_size_;
  int64_t ep_world_size_;
  int64_t tp_world_size_;
  int64_t x_shard_type_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ALLTOALL_ALLGATHER_BATCH_MATMUL_FUSION_H_
