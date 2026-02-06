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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_BATCH_MATMUL_REDUCESCATTER_ALLTOALL_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_BATCH_MATMUL_REDUCESCATTER_ALLTOALL_FUSION_H_

#include <memory>
#include <string>
#include <vector>
#include "include/backend/common/pass_manager/node_pass.h"
#include "primitive/other_ops.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace opt {
class BACKEND_COMMON_EXPORT BatchMatMulReduceScatterAllToAllFusion : public NodePass {
 public:
  explicit BatchMatMulReduceScatterAllToAllFusion(const std::string &name = "batchmatmul_reducescatter_alltoall_fusion")
      : NodePass(name) {}
  ~BatchMatMulReduceScatterAllToAllFusion() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
  AnfNodePtr Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) override;
  std::vector<std::string> MustExistPrimitiveName() const {
    return {prim::kPrimAlltoAll->name(), prim::kPrimAlltoAllV->name()};
  }

 private:
  void ClearAttr();
  bool InferInputDimAndSplitConcatDimByAllToAllCNode();
  CNodePtr PreProcessAndCreateBatchMatMulReduceScatterAllToAllCNode(const FuncGraphPtr func_graph);
  bool IsValidAllToAll(const AnfNodePtr &node);
  CNodePtr alltoall_cnode_;
  CNodePtr reducescatter_cnode_;
  CNodePtr batch_matmul_cnode_;
  CNodePtr bias_add_cnode_;
  int64_t split_dim_;
  int64_t concat_dim_;
  int64_t hidden_size_;
  int64_t ep_world_size_;
  int64_t tp_world_size_;
  int64_t y_shard_type_;
  int64_t mc2_fusion_level_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_BATCH_MATMUL_REDUCESCATTER_ALLTOALL_FUSION_H_
