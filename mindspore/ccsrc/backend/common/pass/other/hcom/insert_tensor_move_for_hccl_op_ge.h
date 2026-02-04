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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_GE_HCOM_INSERT_TENSOR_MOVE_FOR_HCCL_OP_GE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_GE_HCOM_INSERT_TENSOR_MOVE_FOR_HCCL_OP_GE_H_

#include <memory>
#include "include/backend/common/pass_manager/optimizer.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace opt {
class BACKEND_COMMON_EXPORT InsertTensorMoveForHcclOpGe : public PatternProcessPass {
 public:
  explicit InsertTensorMoveForHcclOpGe(bool multigraph = true)
      : PatternProcessPass("insert_tensor_move_for_hccl_op_ge", multigraph) {}
  ~InsertTensorMoveForHcclOpGe() override = default;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  void InsertTensorMove(const FuncGraphPtr &graph, const CNodePtr &hccl_node) const;
  bool NeedInsertTensorMove(const FuncGraphPtr &graph, const AnfNodePtr &input, size_t input_idx,
                            const CNodePtr &cur_node) const;
  bool NeedInsertTensorMoveForSpecialCase(const AnfNodePtr &input, const CNodePtr &cur_node) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_GE_HCOM_INSERT_TENSOR_MOVE_FOR_HCCL_OP_GE_H_
