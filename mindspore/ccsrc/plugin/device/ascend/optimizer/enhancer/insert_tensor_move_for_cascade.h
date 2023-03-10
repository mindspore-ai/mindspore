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
#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_ENHANCER_INSERT_TENSORMOVE_ASYNC_FOR_CASCADE_H_
#define MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_ENHANCER_INSERT_TENSORMOVE_ASYNC_FOR_CASCADE_H_

#include <memory>
#include "include/backend/optimizer/optimizer.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"

namespace mindspore {
namespace opt {
class InsertTensorMoveForCascade : public PatternProcessPass {
 public:
  explicit InsertTensorMoveForCascade(bool multigraph = true)
      : PatternProcessPass("insert_tensor_move_for_cascade", multigraph),
        kernel_select_(std::make_shared<KernelSelect>()) {}
  ~InsertTensorMoveForCascade() override = default;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  AnfNodePtr InsertTensorMove(const FuncGraphPtr &graph, const CNodePtr &hccl_node) const;
  void InsertOutputTensorMove(const FuncGraphPtr &graph, const CNodePtr &hccl_node) const;
  KernelSelectPtr kernel_select_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_ENHANCER_INSERT_TENSORMOVE_ASYNC_FOR_OP_CASCADE_H_
