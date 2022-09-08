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
#ifndef MINDSPORE_INSERT_TENSOR_MOVE_FOR_REF_H
#define MINDSPORE_INSERT_TENSOR_MOVE_FOR_REF_H

#include <memory>
#include "backend/common/optimizer/optimizer.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"

namespace mindspore {
namespace opt {
// When RefNode's output is a GraphOutput, need insert a TensorMove
class InsertTensorMoveForGraphOutputRefNode : public Pass {
 public:
  InsertTensorMoveForGraphOutputRefNode()
      : Pass("insert_tensor_move_for_graphoutput_ref_node"), kernel_select_(std::make_shared<KernelSelect>()) {}
  ~InsertTensorMoveForGraphOutputRefNode() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  KernelSelectPtr kernel_select_;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_INSERT_TENSOR_MOVE_FOR_REF_H
