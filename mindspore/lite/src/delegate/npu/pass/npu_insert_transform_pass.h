/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_PASS_NPU_INSERT_TRANSFORM_PASS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_PASS_NPU_INSERT_TRANSFORM_PASS_H_
#include <vector>
#include "src/delegate/npu/op/npu_op.h"
#include "src/delegate/npu/pass/npu_base_pass.h"
namespace mindspore {
class NPUInsertTransformPass : public NPUBasePass {
 public:
  NPUInsertTransformPass() { name_ = "NPUInsertTransformPass"; }

  int Run(NPUGraph *subgraph) override;

 private:
  int GetInsertState(NPUOp *op);
  int InsertPreNodes(NPUOp *op, std::vector<NPUOp *> *trans_ops);
  int InsertPostNodes(NPUOp *op, std::vector<NPUOp *> *trans_ops);
  int InsertNode(NPUOp *op, NPUOp *post_op, size_t post_input_index, std::vector<NPUOp *> *trans_ops);
  int InsertForInputTensor(NPUOp *op, size_t in_tensor_index, NPUOp *pre_op, std::vector<NPUOp *> *trans_ops);
  int InsertForOutputTensor(NPUOp *op, NPUOp *post_op, size_t post_in_tensor_index, std::vector<NPUOp *> *trans_ops);

 private:
  int total = 0;
  std::vector<NPUOp *> *all_ops_;
  std::vector<mindspore::MSTensor *> *all_tensors_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_PASS_NPU_INSERT_TRANSFORM_PASS_H_
