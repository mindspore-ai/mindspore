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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_PASS_NPU_TRANS_ADDITION_PASS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_PASS_NPU_TRANS_ADDITION_PASS_H_

#include <set>
#include <vector>
#include "src/delegate/npu/op/npu_op.h"
#include "src/delegate/npu/pass/npu_base_pass.h"
#include "src/common/log_util.h"

namespace mindspore {
class NPUTransAdditionPass : public NPUBasePass {
 public:
  NPUTransAdditionPass() { name_ = "NPUTransAdditionPass"; }

  int Run(NPUGraph *subgraph) override;

 private:
  int InsertTransNodes(NPUOp *op, std::vector<NPUOp *> *trans_ops, std::vector<mindspore::MSTensor> graph_outputs);

 private:
  int total_ = 0;
  std::vector<NPUOp *> *all_ops_;
  std::vector<mindspore::MSTensor *> *all_tensors_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_PASS_NPU_TRANS_ADDITION_PASS_H_
