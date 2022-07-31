/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_PASS_COREML_FORMAT_TRANS_PASS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_PASS_COREML_FORMAT_TRANS_PASS_H_

#include <set>
#include <vector>
#include "src/litert/delegate/coreml/op/coreml_op.h"
#include "src/litert/delegate/coreml/pass/coreml_base_pass.h"

namespace mindspore::lite {
class CoreMLFormatTransPass : public CoreMLBasePass {
 public:
  CoreMLFormatTransPass() { name_ = "CoreMLFormatTransPass"; }

  int Run(CoreMLGraph *subgraph) override;

 private:
  int InsertPreNodes(CoreMLOp *op, std::vector<CoreMLOp *> *trans_ops);

  int InsertPostNodes(CoreMLOp *op, std::vector<CoreMLOp *> *trans_ops);

 private:
  int total = 0;
  std::vector<CoreMLOp *> *all_ops_ = nullptr;
  std::vector<mindspore::MSTensor *> *all_tensors_ = nullptr;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_PASS_COREML_FORMAT_TRANS_PASS_H_
