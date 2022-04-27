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

#ifndef MINDSPORE_LITE_SRC_EXPRESSION_OPS_SOFTMAXCE_H_
#define MINDSPORE_LITE_SRC_EXPRESSION_OPS_SOFTMAXCE_H_

#include <vector>
#include <memory>
#include <string>
#include "src/expression/node.h"
#include "inner/model_generated.h"
#include "include/api/net.h"

namespace mindspore {
namespace lite {
class SoftmaxCrossEntropyM : public Node {
 public:
  SoftmaxCrossEntropyM();
  explicit SoftmaxCrossEntropyM(const SoftMaxCrossEntropyCfg &cfg);
  std::vector<EXPR *> Grad(EXPR *expr) override;
  int UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) override;
  std::vector<EXPR *> construct(const std::vector<EXPR *> &inputs) override;

 private:
  Node *GetReductionNode(const std::string &mode, const std::vector<int> &axis);
  Node *reduce_ = nullptr;
};

namespace NN {
Node *SoftmaxCrossEntropy(const SoftMaxCrossEntropyCfg &cfg);
}  // namespace NN
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXPRESSION_OPS_SOFTMAXCE_H_
