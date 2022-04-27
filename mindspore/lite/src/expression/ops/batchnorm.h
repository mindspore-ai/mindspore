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

#ifndef MINDSPORE_LITE_SRC_EXPRESSION_OPS_BATCHNORM_H_
#define MINDSPORE_LITE_SRC_EXPRESSION_OPS_BATCHNORM_H_

#include <vector>
#include <memory>
#include "src/expression/net.h"
#include "inner/model_generated.h"

namespace mindspore {
namespace lite {
class BatchNorm2dM : public Node {
 public:
  BatchNorm2dM() = default;
  BatchNorm2dM(int outp, float momentum, float epsilon);
  std::vector<EXPR *> Grad(EXPR *expr) override;
  int UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) override;
  void SetLearn() override;
};

class BatchNorm2dGradM : public Node {
 public:
  explicit BatchNorm2dGradM(BatchNorm2dM *bn_node);
  int UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) override;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXPRESSION_OPS_BATCHNORM_H_
