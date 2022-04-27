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

#ifndef MINDSPORE_LITE_SRC_EXPRESSION_OPS_CONV_H_
#define MINDSPORE_LITE_SRC_EXPRESSION_OPS_CONV_H_

#include <vector>
#include <memory>
#include <string>
#include "src/expression/cfg.h"
#include "src/expression/node.h"

namespace mindspore {
namespace lite {
class ConvM : public Node {
 public:
  ConvM() = default;
  explicit ConvM(const ConvConfig &cfg);
  std::vector<EXPR *> construct(const std::vector<EXPR *> &inputs) override;
  Param *weight() override { return input(1)->node()->data(); }
  std::vector<EXPR *> Grad(EXPR *expr) override;
  int UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) override;
  void SetLearn() override;

 private:
  int GetMode(std::string mode);
  Node *bias_ = nullptr;
  EXPR *wbias_ = nullptr;
};

class ConvInputGradM : public Node {
 public:
  explicit ConvInputGradM(ConvM *conv_node);
  int UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) override;
};

class ConvFilterGradM : public Node {
 public:
  explicit ConvFilterGradM(ConvM *conv_node);
  int UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) override;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXPRESSION_OPS_CONV_H_
