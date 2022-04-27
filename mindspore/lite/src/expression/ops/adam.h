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

#ifndef MINDSPORE_LITE_SRC_EXPRESSION_OPS_ADAM_H_
#define MINDSPORE_LITE_SRC_EXPRESSION_OPS_ADAM_H_

#include <vector>
#include <set>
#include <memory>
#include "include/api/net.h"
#include "src/expression/net.h"
#include "inner/model_generated.h"

namespace mindspore {
namespace lite {
class AdamM : public Node {
 public:
  AdamM() = default;
  AdamM(std::set<Node *> &&learn, const AdamConfig &cfg);
  std::vector<EXPR *> Clone(EXPR *grad, EXPR *weight) override;
  void AddNetOutput(std::vector<EXPR *> *output) override {
    output->push_back(assign1_);
    output->push_back(assign2_);
  }
  int UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) override;

 private:
  float weight_decay_;
  float loss_scale_;
  EXPR *assign1_{nullptr};
  EXPR *assign2_{nullptr};
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXPRESSION_OPS_ADAM_H_
