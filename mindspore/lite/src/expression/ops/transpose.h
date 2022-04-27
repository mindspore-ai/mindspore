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

#ifndef MINDSPORE_LITE_SRC_EXPRESSION_OPS_TRANSPOSE_H_
#define MINDSPORE_LITE_SRC_EXPRESSION_OPS_TRANSPOSE_H_

#include <vector>
#include <memory>
#include "src/expression/node.h"
#include "inner/model_generated.h"

namespace mindspore {
namespace lite {
class TransposeM : public Node {
 public:
  TransposeM() = default;
  explicit TransposeM(const std::vector<int> &vector);
  static EXPR *TransposeCHW2HWC(EXPR *in) {
    std::vector<int> res = {0, 2, 3, 1};
    auto trans = new (std::nothrow) TransposeM(res);
    if (trans == nullptr) {
      return nullptr;
    }
    return (*trans)({in}).front();
  }
  static EXPR *TransposeHWC2CHW(EXPR *in) {
    std::vector<int> res = {0, 3, 1, 2};
    auto trans = new (std::nothrow) TransposeM(res);
    if (trans == nullptr) {
      return nullptr;
    }
    return (*trans)({in}).front();
  }
  int UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) override;
  std::vector<EXPR *> Grad(EXPR *yt) override;

 private:
  std::vector<int> Invert(const std::vector<int> &vec);
};

namespace NN {
Node *Transpose(const std::vector<int> &permute);
}  // namespace NN
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXPRESSION_OPS_TRANSPOSE_H_
