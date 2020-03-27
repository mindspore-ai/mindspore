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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_IRPASS_CAST_ELIMINATE_H_
#define MINDSPORE_CCSRC_OPTIMIZER_IRPASS_CAST_ELIMINATE_H_

#include "optimizer/irpass.h"
#include "optimizer/optimizer.h"
#include "ir/visitor.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimCast, X, T}
class CastSameTypeEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
  void Visit(const AnfNodePtr &node) override;
  void Reset() {
    src_ = nullptr;
    tgt_ = nullptr;
  }

 private:
  AnfNodePtr src_{nullptr}, tgt_{nullptr};
};

// {prim::kPrimCast, {prim::kPrimCast, X, Y}, T}
class TwoCastEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
  void Visit(const AnfNodePtr &node) override;
  void Reset() {
    x_ = nullptr;
    t_ = nullptr;
  }

 private:
  AnfNodePtr x_{nullptr}, t_{nullptr};
};

class CastEliminater {
 public:
  CastEliminater() : cast_same_type_eliminater_(), two_cast_eliminater_() {}
  ~CastEliminater() = default;

  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) {
    auto new_node = cast_same_type_eliminater_(optimizer, node);
    if (new_node != nullptr) {
      return new_node;
    }

    new_node = two_cast_eliminater_(optimizer, node);
    if (new_node != nullptr) {
      return new_node;
    }

    return nullptr;
  }

 private:
  CastSameTypeEliminater cast_same_type_eliminater_;
  TwoCastEliminater two_cast_eliminater_;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_OPTIMIZER_IRPASS_CAST_ELIMINATE_H_
