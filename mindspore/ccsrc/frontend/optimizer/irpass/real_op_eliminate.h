/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_REAL_OP_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_REAL_OP_ELIMINATE_H_

#include "frontend/optimizer/anf_visitor.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
namespace irpass {
class RealOpEliminate : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimRealInner, {IsNode})(node);
    auto src_type = src_->Type();
    if (src_type == nullptr || !src_type->isa<TensorType>()) {
      return src_;
    }

    src_type = src_type->cast<TensorTypePtr>()->element();
    MS_EXCEPTION_IF_NULL(src_type);
    // Real ops only makes sense when input data type is complex number.
    if (src_type->type_id() == kNumberTypeComplex64 || src_type->type_id() == kNumberTypeComplex128) {
      auto new_node = NewCNode({NewValueNode(prim::kPrimReal), src_}, node->func_graph());
      new_node->set_abstract(node->abstract());
      return new_node;
    }

    return src_;
  }
  void Visit(const AnfNodePtr &node) override {
    if (src_ == nullptr) {
      src_ = node;
    }
  }
  void Reset() { src_ = nullptr; }

 private:
  AnfNodePtr src_{nullptr};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_REAL_OP_ELIMINATE_H_
