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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_CONVERT_TENSOR_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_CONVERT_TENSOR_ELIMINATE_H_

#include "frontend/optimizer/anf_visitor.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "pipeline/jit/static_analysis/prim.h"

namespace mindspore {
namespace opt {
namespace irpass {
class ConvertTensorEliminate : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    auto fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    constexpr size_t tensor_index = 1;
    auto x = cnode->input(tensor_index);
    bool is_adapter = IsAdapterTensor(x);
    if (IsPrimitiveCNode(node, prim::kPrimConvertToAdapterTensor)) {
      // {prim::kPrimConvertToAdapterTensor, x} -> x
      if (is_adapter) {
        return x;
      }
      // {prim::kPrimConvertToAdapterTensor, {prim::kPrimConvertToMsTensor, inp}} ->
      // {prim::kPrimConvertToAdapterTensor, inp}
      if (IsPrimitiveCNode(x, prim::kPrimConvertToMsTensor)) {
        auto x_cnode = x->cast<CNodePtr>();
        auto inp = x_cnode->input(tensor_index);
        auto new_node = fg->NewCNode({NewValueNode(prim::kPrimConvertToAdapterTensor), inp});
        new_node->set_abstract(node->abstract());
        return new_node;
      }
    }
    if (IsPrimitiveCNode(x, prim::kPrimConvertToMsTensor)) {
      // {prim::kPrimConvertToMsTensor, x} -> x
      if (!is_adapter) {
        return x;
      }
      // {prim::kPrimConvertToMsTensor, {prim::kPrimConvertToAdapterTensor, inp}} ->
      // {prim::kPrimConvertToMsTensor, inp}
      if (IsPrimitiveCNode(x, prim::kPrimConvertToAdapterTensor)) {
        auto x_cnode = x->cast<CNodePtr>();
        auto inp = x_cnode->input(tensor_index);
        auto new_node = fg->NewCNode({NewValueNode(prim::kPrimConvertToMsTensor), inp});
        new_node->set_abstract(node->abstract());
        return new_node;
      }
    }
    return nullptr;
  }

 private:
  bool IsAdapterTensor(const AnfNodePtr &node) const {
    auto abs = node->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    auto abs_tensor = dyn_cast<abstract::AbstractTensor>(abs);
    MS_EXCEPTION_IF_NULL(abs_tensor);
    return abs_tensor->is_adapter();
  }
};

class ConvertTensorAllEliminate : public AnfVisitor {
 public:
  // {prim::kPrimConvertToAdapterTensor, x} -> x
  // {prim::kPrimConvertToMsTensor, x} -> x
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!IsPrimitiveCNode(node, prim::kPrimConvertToAdapterTensor) &&
        !IsPrimitiveCNode(node, prim::kPrimConvertToMsTensor)) {
      return nullptr;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    constexpr size_t tensor_index = 1;
    auto tensor = cnode->input(tensor_index);
    tensor->set_abstract(node->abstract());
    return tensor;
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_CONVERT_TENSOR_ELIMINATE_H_
