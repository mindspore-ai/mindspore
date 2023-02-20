/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_CONVERT_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_CONVERT_H_

#include <vector>
#include <memory>
#include <string>
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/anf_visitor.h"
#include "ir/func_graph.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimPrint, Xs} -> {prim::kPrimPrint, {prim::kPrinMakeTuple, Xs}}
class PrintTupleWrapper : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!IsPrimitiveCNode(node, prim::kPrimPrint)) {
      return nullptr;
    }

    // already be {prim::kPrimPrint, {prim::kPrinMakeTuple, Xs}}
    auto cnode = node->cast<CNodePtr>();
    constexpr size_t cnode_input_size = 2;
    if (cnode->size() == cnode_input_size && IsPrimitiveCNode(cnode->input(1), prim::kPrimMakeTuple)) {
      return nullptr;
    }

    std::vector<AnfNodePtr> args;
    args.push_back(NewValueNode(prim::kPrimMakeTuple));

    // {prim::kPrimPrint, Xs}
    auto &inputs = cnode->inputs();
    (void)args.insert(args.cend(), inputs.cbegin() + 1, inputs.cend());

    // {prim::kPrinMakeTuple, Xs}
    auto fg = node->func_graph();
    auto tuple = NewCNode(args, fg);
    auto print = GetValueNode<PrimitivePtr>(cnode->input(0));
    return NewCNode({NewValueNode(print), tuple}, fg);
  }
};

class PrintConstStringWrapper : public AnfVisitor {
  bool CheckNeedConvert(const AbstractBasePtr &abs) const {
    if (abs == nullptr) {
      return false;
    }
    if (abs->isa<abstract::AbstractSequence>()) {
      auto sequence_abs = abs->cast<abstract::AbstractSequencePtr>();
      const auto &elements = sequence_abs->elements();
      return std::any_of(elements.cbegin(), elements.cend(),
                         [&](const AbstractBasePtr &ele) { return CheckNeedConvert(ele); });
    }
    return !abs->isa<abstract::AbstractScalar>() && !abs->isa<abstract::AbstractTensor>();
  }

  AnfNodePtr ConvertString(const AbstractBasePtr &abs) const {
    const auto &value = abs->GetValueTrack();
    string str_content = (value == nullptr) ? "None" : value->ToString();
    auto value_node = NewValueNode(str_content);
    value_node->set_abstract(std::make_shared<abstract::AbstractScalar>(str_content));
    return value_node;
  }

  AnfNodePtr ConvertInput(const AnfNodePtr &input, const FuncGraphPtr &fg) {
    auto abs = input->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    if (!CheckNeedConvert(abs)) {
      return input;
    }
    if (abs->isa<abstract::AbstractSequence>()) {
      auto sequence_abs = abs->cast<abstract::AbstractSequencePtr>();
      const auto &elements = sequence_abs->elements();
      std::vector<AnfNodePtr> new_squence_inputs{NewValueNode(prim::kPrimMakeTuple)};
      AbstractBasePtrList new_seq_abstracts;
      for (size_t index = 0; index < elements.size(); ++index) {
        const auto &element = elements[index];
        std::vector<AnfNodePtr> new_item_inputs{NewValueNode(prim::kPrimTupleGetItem), input,
                                                NewValueNode(static_cast<int64_t>(index))};
        auto item = fg->NewCNode(new_item_inputs);
        item->set_abstract(element);
        auto new_item = ConvertInput(item, fg);
        new_squence_inputs.push_back(new_item);
        new_seq_abstracts.push_back(new_item->abstract());
      }
      auto new_sequence = fg->NewCNode(new_squence_inputs);
      new_sequence->set_abstract(std::make_shared<abstract::AbstractTuple>(new_seq_abstracts));
      return new_sequence;
    }
    return ConvertString(abs);
  }

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!IsPrimitiveCNode(node, prim::kPrimPrint)) {
      return nullptr;
    }
    auto cnode = node->cast<CNodePtr>();
    auto &inputs = cnode->inputs();
    auto fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    // Check if need convert.
    bool print_input_need_convert = false;
    // print(xxx, IO)
    for (size_t index = 1; index < inputs.size() - 1; ++index) {
      const auto &input = inputs[index];
      MS_EXCEPTION_IF_NULL(input);
      if (CheckNeedConvert(input->abstract())) {
        print_input_need_convert = true;
        break;
      }
    }
    if (!print_input_need_convert) {
      return nullptr;
    }
    std::vector<AnfNodePtr> args{cnode->input(0)};
    for (size_t index = 1; index < inputs.size() - 1; ++index) {
      const auto &input = inputs[index];
      auto arg = ConvertInput(input, fg);
      args.push_back(arg);
    }
    auto io_monad = inputs.back();
    args.push_back(io_monad);
    auto ret = fg->NewCNode(args);
    return ret;
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // #ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_CONVERT_H_
