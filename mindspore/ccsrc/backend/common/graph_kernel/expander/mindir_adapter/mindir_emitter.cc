/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/expander/mindir_adapter/mindir_emitter.h"
#include <algorithm>
#include "ir/primitive.h"
#include "backend/common/graph_kernel/expander/mindir_adapter/anf_node_holder.h"
#include "backend/common/graph_kernel/model/op_register.h"

namespace mindspore::graphkernel::expander {
NodePtr MindirEmitter::EmitOp(MetaOp op, const NodePtrList &args, const NodePtrDict &kargs) {
  auto func = emit_functions[static_cast<int>(op)];
  if (func == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "The emit function of op " << MetaOpStr[static_cast<int>(op)] << "("
                               << static_cast<int>(op) << ") is nullptr.";
  }
  return func(this, MetaOpStr[static_cast<int>(op)], args, kargs);
}

NodePtrList MindirEmitter::Inputs(const CNodePtr &cnode) {
  NodePtrList result(cnode->size() - 1);
  auto &inputs = cnode->inputs();
  (void)std::transform(inputs.cbegin() + 1, inputs.cend(), result.begin(), [this](const AnfNodePtr &node) {
    auto p = func_graph_->add_parameter();
    p->set_abstract(node->abstract());
    p->set_kernel_info(node->kernel_info_ptr());
    return NewNode(p);
  });
  infer_->HandleInputs(result);
  return result;
}

NodePtr MindirEmitter::DefaultEmitFunc(const std::string &op_name, const NodePtrList &args, const NodePtrDict &kargs) {
  auto prim = std::make_shared<Primitive>(op_name);
  if (!kargs.empty()) {
    HashMap<std::string, ValuePtr> attrs;
    for (auto &[k, v] : kargs) {
      attrs[k] = v->GetValue();
    }
    prim->SetAttrs(attrs);
  }
  AnfNodePtrList inputs(args.size() + 1);
  inputs[0] = NewValueNode(prim);
  (void)std::transform(args.cbegin(), args.cend(), inputs.begin() + 1,
                       [](const NodePtr &node) { return node->as<AnfNodePtr>(); });
  auto node = NewNode(NewCNode(inputs));
  infer_->InferOp(node, prim, args);
  return node;
}
}  // namespace mindspore::graphkernel::expander
