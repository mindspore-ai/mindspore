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

#include "backend/common/graph_kernel/expander/mindir_adapter/infer.h"
#include <algorithm>
#include <memory>
#include "backend/common/graph_kernel/model/op_register.h"
#include "backend/common/graph_kernel/model/node.h"
#include "backend/common/graph_kernel/core/graph_kernel_callback.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore::graphkernel::expander {
void InferByDeviceInfo::InferOp(const NodePtr &node, const PrimitivePtr &prim, const NodePtrList &args) {
  auto inner_op = inner::OpRegistry::Instance().NewOp(prim->name());
  inner::NodePtrList inputs(args.size());
  (void)std::transform(args.cbegin(), args.cend(), inputs.begin(), [this](const NodePtr &no) {
    auto &inner_node = this->inner_node_cache_[no];
    MS_EXCEPTION_IF_NULL(inner_node);
    return inner_node;
  });
  inner_node_cache_[node] = inner_op;
  auto outs = inner_op->Infer(inputs, prim->attrs());
  inner_op->SetAttrs(prim->attrs());
  inner_op->SetBaseInfo(outs);
  auto anfnode = node->as<AnfNodePtr>();
  if (outs.size() == 1) {
    anfnode->set_abstract(std::make_shared<abstract::AbstractTensor>(TypeIdToType(outs[0].type), outs[0].shape));
  } else {
    AbstractBasePtrList abs_list(outs.size());
    (void)std::transform(outs.cbegin(), outs.cend(), abs_list.begin(), [](const inner::NodeBase &out_info) {
      return std::make_shared<abstract::AbstractTensor>(TypeIdToType(out_info.type), out_info.shape);
    });
    anfnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abs_list));
  }
  Callback::Instance()->SetBasicNodeKernelInfo(anfnode, outs);
}

void InferByDeviceInfo::SetValue(const NodePtr &node) {
  auto v = node->GetValue();
  node->as<AnfNodePtr>()->set_abstract(v->ToAbstract());
  if (v->isa<tensor::Tensor>()) {
    auto tensor = v->cast<tensor::TensorPtr>();
    inner_node_cache_[node] = std::make_shared<inner::ConstTensorNode>(tensor);
    Callback::Instance()->SetBasicNodeKernelInfo(node->as<AnfNodePtr>(),
                                                 {{tensor->shape(), tensor->data_type(), kOpFormat_DEFAULT}});
  }
}

void InferByDeviceInfo::HandleInputs(const NodePtrList &inputs) {
  auto cb = Callback::Instance();
  for (auto &inp : inputs) {
    auto anfnode = inp->as<AnfNodePtr>();
    auto value = anfnode->abstract()->BuildValue();
    if (value->isa<tensor::Tensor>()) {
      auto tensor = value->cast<tensor::TensorPtr>();
      auto &t = inner_node_cache_[inp] = std::make_shared<inner::ConstTensorNode>(tensor);
      t->shape = tensor->shape();
      t->format = cb->GetOutputFormat(anfnode, 0);
    } else {
      inner::NodeBase node_base;
      node_base.shape = cb->GetOutputShape(anfnode, 0);
      node_base.type = cb->GetOutputType(anfnode, 0);
      node_base.format = cb->GetOutputFormat(anfnode, 0);
      inner_node_cache_[inp] = std::make_shared<inner::ParamNode>(node_base);
    }
  }
}
void InferByHostInfo::InferOp(const NodePtr &node, const PrimitivePtr &prim, const NodePtrList &args) {
  // refer to CppInfer::InferAnfnode
  auto anfnode = node->as<AnfNodePtr>();
  if (anfnode->isa<ValueNode>()) {
    anfnode->set_abstract(anfnode->cast<ValueNodePtr>()->value()->ToAbstract());
    return;
  }
  auto cnode = anfnode->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  AbstractBasePtrList abs_list;
  abs_list.reserve(cnode->size());
  (void)std::transform(args.cbegin(), args.cend(), std::back_inserter(abs_list), [](const NodePtr &node) {
    auto anfnode = node->as<AnfNodePtr>();
    const auto &abs = anfnode->abstract();
    if (abs == nullptr) {
      MS_EXCEPTION_IF_CHECK_FAIL(anfnode->isa<ValueNode>(), anfnode->ToString() + " has no abstract");
      return anfnode->cast<ValueNodePtr>()->value()->ToAbstract();
    }
    return abs;
  });
  auto found = abstract::GetPrimitiveInferImpl(prim);
  if (found.has_value() && found.value().IsImplInferShapeAndType()) {
    cnode->set_abstract(found.value().InferShapeAndType(nullptr, prim, abs_list));
  } else {
    MS_LOG(EXCEPTION) << "The infer function of [" << prim->name() << "] is not defined.";
  }
}
}  // namespace mindspore::graphkernel::expander
