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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_EXPANDER_MINDIR_EMITTER_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_EXPANDER_MINDIR_EMITTER_H_
#include <memory>
#include <functional>
#include <string>
#include "backend/ms_backend/graph_fusion/expander/base/emitter.h"
#include "ir/func_graph.h"
#include "backend/ms_backend/graph_fusion/expander/mindir_adapter/anf_node_holder.h"
#include "backend/ms_backend/graph_fusion/expander/mindir_adapter/infer.h"

namespace mindspore::graphkernel::expander {
class MindirEmitter : public Emitter {
 public:
  MindirEmitter(const FuncGraphPtr &fg, bool use_device_info, const ScopePtr &scope = nullptr)
      : func_graph_(fg), use_node_device_info_(use_device_info), scope_(scope) {
    if (use_device_info) {
      infer_ = std::make_unique<InferByDeviceInfo>();
    } else {
      infer_ = std::make_unique<InferByHostInfo>();
    }
  }
  ~MindirEmitter() = default;
  NodePtrList Inputs(const CNodePtr &cnode);
  const FuncGraphPtr &func_graph() const { return func_graph_; }

  NodePtr MakeTuple(const NodePtrList &inputs) { return DefaultEmitFunc("MakeTuple", inputs, {}); }
  NodePtr EmitValue(const ValuePtr &v) override {
    auto node = NewNode(NewValueNode(v));
    infer_->SetValue(node);
    return node;
  }

 protected:
  NodePtr EmitOp(MetaOp op, const NodePtrList &args, const NodePtrDict &kargs) override;

  CNodePtr NewCNode(const AnfNodePtrList &inputs) const {
    auto cnode = func_graph_->NewCNode(inputs);
    if (scope_ != nullptr) {
      cnode->set_scope(scope_);
    }
    return cnode;
  }
  NodePtr NewNode(const AnfNodePtr &node) const {
    return use_node_device_info_ ? std::static_pointer_cast<Node>(std::make_shared<AnfNodeHolderWithDeviceInfo>(node))
                                 : std::static_pointer_cast<Node>(std::make_shared<AnfNodeHolderWithHostInfo>(node));
  }
  std::unique_ptr<MindirInfer> infer_;
  FuncGraphPtr func_graph_;
  bool use_node_device_info_;
  ScopePtr scope_;

  // emit functions
  using EmitFunc =
    std::function<NodePtr(MindirEmitter *, const std::string &, const NodePtrList &, const NodePtrDict &)>;
  // put 'args' in inputs and put 'kargs' in attrs
  NodePtr DefaultEmitFunc(const std::string &op_name, const NodePtrList &args, const NodePtrDict &kargs);
  NodePtr ReduceEmitFunc(const std::string &op_name, const NodePtrList &args, const NodePtrDict &kargs);

  inline static EmitFunc emit_functions[static_cast<int>(MetaOp::MetaOpNum)] = {
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Abs
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Add
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Assign
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::BroadcastTo
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Cast
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Floor
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Trunc
    nullptr,                          // MetaOp::Concat
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Cosh
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Div
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::RealDiv
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Equal
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Exp
    nullptr,                          // MetaOp::Gather
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Greater
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::GreaterEqual
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::IsInf
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::IsNan
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Less
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::LessEqual
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Log
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::LogicalAnd
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::LogicalNot
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::LogicalOr
    nullptr,                          // MetaOp::MatMul
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Mul
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Neg
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Reciprocal
    &MindirEmitter::ReduceEmitFunc,   // MetaOp::ReduceMax
    &MindirEmitter::ReduceEmitFunc,   // MetaOp::ReduceMin
    &MindirEmitter::ReduceEmitFunc,   // MetaOp::ReduceSum
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Reshape
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Rsqrt
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Select
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Shape
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Sinh
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Sqrt
    nullptr,                          // MetaOp::StridedSlice
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Sub
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Tanh
    nullptr,                          // MetaOp::TensorScatterAdd
    nullptr,                          // MetaOp::Transpose
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Maximum
    &MindirEmitter::DefaultEmitFunc,  // MetaOp::Minimum
  };
};
}  // namespace mindspore::graphkernel::expander
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_EXPANDER_MINDIR_EMITTER_H_
