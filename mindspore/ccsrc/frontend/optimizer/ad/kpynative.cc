/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#include <memory>
#include <string>
#include <utility>
#include "ir/anf.h"
#include "frontend/optimizer/ad/adjoint.h"
#include "frontend/optimizer/ad/dfunctor.h"
#include "frontend/optimizer/ad/kpynative.h"
#include "frontend/operator/ops.h"
#include "utils/symbolic.h"
#include "utils/primitive_utils.h"
#include "utils/ms_context.h"
#include "utils/info.h"
#include "debug/trace.h"

namespace mindspore {
namespace ad {
extern KPrim g_k_prims;

class KPynativeCellImpl : public KPynativeCell {
 public:
  explicit KPynativeCellImpl(const AnfNodePtrList &cell_inputs) : cell_inputs_(cell_inputs) {
    tape_ = std::make_shared<FuncGraph>();
    for (size_t i = 0; i < cell_inputs.size(); ++i) {
      tape_->add_parameter();
    }
  }
  ~KPynativeCellImpl() override = default;
  bool KPynativeOp(const CNodePtr &c_node, const ValuePtrList &op_args, const ValuePtr &out);
  bool KPynativeWithBProp(const CNodePtr &c_node, const ValuePtrList &op_args, const ValuePtr &out,
                          const FuncGraphPtr &bprop_fg);
  FuncGraphPtr Finish(const AnfNodePtrList &weights, bool grad_inputs, bool grad_weights);
  FuncGraphPtr bg() { return tape_; }

 private:
  FuncGraphPtr tape_;
  std::unordered_map<AnfNodePtr, AdjointPtr> anfnode_to_adjoin_;
  AnfNodePtrList cell_inputs_;
  // Last cnode of this Cell, may be a primitve op or cell with user defined bprop.
  AnfNodePtr last_node_;

  bool BackPropagate(const CNodePtr &cnode_primal, const CNodePtr &bprop_app);
  bool BuildBProp(const CNodePtr &c_node, const ValuePtrList &op_args, const ValuePtr &out,
                  const FuncGraphPtr &bprop_fg);
};
using KPynativeCellImplPtr = std::shared_ptr<KPynativeCellImpl>;

KPynativeCellPtr GradPynativeCellBegin(const AnfNodePtrList &cell_inputs) {
  return std::make_shared<KPynativeCellImpl>(cell_inputs);
}

FuncGraphPtr GetPynativeBg(const KPynativeCellPtr &k_cell) {
  auto k_cell_impl = std::dynamic_pointer_cast<KPynativeCellImpl>(k_cell);
  return k_cell_impl->bg();
}

FuncGraphPtr GradPynativeCellEnd(const KPynativeCellPtr &k_cell, const AnfNodePtrList &weights, bool grad_inputs,
                                 bool grad_weights) {
  auto k_cell_impl = std::dynamic_pointer_cast<KPynativeCellImpl>(k_cell);
  return k_cell_impl->Finish(weights, grad_inputs, grad_weights);
}

FuncGraphPtr KPynativeCellImpl::Finish(const AnfNodePtrList &weights, bool grad_inputs, bool grad_weights) {
  for (size_t i = 0; i < weights.size(); ++i) {
    tape_->add_parameter();
  }
  // sens parameter;
  auto sens_param = tape_->add_parameter();
  auto last_node_adjoint_iter = anfnode_to_adjoin_.find(last_node_);
  if (last_node_adjoint_iter == anfnode_to_adjoin_.end()) {
    MS_LOG(EXCEPTION) << "BackPropagate adjoint does not exist for input: " << last_node_->ToString();
  }
  // Set dout of last node to sens;
  last_node_adjoint_iter->second->AccumulateDout(sens_param);

  // Replace dout hole of all adjoint.
  for (auto &adjoint_iter : anfnode_to_adjoin_) {
    adjoint_iter.second->CallDoutHole();
  }

  // Return the gradient;
  AnfNodePtrList node_list{NewValueNode(prim::kPrimMakeTuple)};
  if (grad_inputs) {
    for (auto input : cell_inputs_) {
      auto input_adjoint_iter = anfnode_to_adjoin_.find(input);
      if (input_adjoint_iter == anfnode_to_adjoin_.end()) {
        MS_LOG(EXCEPTION) << "BackPropagate adjoint does not exist for input: " << input->ToString();
      }
      node_list.push_back(input_adjoint_iter->second->RealDout());
    }
  }
  if (grad_weights) {
    for (auto weight : weights) {
      auto input_adjoint_iter = anfnode_to_adjoin_.find(weight);
      if (input_adjoint_iter == anfnode_to_adjoin_.end()) {
        MS_LOG(EXCEPTION) << "BackPropagate adjoint does not exist for input: " << weight->ToString();
      }
      node_list.push_back(input_adjoint_iter->second->RealDout());
    }
  }
  auto tape_output = tape_->NewCNode(node_list);
  tape_->set_output(tape_output);
  // Replace AnfNode with parameter of tape_;
  auto mng = MakeManager({tape_}, false);
  auto tr = mng->Transact();
  const auto &parameters = tape_->parameters();
  for (size_t i = 0; i < cell_inputs_.size(); ++i) {
    tr.Replace(cell_inputs_[i], parameters[i]);
  }
  for (size_t i = 0; i < weights.size(); ++i) {
    tr.Replace(weights[i], parameters[cell_inputs_.size() + i]);
  }
  tr.Commit();

  return tape_;
}

bool GradPynativeOp(const KPynativeCellPtr &k_cell, const CNodePtr &c_node, const ValuePtrList &op_args,
                    const ValuePtr &out) {
  auto k_cell_impl = std::dynamic_pointer_cast<KPynativeCellImpl>(k_cell);
  return k_cell_impl->KPynativeOp(c_node, op_args, out);
}

bool KPynativeCellImpl::KPynativeOp(const CNodePtr &c_node, const ValuePtrList &op_args, const ValuePtr &out) {
  MS_EXCEPTION_IF_NULL(c_node);
  auto prim = GetCNodePrimitive(c_node);
  if (prim == nullptr) {
    MS_LOG(EXCEPTION) << "should be primitive, but: " << c_node->DebugString();
  }
  auto bprop_fg = g_k_prims.GetBprop(prim);
  MS_EXCEPTION_IF_NULL(bprop_fg);
  BuildBProp(c_node, op_args, out, bprop_fg);

  return true;
}

bool GradPynativeWithBProp(const KPynativeCellPtr &k_cell, const CNodePtr &c_node, const ValuePtrList &op_args,
                           const ValuePtr &out, const FuncGraphPtr &bprop_fg) {
  auto k_cell_impl = std::dynamic_pointer_cast<KPynativeCellImpl>(k_cell);
  return k_cell_impl->KPynativeWithBProp(c_node, op_args, out, bprop_fg);
}

bool KPynativeCellImpl::KPynativeWithBProp(const CNodePtr &c_node, const ValuePtrList &op_args, const ValuePtr &out,
                                           const FuncGraphPtr &bprop_fg) {
  MS_EXCEPTION_IF_NULL(c_node);
  auto primal_fg = GetCNodeFuncGraph(c_node);
  if (primal_fg == nullptr) {
    MS_LOG(EXCEPTION) << "should be func graph, but: " << c_node->DebugString();
  }
  MS_EXCEPTION_IF_NULL(bprop_fg);
  BuildBProp(c_node, op_args, out, bprop_fg);

  return true;
}

FuncGraphPtr OptimizeBPropFuncGraph(const FuncGraphPtr &bprop_fg, const CNodePtr &c_node, const ValuePtrList &op_args,
                                    const ValuePtr &out) {
  return bprop_fg;
}

bool KPynativeCellImpl::BackPropagate(const CNodePtr &cnode_primal, const CNodePtr &bprop_app) {
  for (size_t i = 1; i < cnode_primal->size(); i++) {
    auto din = tape_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), bprop_app, NewValueNode(SizeToLong(i - 1))});
    auto input = cnode_primal->input(i);
    // Backprop sens wrt inputs.
    auto input_adjoint_iter = anfnode_to_adjoin_.find(input);
    if (input_adjoint_iter == anfnode_to_adjoin_.end()) {
      MS_LOG(EXCEPTION) << "BackPropagate adjoint does not exist input[" << i << "] " << input->ToString() << ".";
    }
    input_adjoint_iter->second->AccumulateDout(din);
  }
  return true;
}

bool KPynativeCellImpl::BuildBProp(const CNodePtr &c_node, const ValuePtrList &op_args, const ValuePtr &out,
                                   const FuncGraphPtr &bprop_fg) {
  auto anfnode_adjoint_iter = anfnode_to_adjoin_.find(c_node);
  if (anfnode_adjoint_iter != anfnode_to_adjoin_.end()) {
    MS_LOG(EXCEPTION) << "CNode should be unique, but: " << c_node->DebugString();
  }
  // Book-keeping last cnode, as dout of this node will be given from outside;
  last_node_ = c_node;
  auto cnode_adjoint = std::make_shared<Adjoint>(c_node, NewValueNode(out), tape_);
  anfnode_to_adjoin_.emplace(c_node, cnode_adjoint);

  // Optimize the bprop_fg based on value.
  auto optimized_bprop_fg = OptimizeBPropFuncGraph(bprop_fg, c_node, op_args, out);
  AnfNodePtrList node_list{NewValueNode(optimized_bprop_fg)};

  for (size_t i = 1; i < c_node->inputs().size(); ++i) {
    auto inp_i = c_node->input(i);
    auto anfnode_adjoint_iter = anfnode_to_adjoin_.find(inp_i);
    if (anfnode_adjoint_iter == anfnode_to_adjoin_.end()) {
      if (inp_i->isa<CNode>()) {
        MS_LOG(EXCEPTION) << "cannot find adjoint for anfnode: " << inp_i->DebugString();
      } else {
        auto inp_i_adjoint = std::make_shared<Adjoint>(inp_i, NewValueNode(op_args[i - 1]), tape_);
        anfnode_to_adjoin_.emplace(inp_i, inp_i_adjoint);
      }
    }
    node_list.push_back(inp_i);
  }
  node_list.push_back(NewValueNode(out));
  node_list.push_back(cnode_adjoint->dout());

  auto bprop_app = tape_->NewCNode(node_list);
  cnode_adjoint->RegisterDoutUser(bprop_app, node_list.size() - 1);
  BackPropagate(c_node, bprop_app);

  return true;
}
}  // namespace ad
}  // namespace mindspore
