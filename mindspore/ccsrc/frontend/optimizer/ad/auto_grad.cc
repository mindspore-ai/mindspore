/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#include "frontend/optimizer/ad/auto_grad.h"
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include "mindspore/core/ops/core_ops.h"
#include "ir/anf.h"
#include "frontend/optimizer/ad/adjoint.h"
#include "utils/info.h"
#include "pipeline/jit/debug/trace.h"
#include "pipeline/pynative/grad/bprop_expander/bprop.h"
#include "pipeline/pynative/pynative_utils.h"
#include "utils/profile.h"
#include "include/common/utils/primitive_utils.h"
#include "pipeline/jit/pass.h"
namespace mindspore {
namespace ad {
namespace {
constexpr char kAttrZerosLikeCSR[] = "zero_like_csr_node";
constexpr char kAttrZerosLikeCOO[] = "zero_like_coo_node";
constexpr char kAttrOnesLikeCSR[] = "ones_like_csr_node";
constexpr char kAttrOnesLikeCOO[] = "ones_like_coo_node";
enum class SpecialType { kZerosLikeType = 0, kOnesLikeType = 1 };
const std::map<SpecialType, std::shared_ptr<Primitive>> kValueType{{SpecialType::kZerosLikeType, prim::kPrimZerosLike},
                                                                   {SpecialType::kOnesLikeType, prim::kPrimOnesLike}};
const std::vector<PrimitivePtr> kGradBlackList{prim::kPrimMakeTuple, prim::kPrimTupleGetItem, prim::kPrimStopGradient,
                                               prim::kPrimUpdateState, prim::kPrimNPUAllocFloatStatus};
AnfNodePtr BuildSpecialLikeValue(const FuncGraphPtr &tape, const ValuePtr &value, const SpecialType &type);
void ClearDeviceAddress(const ValuePtr &value) {
  std::vector<tensor::TensorPtr> tensors;
  TensorValueToTensor(value, &tensors);
  for (auto tensor : tensors) {
    tensor->set_device_address(nullptr);
    tensor->set_is_forward_output(false);
  }
}

ValuePtr FilterSensValues(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>() || value->isa<tensor::COOTensor>() || value->isa<tensor::CSRTensor>()) {
    return value;
  } else if (value->isa<ValueSequence>()) {
    std::vector<ValuePtr> value_list;
    auto value_seq = value->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(value_seq);
    for (auto filter_value : value_seq->value()) {
      if (FilterSensValues(filter_value) != nullptr) {
        (void)value_list.emplace_back(filter_value);
      }
    }
    return std::make_shared<ValueTuple>(value_list);
  } else {
    MS_LOG(DEBUG) << "value type: " << value->ToString();
    return nullptr;
  }
}

bool IsPrimNeedGrad(const PrimitivePtr &prim) {
  for (const auto &no_need_grad_prim : kGradBlackList) {
    if (IsPrimitiveEquals(prim, no_need_grad_prim)) {
      return false;
    }
  }
  return true;
}

AnfNodePtr BuildSpecialLikeCSRTensor(const FuncGraphPtr &tape, const ValuePtr &value, const SpecialType &type) {
  MS_EXCEPTION_IF_NULL(value);

  auto csr_tensor = value->cast<tensor::CSRTensorPtr>();
  MS_EXCEPTION_IF_NULL(csr_tensor);
  auto indptr = csr_tensor->GetIndptr();
  auto cloned_indptr = ShallowCopyTensorValue(indptr);
  ClearDeviceAddress(cloned_indptr);

  auto indptr_node = NewValueNode(cloned_indptr);
  indptr_node->set_abstract(cloned_indptr->ToAbstract()->Broaden());
  auto indices = csr_tensor->GetIndices();
  auto cloned_indices = ShallowCopyTensorValue(indices);
  ClearDeviceAddress(cloned_indices);
  auto indices_node = NewValueNode(cloned_indices);
  indices_node->set_abstract(cloned_indices->ToAbstract()->Broaden());

  auto data = csr_tensor->GetValues();
  auto cloned_data = ShallowCopyTensorValue(data);
  ClearDeviceAddress(cloned_data);
  auto value_node = NewValueNode(cloned_data);
  value_node->set_abstract(cloned_data->ToAbstract()->Broaden());

  auto zero_like_value = BuildSpecialLikeValue(tape, cloned_data, type);
  auto shape = csr_tensor->shape();
  auto value_shape = NewValueNode(shape);
  std::vector<abstract::AbstractBasePtr> abstract_shape;
  (void)std::transform(
    shape.begin(), shape.end(), std::back_inserter(abstract_shape),
    [](auto shp) -> abstract::AbstractScalarPtr { return std::make_shared<abstract::AbstractScalar>(shp); });
  auto abs_shape = std::make_shared<abstract::AbstractTuple>(abstract_shape);
  value_shape->set_abstract(abs_shape);
  auto special_like_csr_node =
    tape->NewCNode({NewValueNode(prim::kPrimMakeTuple), indptr_node, indices_node, zero_like_value, value_shape});
  special_like_csr_node->set_abstract(value->ToAbstract()->Broaden());
  special_like_csr_node->AddAttr(kAttrZerosLikeCSR, MakeValue(True));
  if (type == SpecialType::kZerosLikeType) {
    special_like_csr_node->AddAttr(kAttrZerosLikeCSR, MakeValue(True));
  } else {
    special_like_csr_node->AddAttr(kAttrOnesLikeCSR, MakeValue(True));
  }
  return special_like_csr_node;
}

AnfNodePtr BuildSpecialLikeCOOTensor(const FuncGraphPtr &tape, const ValuePtr &value, const SpecialType &type) {
  MS_EXCEPTION_IF_NULL(value);

  auto coo_tensor = value->cast<tensor::COOTensorPtr>();
  MS_EXCEPTION_IF_NULL(coo_tensor);
  auto indices = coo_tensor->GetIndices();
  auto cloned_indices = ShallowCopyTensorValue(indices);
  ClearDeviceAddress(cloned_indices);

  auto indices_node = NewValueNode(cloned_indices);
  indices_node->set_abstract(cloned_indices->ToAbstract()->Broaden());
  auto data = coo_tensor->GetValues();
  auto cloned_data = ShallowCopyTensorValue(data);
  ClearDeviceAddress(cloned_data);
  auto value_node = NewValueNode(cloned_data);
  value_node->set_abstract(cloned_data->ToAbstract()->Broaden());

  auto special_like_value = BuildSpecialLikeValue(tape, cloned_data, type);
  auto shape = coo_tensor->shape();
  auto value_shape = NewValueNode(shape);
  std::vector<abstract::AbstractBasePtr> abstract_shape;
  (void)std::transform(
    shape.begin(), shape.end(), std::back_inserter(abstract_shape),
    [](auto shp) -> abstract::AbstractScalarPtr { return std::make_shared<abstract::AbstractScalar>(shp); });
  auto abs_shape = std::make_shared<abstract::AbstractTuple>(abstract_shape);
  value_shape->set_abstract(abs_shape);
  auto special_like_coo_node =
    tape->NewCNode({NewValueNode(prim::kPrimMakeTuple), indices_node, special_like_value, value_shape});
  special_like_coo_node->set_abstract(value->ToAbstract()->Broaden());
  if (type == SpecialType::kZerosLikeType) {
    special_like_coo_node->AddAttr(kAttrZerosLikeCOO, MakeValue(True));
  } else {
    special_like_coo_node->AddAttr(kAttrOnesLikeCOO, MakeValue(True));
  }
  return special_like_coo_node;
}

AnfNodePtr BuildSpecialLikeValue(const FuncGraphPtr &tape, const ValuePtr &value, const SpecialType &type) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>() || value->isa<Scalar>()) {
    auto vlaue_node = NewValueNode(value);
    vlaue_node->set_abstract(value->ToAbstract()->Broaden());
    auto primitive = kValueType.at(type);
    MS_EXCEPTION_IF_NULL(primitive);
    auto special_like_value = tape->NewCNode({NewValueNode(primitive), vlaue_node});
    special_like_value->set_abstract(value->ToAbstract()->Broaden());
    return special_like_value;
  } else if (value->isa<tensor::CSRTensor>()) {
    return BuildSpecialLikeCSRTensor(tape, value, type);
  } else if (value->isa<tensor::COOTensor>()) {
    return BuildSpecialLikeCOOTensor(tape, value, type);
  } else if (value->isa<ValueSequence>()) {
    std::vector<AnfNodePtr> args;
    (void)args.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    auto tuple = value->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(tuple);
    for (size_t i = 0; i < tuple->size(); ++i) {
      const auto &v = tuple->value()[i];
      AnfNodePtr special_like_value = BuildSpecialLikeValue(tape, v, type);
      (void)args.emplace_back(special_like_value);
    }
    auto special_like_value = tape->NewCNode(args);
    special_like_value->set_abstract(value->ToAbstract()->Broaden());
    return special_like_value;
  } else {
    MS_EXCEPTION(TypeError) << "For value" << value->type()->ToString() << "`, the type is not tensor or sequence";
  }
}

AnfNodePtr BuildZerosLikeNode(const FuncGraphPtr &tape, const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  auto cloned_value = ShallowCopyTensorValue(value);
  ClearDeviceAddress(cloned_value);
  return BuildSpecialLikeValue(tape, cloned_value, SpecialType::kZerosLikeType);
}

AnfNodePtr BuildOnesLikeNode(const FuncGraphPtr &tape, const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  auto cloned_value = ShallowCopyTensorValue(value);
  ClearDeviceAddress(cloned_value);
  return BuildSpecialLikeValue(tape, cloned_value, SpecialType::kOnesLikeType);
}

bool IsZerosLikeNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  if (IsPrimitiveCNode(cnode, prim::kPrimZerosLike)) {
    return true;
  } else if (IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) || IsPrimitiveCNode(cnode, prim::kPrimMakeList)) {
    if (cnode->HasAttr(kAttrZerosLikeCSR) || cnode->HasAttr(kAttrZerosLikeCOO)) {
      return true;
    }
    for (size_t i = 1; i < cnode->size(); ++i) {
      if (!IsZerosLikeNode(cnode->input(i))) {
        return false;
      }
    }
    return true;
  }
  return false;
}

FuncGraphPtr OptimizeBpropBuilder(const FuncGraphPtr &bprop_func_graph) {
  pipeline::ResourcePtr resource = std::make_shared<pipeline::Resource>();
  resource->set_func_graph(bprop_func_graph);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(bprop_func_graph);
  auto after_opt_bg = pipeline::OptGradGraphPass(resource);
  pynative::PyNativeAlgo::Common::DumpGraphIR("bprop_builder_after_opt.ir", after_opt_bg);
  return after_opt_bg;
}
}  // namespace

AnfNodePtr FunctionNode::HyperAdd(const AnfNodePtr &left_node, const AnfNodePtr &right_node) {
  MS_EXCEPTION_IF_NULL(left_node);
  MS_EXCEPTION_IF_NULL(right_node);

  if (IsZerosLikeNode(right_node)) {
    return left_node;
  }
  if (IsZerosLikeNode(left_node)) {
    return right_node;
  }
  if (!IsPrimitiveCNode(left_node, prim::kPrimMakeTuple)) {
    auto add_result = tape_->NewCNode({NewValueNode(prim::kPrimAdd), left_node, right_node});
    add_result->set_abstract(right_node->abstract()->Broaden());
    return add_result;
  } else if (IsPrimitiveCNode(left_node, prim::kPrimMakeTuple) && IsPrimitiveCNode(right_node, prim::kPrimMakeTuple)) {
    auto left_cnode = left_node->cast<CNodePtr>();
    auto right_cnode = right_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(right_cnode);
    std::vector<AnfNodePtr> inputs = {NewValueNode(prim::kPrimMakeTuple)};
    AbstractBasePtrList abs;
    for (size_t i = 1; i < left_cnode->size(); ++i) {
      auto add_result = HyperAdd(left_cnode->input(i), right_cnode->input(i));
      (void)abs.emplace_back(add_result->abstract());
      (void)inputs.emplace_back(add_result);
    }
    auto add_tuple = tape_->NewCNode(inputs);
    add_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(abs));
    return add_tuple;
  } else {
    MS_LOG(EXCEPTION) << "unknown cnode type" << left_node->DebugString();
  }
}

void FunctionNode::AddEdge(const AnfNodePtr &next_node, const AnfNodePtr &din) {
  MS_EXCEPTION_IF_NULL(next_node);
  MS_EXCEPTION_IF_NULL(din);
  (void)next_edges_.emplace_back(std::make_pair(next_node, din));
  if (din == fake_dout_) {
    (void)need_replace_edges_.emplace_back(next_edges_.size() - 1);
  }
}

void FunctionNode::UpdateAccumulativeDout(const AnfNodePtr &new_dout) {
  MS_EXCEPTION_IF_NULL(new_dout);
  accumulate_dout_ = HyperAdd(accumulate_dout_, new_dout);
}

void FunctionNode::ReplaceEdges() {
  MS_EXCEPTION_IF_NULL(accumulate_dout_);
  for (const auto index : need_replace_edges_) {
    next_edges_[index].second = accumulate_dout_;
  }
}

AutoGradCellImpl::AutoGradCellImpl(const AnfNodePtrList &cell_inputs, const std::vector<ValuePtr> &input_param_values)
    : tape_(std::make_shared<FuncGraph>()), cell_inputs_(cell_inputs) {
  tape_->debug_info()->set_name("grad_top");
  MS_LOG(DEBUG) << "Start AutoGradCellImpl: "
                << "cell_inputs size: " << cell_inputs.size();
  for (size_t i = 0; i < cell_inputs.size(); ++i) {
    TraceGuard trace_guard(std::make_shared<TraceCopy>(cell_inputs[i]->debug_info()));
    auto parameter = tape_->add_parameter();
    parameter->set_abstract(input_param_values[i]->ToAbstract()->Broaden());
    auto zeros_like_dout = BuildZerosLikeNode(tape_, input_param_values[i]);
    auto func_node = std::make_shared<FunctionNode>(tape_, zeros_like_dout);
    auto input_adjoint = std::make_shared<VariableNode>(func_node, input_param_values[i]);
    anfnode_to_variable_adjoint_.insert(std::make_pair(cell_inputs[i], input_adjoint));
  }
}

bool AutoGradCellImpl::KPynativeOp(const GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(grad_param);

  MS_LOG(DEBUG) << "Forward cnode: " << grad_param->cnode->DebugString();
  auto prim = GetCNodePrimitive(grad_param->cnode);
  if (prim == nullptr) {
    MS_LOG(EXCEPTION) << "Should be primitive, but: " << grad_param->cnode->DebugString();
  }
  if (!IsPrimNeedGrad(prim)) {
    MS_LOG(DEBUG) << "Prim " << prim->name() << " not need do op grad";
    return true;
  }
  // anfnode_to_variable_adjoint_ hold out value, to avoid device not release, clear its device_address
  auto cloned_value = ShallowCopyTensorValue(grad_param->out);
  ClearDeviceAddress(cloned_value);
  AnfNodePtr dout = BuildSpecialLikeValue(tape_, cloned_value, SpecialType::kZerosLikeType);
  auto fn = std::make_shared<FunctionNode>(tape_, dout);
  auto variable_adjoint = std::make_shared<VariableNode>(fn, cloned_value);
  if (!grad_param->grad_by_value) {
    BuildKNode(grad_param, variable_adjoint);
    need_do_manager_replace_ = true;
  }
  CNodePtr input_node = ConstructBpropGraphInput(grad_param, dout);
  MS_LOG(DEBUG) << "Construct input cnode: " << input_node->DebugString();
  std::vector<CNodePtr> outputs;
#ifndef ENABLE_TEST
  if (IsPrimitiveEquals(prim, prim::kPrimHookBackward) || IsPrimitiveEquals(prim, prim::kPrimCellBackwardHook)) {
    BuildBPropCutCNode(input_node, &outputs);
  } else {
    mindspore::BuildBprop(input_node, &outputs, &users_);
    if (outputs.empty()) {
      MS_LOG(DEBUG) << "expander has no bprop of this prim: " << grad_param->cnode->DebugString();
      BuildCustomBpropCNode(input_node, &outputs);
    }
  }
#else
  if (IsPrimitiveEquals(prim, prim::kPrimHookBackward) || IsPrimitiveEquals(prim, prim::kPrimCellBackwardHook)) {
    BuildBPropCutCNode(input_node, &outputs);
  } else {
    BuildCustomBpropCNode(input_node, &outputs);
  }
#endif
  if (!outputs.empty()) {
    UpdateNextEdges(fn, grad_param->cnode, outputs, grad_param->op_args);
  } else {
    MS_LOG(DEBUG) << "this op has not custom bprop: " << grad_param->cnode->DebugString();
    variable_adjoint->set_is_fake_bprop(true);
    variable_adjoint->set_fake_prim_name(prim->name());
  }
  anfnode_to_variable_adjoint_.insert(std::make_pair(grad_param->cnode, variable_adjoint));
  // record last_node for brackpropagate
  last_node_ = grad_param->cnode;
  return true;
}

bool AutoGradCellImpl::KPynativeWithFProp(const GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(grad_param);
  AnfNodePtrList args_node_list;
  CNodePtr bprop_cnode = nullptr;
  AnfNodePtr k_node = nullptr;
  AnfNodePtr dout = nullptr;
  if (grad_param->grad_by_value) {
    for (size_t i = 0; i < grad_param->op_args.size(); ++i) {
      auto input_node = grad_param->cnode->input(i + 1);
      if (input_node->isa<Parameter>()) {
        if (input_node->abstract() == nullptr) {
          input_node->set_abstract(grad_param->op_args[i]->ToAbstract()->Broaden());
        }
        (void)args_node_list.emplace_back(input_node);
        continue;
      }
      auto v_node = NewValueNode(grad_param->op_args[i]);
      v_node->set_abstract(grad_param->op_args[i]->ToAbstract()->Broaden());
      (void)args_node_list.emplace_back(v_node);
    }
    bprop_cnode = GetBPropFromFProp(grad_param->fprop_fg, args_node_list, grad_param->out, &dout);
  } else {
    BuildKNodeListFromPrimalCNode(grad_param->cnode, grad_param->op_args, &args_node_list);
    bprop_cnode = GetBPropFromFProp(grad_param->fprop_fg, args_node_list, grad_param->out, &dout);
  }

  std::vector<CNodePtr> outputs;
  for (size_t i = 1; i < grad_param->cnode->size(); ++i) {
    // bprop_app[0] env
    CNodePtr din = tape_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), bprop_cnode, NewValueNode(SizeToLong(i))});
    din->set_abstract(grad_param->op_args[i - 1]->ToAbstract()->Broaden());
    (void)outputs.emplace_back(din);
  }
  auto fn = std::make_shared<FunctionNode>(tape_, dout);
  auto variable_adjoint = std::make_shared<VariableNode>(fn, grad_param->out);
  variable_adjoint->set_k_node(k_node);
  UpdateNextEdges(fn, grad_param->cnode, outputs, grad_param->op_args);
  anfnode_to_variable_adjoint_.insert(std::make_pair(grad_param->cnode, variable_adjoint));
  need_do_manager_replace_ = true;
  return true;
}

CNodePtr AutoGradCellImpl::GetBPropFromFProp(const FuncGraphPtr &fprop_fg, const AnfNodePtrList &args,
                                             const ValuePtr &out, AnfNodePtr *const tape_dout) {
  // Wrap tuple_getitem(fprop_app, 1) in a FuncGraph and optimize it;
  auto bprop_builder = std::make_shared<FuncGraph>();
  bprop_builder->debug_info()->set_name("bprop_builder");

  AnfNodePtrList fprop_app_inputs{NewValueNode(fprop_fg)};
  AnfNodePtrList bprop_builder_inputs;
  for (const auto &arg : args) {
    auto param = bprop_builder->add_parameter();
    param->set_abstract(arg->abstract());
    (void)fprop_app_inputs.emplace_back(param);
    (void)bprop_builder_inputs.emplace_back(arg);
  }
  auto fprop_app = bprop_builder->NewCNode(fprop_app_inputs);
  auto get_bprop =
    bprop_builder->NewCNode({NewValueNode(prim::kPrimTupleGetItem), fprop_app, NewValueNode(static_cast<int64_t>(1))});

  // Get graph after optimize
  AnfNodePtrList node_list{get_bprop};
  auto dout = bprop_builder->add_parameter();
  MS_EXCEPTION_IF_NULL(out);
  dout->set_abstract(out->ToAbstract()->Broaden());
  (void)node_list.emplace_back(dout);
  auto call_bprop = bprop_builder->NewCNode(node_list);
  bprop_builder->set_output(call_bprop);
  auto after_opt_fg = OptimizeBpropBuilder(bprop_builder);
  // Call by tape_
  MS_EXCEPTION_IF_NULL(tape_dout);
  *tape_dout = BuildZerosLikeNode(tape_, out);
  (void)bprop_builder_inputs.emplace_back(*tape_dout);
  bprop_builder_inputs.insert(bprop_builder_inputs.cbegin(), NewValueNode(after_opt_fg));
  get_bprop = tape_->NewCNode(bprop_builder_inputs);
  // tape_dout is set by next op
  AddUser(*tape_dout, get_bprop, bprop_builder_inputs.size() - 1);
  return get_bprop;
}

void AutoGradCellImpl::UpdateOutputNodeOfTopCell(const AnfNodePtr &output_node, const ValuePtr &sens_out) {
  MS_EXCEPTION_IF_NULL(output_node);
  MS_EXCEPTION_IF_NULL(sens_out);
  MS_LOG(DEBUG) << "Real output node of top cell is " << output_node->DebugString();
  last_node_ = output_node;
  sens_value_ = FilterSensValues(sens_out);
}

FuncGraphPtr AutoGradCellImpl::Finish(const AnfNodePtrList &weights, const std::vector<size_t> &grad_position,
                                      const GradAttr &grad_attr) {
  // Set sens node and weights node
  SetSensAndWeights(weights, grad_attr.has_sens);

  // BackPropagate sensitivity, except when the last node is a valuenode which may be obtained by constant folding;
  if (!last_node_->isa<ValueNode>() && !last_node_->isa<Parameter>()) {
    (void)BackPropagate();
  }
  // Return the gradient;
  if (grad_attr.get_by_position && grad_position.empty()) {
    MS_LOG(EXCEPTION) << "grad_position should not be empty when grad by position!";
  }
  SetOutput(weights, grad_position, grad_attr);
  // Replace Parameter of primal funcgraph  with parameter of tape_;
  ReplacePrimalParameter(weights, grad_attr.has_sens);
  pynative::PyNativeAlgo::Common::DumpGraphIR("before_final_opt.ir", tape_);
  return tape_;
}

CNodePtr AutoGradCellImpl::ConstructBpropGraphInput(const GradParamPtr &grad_param, const AnfNodePtr &dout) {
  MS_EXCEPTION_IF_NULL(grad_param);
  std::vector<AnfNodePtr> node_list;
  (void)node_list.emplace_back(grad_param->cnode->input(0));
  if (grad_param->grad_by_value) {
    for (size_t i = 0; i < grad_param->op_args.size(); ++i) {
      const auto &v = grad_param->op_args[i];
      auto node = grad_param->cnode->input(i + 1);
      if (node->isa<Parameter>()) {
        node_list.emplace_back(node);
        node->set_abstract(v->ToAbstract());
        continue;
      }
      auto v_node = NewValueNode(grad_param->op_args[i]);
      v_node->set_abstract(grad_param->op_args[i]->ToAbstract());
      node_list.emplace_back(v_node);
    }
  } else {
    // Input is a Parameter or cnode, not a value node
    BuildKNodeListFromPrimalCNode(grad_param->cnode, grad_param->op_args, &node_list);
  }
  auto out_node = NewValueNode(grad_param->out);
  auto out_abs = grad_param->out->ToAbstract()->Broaden();
  out_node->set_abstract(out_abs);
  // set out
  node_list.emplace_back(out_node);
  // set dout
  node_list.emplace_back(dout);
  auto input_node = tape_->NewCNode(node_list);
  input_node->set_abstract(out_abs);
  return input_node;
}

void AutoGradCellImpl::BuildKNodeListFromPrimalCNode(const CNodePtr &cnode, const ValuePtrList &op_args,
                                                     std::vector<AnfNodePtr> *const node_list) {
  MS_EXCEPTION_IF_NULL(cnode);
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    MS_LOG(DEBUG) << "Find input knode of node " << cnode->input(i)->DebugString();
    if (cnode->input(i)->isa<CNode>()) {
      const auto input_adjoint_iter = anfnode_to_variable_adjoint_.find(cnode->input(i));
      if (input_adjoint_iter == anfnode_to_variable_adjoint_.end()) {
        MS_LOG(EXCEPTION) << "Cannot find input in adjoint map, inp: " << cnode->input(i)->DebugString();
      }
      MS_EXCEPTION_IF_NULL(input_adjoint_iter->second->k_node());
      (void)node_list->emplace_back(input_adjoint_iter->second->k_node());
    } else {
      cnode->input(i)->set_abstract(op_args[i - 1]->ToAbstract());
      (void)node_list->emplace_back(cnode->input(i));
    }
  }
}

void AutoGradCellImpl::BuildKNode(const GradParamPtr &grad_param, const VariableNodePtr &VariableNode) {
  MS_EXCEPTION_IF_NULL(grad_param);
  AnfNodePtrList node_list;
  for (size_t i = 0; i < grad_param->cnode->inputs().size(); ++i) {
    (void)node_list.emplace_back(BuildKNodeForCNodeInput(grad_param->cnode->input(i)));
  }
  auto k_node = tape_->NewCNode(node_list);
  k_node->set_abstract(grad_param->out->ToAbstract()->Broaden());
  VariableNode->set_k_node(k_node);
}

AnfNodePtr AutoGradCellImpl::BuildKNodeForCNodeInput(const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(input_node);
  if (input_node->isa<CNode>()) {
    const auto input_adjoint_iter = anfnode_to_variable_adjoint_.find(input_node);
    if (input_adjoint_iter == anfnode_to_variable_adjoint_.end()) {
      MS_LOG(EXCEPTION) << "cannot find input in adjoint map, inp: " << input_node->DebugString();
    }
    return input_adjoint_iter->second->k_node();
  } else {
    return input_node;
  }
}

bool GradPynativeOp(const AutoGradCellImplPtr &k_cell, const GradParamPtr &grad_param) {
  return k_cell->KPynativeOp(grad_param);
}

void AutoGradCellImpl::UpdateNextEdges(const FunctionNodePtr &fn, const CNodePtr &cnode,
                                       const std::vector<CNodePtr> &dins, const ValuePtrList &op_args) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (dins.size() != op_args.size()) {
    MS_LOG(EXCEPTION) << "The size of dins is not same as op_args";
  }
  for (size_t i = 0; i < op_args.size(); ++i) {
    auto node = cnode->input(i + 1);
    auto din = dins[i];
    UpdateNextEdges(fn, node, din, op_args[i]);
  }
}

void AutoGradCellImpl::UpdateNextEdges(const FunctionNodePtr &fn, const AnfNodePtr &node, const AnfNodePtr &din,
                                       const ValuePtr &op_arg) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(din);
  MS_EXCEPTION_IF_NULL(op_arg);
  if (anfnode_to_variable_adjoint_.find(node) != anfnode_to_variable_adjoint_.end()) {
    auto variable = anfnode_to_variable_adjoint_.at(node);
    auto real_din = GetRealDin(fn, variable->out_value(), op_arg, din);
    fn->AddEdge(node, real_din);
  } else if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    if (IsPrimitiveCNode(cnode, prim::kPrimStopGradient) || IsPrimitiveCNode(cnode, prim::kPrimUpdateState)) {
      return;
    }
    if (IsPrimitiveCNode(cnode, prim::kPrimMakeTuple)) {
      if (!op_arg->isa<ValueSequence>()) {
        MS_LOG(EXCEPTION) << "op_arg type is not valuesequence";
      }
      auto value_seq = op_arg->cast<ValueSequencePtr>();
      for (size_t i = 0; i < value_seq->value().size(); ++i) {
        auto input_node = cnode->input(i + 1);
        auto sub_value = value_seq->value()[i];
        CNodePtr new_din = tape_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), din, NewValueNode(SizeToLong(i))});
        new_din->set_abstract(sub_value->ToAbstract()->Broaden());
        if (din == fn->fake_dout()) {
          AddUser(fn->fake_dout(), new_din, 1);
        }
        UpdateNextEdges(fn, input_node, new_din, sub_value);
      }
    } else if (IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem)) {
      auto src_node = cnode->input(1);
      auto index_value = GetValueNode<Int64ImmPtr>(cnode->input(2));
      if (index_value == nullptr) {
        MS_LOG(EXCEPTION) << "CNode input 2 should be a Int64Imm, CNode: " << cnode->DebugString();
      }
      UpdateNextEdges(fn, src_node, din, op_arg);
    } else {
      MS_LOG(EXCEPTION) << "Cnode should be tuplegetitem or maketuple " << cnode->DebugString();
    }
  } else if (node->isa<Parameter>()) {
    auto param = node->cast<ParameterPtr>();
    auto tensor = param->default_param();
    MS_EXCEPTION_IF_NULL(tensor);
    AddParameterNode(param, tensor);
    UpdateNextEdges(fn, node, din, op_arg);
  } else {
    MS_LOG(DEBUG) << "It is not a cnode: " << node->DebugString();
    return;
  }
}

void AutoGradCellImpl::BuildForwardLastNode() {
  if (last_node_->isa<ValueNode>() ||
      anfnode_to_variable_adjoint_.find(last_node_) != anfnode_to_variable_adjoint_.end()) {
    return;
  }
  if (anfnode_to_variable_adjoint_.find(last_node_) == anfnode_to_variable_adjoint_.end()) {
    auto zeros_like_node = BuildZerosLikeNode(tape_, sens_value_);
    auto fn = std::make_shared<FunctionNode>(tape_, zeros_like_node);
    // If last_node is a maketuple or tuplegetitem, need update next edges,
    // if last_node is parameter, not need to update next edges.
    if (last_node_->isa<CNode>()) {
      UpdateNextEdges(fn, last_node_, zeros_like_node, sens_value_);
    }
    auto input_adjoint = std::make_shared<VariableNode>(fn, sens_value_);
    anfnode_to_variable_adjoint_.insert(std::make_pair(last_node_, input_adjoint));
  } else {
    MS_LOG(EXCEPTION) << "Unprocessed node: " << last_node_->DebugString();
  }
}

void AutoGradCellImpl::AddParameterNode(const AnfNodePtr &parameter, const ValuePtr &tensor) {
  MS_EXCEPTION_IF_NULL(parameter);
  MS_EXCEPTION_IF_NULL(tensor);
  auto zeros_like_dout = BuildZerosLikeNode(tape_, tensor);
  auto func_node = std::make_shared<FunctionNode>(tape_, zeros_like_dout);
  auto input_adjoint = std::make_shared<VariableNode>(func_node, tensor);
  anfnode_to_variable_adjoint_.insert(std::make_pair(parameter, input_adjoint));
  weights_.push_back(parameter);
}

AnfNodePtr AutoGradCellImpl::GetRealDin(const FunctionNodePtr &fn, const ValuePtr &out_value, const ValuePtr &sub_value,
                                        const AnfNodePtr &din) {
  MS_EXCEPTION_IF_NULL(out_value);
  MS_EXCEPTION_IF_NULL(sub_value);
  MS_EXCEPTION_IF_NULL(din);
  std::string out_value_id = pynative::PyNativeAlgo::Common::GetIdByValue(out_value);
  std::string sub_value_id = pynative::PyNativeAlgo::Common::GetIdByValue(sub_value);
  if (out_value_id == sub_value_id) {
    return din;
  } else if (out_value->isa<tensor::Tensor>()) {
    return BuildZerosLikeNode(tape_, out_value);
  } else if (out_value->isa<ValueSequence>()) {
    std::vector<AnfNodePtr> inputs;
    if (out_value->isa<ValueTuple>()) {
      (void)inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    } else {
      (void)inputs.emplace_back(NewValueNode(prim::kPrimMakeList));
    }
    auto value_seq = out_value->cast<ValueSequencePtr>();
    int index = -1;
    for (auto value : value_seq->value()) {
      auto real_din = GetRealDin(fn, value, sub_value, din);
      (void)inputs.emplace_back(real_din);

      // if exist din == fake_dout, we record it in user vector
      if (din == fn->fake_dout() && real_din == din) {
        index = inputs.size() - 1;
      }
    }
    auto new_din = tape_->NewCNode(inputs);
    new_din->set_abstract(out_value->ToAbstract()->Broaden());
    if (index != -1) {
      AddUser(fn->fake_dout(), new_din, index);
    }
    return new_din;
  }
  return nullptr;
}

void AutoGradCellImpl::BuildBPropCutCNode(const CNodePtr &cnode, std::vector<CNodePtr> *outputs) {
  auto prim = GetCNodePrimitive(cnode);
  if (prim == nullptr) {
    MS_LOG(EXCEPTION) << "Should be primitive, but: " << cnode->DebugString();
  }

  auto prim_py = prim->cast<PrimitivePyPtr>();
  MS_EXCEPTION_IF_NULL(prim_py);
  auto bprop_cut = std::make_shared<PrimitivePy>("bprop_cut");
  bprop_cut->CopyHookFunction(prim_py);
  prim_py->AddBpropCutPrim(bprop_cut);
  if (prim->HasAttr("cell_id")) {
    auto cell_id = GetValue<std::string>(prim->GetAttr("cell_id"));
    if (cell_id != "") {
      (void)bprop_cut->AddAttr("cell_hook", MakeValue(true));
      (void)bprop_cut->AddAttr("cell_id", MakeValue(cell_id));
    }
  }
  if (prim->HasAttr("custom_op_bprop")) {
    (void)bprop_cut->AddAttr("custom_op_bprop", MakeValue(true));
  }

  std::vector<AnfNodePtr> inputs{NewValueNode(bprop_cut)};
  auto output = tape_->NewCNode(inputs);
  AbstractBasePtrList abs;
  size_t args_size = cnode->size() - 2;
  for (size_t i = 1; i < cnode->size(); ++i) {
    output->add_input(cnode->input(i));
    AddUser(cnode->input(i), output, i);
    if (i < args_size) {
      auto din = tape_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), output, NewValueNode(SizeToLong(i - 1))});
      din->set_abstract(cnode->input(i)->abstract()->Broaden());
      outputs->emplace_back(din);
      (void)abs.emplace_back(din->abstract());
    }
  }
  output->set_abstract(std::make_shared<abstract::AbstractTuple>(abs));
  return;
}

void AutoGradCellImpl::BuildCustomBpropCNode(const CNodePtr &cnode, std::vector<CNodePtr> *outputs) {
  auto prim = GetCNodePrimitive(cnode);
  if (prim == nullptr) {
    MS_LOG(EXCEPTION) << "Should be primitive, but: " << cnode->DebugString();
  }
  MS_LOG(DEBUG) << "Build custom bprop: " << prim->name();
  auto prim_py = prim->cast<PrimitivePyPtr>();
  {
    py::gil_scoped_acquire gil;
    py::function fn;
    if (prim->is_base()) {
      fn = GetBpropFunction(prim->name());
    } else {
      fn = prim->cast_ptr<PrimitivePy>()->GetBpropFunction();
      if (py::isinstance<py::none>(fn)) {
        fn = GetBpropFunction(prim->name());
      }
    }
    if (!fn || py::isinstance<py::none>(fn)) {
      MS_LOG(INFO) << "Fail to find bprop function for " << prim->name() << ". fn: " << py::str(fn);
      return;
    }
    prim_py->AddBackwardHookFn(0, fn);
    prim_py->AddAttr("custom_op_bprop", MakeValue(True));
  }
  BuildBPropCutCNode(cnode, outputs);
}

void AutoGradCellImpl::SetSensAndWeights(const AnfNodePtrList &weights, bool has_sens_arg) {
  MS_EXCEPTION_IF_NULL(last_node_);
  MS_LOG(DEBUG) << "Last node info " << last_node_->DebugString();

  BuildForwardLastNode();

  // Add sens parameter
  ParameterPtr sens_param = nullptr;
  if (has_sens_arg) {
    sens_param = tape_->add_parameter();
    sens_param->debug_info()->set_name("sens");
    sens_param->set_abstract(sens_value_->ToAbstract()->Broaden());
  }

  // update dout for dout
  if (anfnode_to_variable_adjoint_.find(last_node_) != anfnode_to_variable_adjoint_.end()) {
    auto variable = anfnode_to_variable_adjoint_.at(last_node_);
    if (has_sens_arg && sens_param != nullptr) {
      variable->fn()->UpdateAccumulativeDout(sens_param);
    } else {
      variable->fn()->UpdateAccumulativeDout(BuildOnesLikeNode(tape_, sens_value_));
    }
  }

  // Add weights parameter
  need_grad_weights_.clear();
  for (const auto &weight : weights) {
    TraceGuard trace_guard(std::make_shared<TraceCopy>(weight->debug_info()));
    auto p = tape_->add_parameter();
    (void)need_grad_weights_.emplace(weight);
    auto input_w = weight->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(input_w);
    // Use name to match weight parameter in high order
    auto default_param = input_w->default_param();
    p->set_name(input_w->name());
    p->set_default_param(default_param);
    p->set_abstract(default_param->ToAbstract()->Broaden());
  }
}

OrderedMap<AnfNodePtr, VariableNodePtr>::reverse_iterator AutoGradCellImpl::GetLastNodeReverseIter() {
  for (auto iter = anfnode_to_variable_adjoint_.rbegin(); iter != anfnode_to_variable_adjoint_.rend(); ++iter) {
    if (!iter->first->isa<CNode>()) {
      continue;
    }
    if (iter->first->cast<CNodePtr>() == last_node_) {
      auto &variable = anfnode_to_variable_adjoint_[last_node_];
      variable->set_is_need_propagate(true);
      return iter;
    }
  }
  return anfnode_to_variable_adjoint_.rend();
}

void AutoGradCellImpl::BackPropagate() {
  const auto &last_node_reverse_iter = GetLastNodeReverseIter();
  for (auto iter = last_node_reverse_iter; iter != anfnode_to_variable_adjoint_.rend(); ++iter) {
    MS_LOG(DEBUG) << "BackPropagate cnode: " << iter->first->DebugString();
    auto variable = iter->second;
    if (!variable->is_need_propagate()) {
      continue;
    }
    if (variable->is_need_propagate() && variable->is_fake_bprop()) {
      MS_LOG(EXCEPTION) << variable->fake_prim_name() << " op has not corresponding bprop!";
    }
    auto fn = variable->fn();
    // replace real dout to fake dout
    Replace(fn->fake_dout(), fn->RealDout());
    // replace edges which exist fake dout
    fn->ReplaceEdges();

    auto &next_edges = fn->next_edges();
    for (const auto &next_edge : next_edges) {
      auto node = next_edge.first;
      auto din = next_edge.second;
      if (anfnode_to_variable_adjoint_.find(node) == anfnode_to_variable_adjoint_.end()) {
        MS_LOG(EXCEPTION) << "current node not find corresponding node";
      }
      auto last_variable = anfnode_to_variable_adjoint_[node];
      last_variable->fn()->UpdateAccumulativeDout(din);
      last_variable->set_is_need_propagate(true);
    }
  }
}

AnfNodePtr AutoGradCellImpl::GetGradNodeByIndex(const AnfNodePtrList &node_list, size_t index) {
  if (index >= node_list.size()) {
    MS_LOG(EXCEPTION) << "Position index " << index << " is exceed input size.";
  }
  auto grad_node = node_list[index];
  MS_EXCEPTION_IF_NULL(grad_node);

  const auto &input_adjoint_iter = anfnode_to_variable_adjoint_.find(grad_node);
  if (input_adjoint_iter == anfnode_to_variable_adjoint_.end()) {
    // If weight is not used in the forward network, just return zeros_like() as dout.
    if (grad_node->isa<Parameter>()) {
      MS_LOG(WARNING) << "Weight does not participate in forward calculation, weight: " << grad_node->DebugString();
      auto w = grad_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(w);
      auto default_param = w->default_param();
      MS_EXCEPTION_IF_NULL(default_param);
      return BuildZerosLikeNode(tape_, default_param);
    }
    // If input is not used in the forward network, just return zeros_like() as dout.
    MS_LOG(EXCEPTION) << "Input does not participate in forward calculation, input: " << grad_node->DebugString();
    // to do
    // return BuildZerosLikeNode(tape_, grad_node);
    return nullptr;
  }
  return input_adjoint_iter->second->fn()->RealDout();
}

AnfNodePtr AutoGradCellImpl::GetInputGrad(bool grad_all_inputs, bool get_by_position,
                                          const std::vector<size_t> &grad_position) {
  std::vector<size_t> grad_pos_list;
  if (get_by_position) {
    grad_pos_list = grad_position;
  } else if (grad_all_inputs) {
    grad_pos_list.resize(cell_inputs_.size());
    iota(grad_pos_list.begin(), grad_pos_list.end(), 0);
  } else {
    return nullptr;
  }

  AnfNodePtrList inputs_grad_list{NewValueNode(prim::kPrimMakeTuple)};
  AbstractBasePtrList inputs_grad_spec;
  if (!cell_inputs_.empty()) {
    for (size_t index : grad_pos_list) {
      auto grad_node = GetGradNodeByIndex(cell_inputs_, index);
      MS_EXCEPTION_IF_NULL(grad_node);
      (void)inputs_grad_list.emplace_back(grad_node);
      (void)inputs_grad_spec.emplace_back(grad_node->abstract());
    }
    constexpr size_t single_pos_size = 1;
    if (get_by_position && grad_pos_list.size() == single_pos_size) {
      return inputs_grad_list[single_pos_size];
    }
  }
  auto input_grad_ret = tape_->NewCNode(inputs_grad_list);
  input_grad_ret->set_abstract(std::make_shared<abstract::AbstractTuple>(inputs_grad_spec));
  return input_grad_ret;
}

AnfNodePtr AutoGradCellImpl::GetWeightGrad(bool grad_weights, const AnfNodePtrList &weights,
                                           bool weight_param_is_tuple) {
  // No need to return gradient of weights.
  if (!grad_weights) {
    return nullptr;
  }
  if (weight_param_is_tuple) {
    AnfNodePtrList weights_grad_list{NewValueNode(prim::kPrimMakeTuple)};
    AbstractBasePtrList weights_grad_spec;
    for (size_t index = 0; index < weights.size(); ++index) {
      auto grad_node = GetGradNodeByIndex(weights, index);
      MS_EXCEPTION_IF_NULL(grad_node);
      (void)weights_grad_list.emplace_back(grad_node);
      (void)weights_grad_spec.emplace_back(grad_node->abstract());
    }
    auto weight_grad_ret = tape_->NewCNode(weights_grad_list);
    weight_grad_ret->set_abstract(std::make_shared<abstract::AbstractTuple>(weights_grad_spec));
    return weight_grad_ret;
  } else {
    return GetGradNodeByIndex(weights, 0);
  }
}

bool AutoGradCellImpl::IsOutputBothEmpty(const AnfNodePtr &inputs_grad, const AnfNodePtr &weights_grad) const {
  if (!inputs_grad->isa<CNode>() || !weights_grad->isa<CNode>()) {
    return false;
  }
  auto inputs_grad_cnode = inputs_grad->cast<CNodePtr>();
  auto weights_grad_cnode = weights_grad->cast<CNodePtr>();
  if (!IsPrimitiveCNode(inputs_grad_cnode, prim::kPrimMakeTuple) ||
      !IsPrimitiveCNode(weights_grad_cnode, prim::kPrimMakeTuple)) {
    return false;
  }
  constexpr int kEmptyTupeSize = 1;
  if (inputs_grad_cnode->size() != kEmptyTupeSize || weights_grad_cnode->size() != kEmptyTupeSize) {
    return false;
  }
  return true;
}

AnfNodePtr AutoGradCellImpl::GenerateEmptyTupleValue() {
  std::vector<ValuePtr> value_list;
  auto inputs_value = std::make_shared<ValueTuple>(value_list);
  auto weights_value = std::make_shared<ValueTuple>(value_list);
  std::vector<ValuePtr> tuple_list{inputs_value, weights_value};
  auto tuple_value = std::make_shared<ValueTuple>(tuple_list);
  auto tuple_value_node = NewValueNode(tuple_value);
  tuple_value_node->set_abstract(tuple_value->ToAbstract());
  return tuple_value_node;
}

void AutoGradCellImpl::SetOutput(const AnfNodePtrList &weights, const std::vector<size_t> &grad_position,
                                 const GradAttr &grad_attr) {
  auto inputs_grad_ret = GetInputGrad(grad_attr.grad_all_inputs, grad_attr.get_by_position, grad_position);
  auto weights_grad_ret = GetWeightGrad(grad_attr.grad_weights, weights, grad_attr.weight_param_is_tuple);
  // Gradients wrt inputs and weights.
  if (inputs_grad_ret != nullptr && weights_grad_ret != nullptr) {
    if (IsOutputBothEmpty(inputs_grad_ret, weights_grad_ret)) {
      auto tape_output = GenerateEmptyTupleValue();
      tape_->set_output(tape_output);
    } else {
      auto tape_output = tape_->NewCNode({NewValueNode(prim::kPrimMakeTuple), inputs_grad_ret, weights_grad_ret});
      tape_output->set_abstract(std::make_shared<abstract::AbstractTuple>(
        abstract::AbstractBasePtrList{inputs_grad_ret->abstract(), weights_grad_ret->abstract()}));
      tape_->set_output(tape_output);
    }
    return;
  }
  // Gradients wrt inputs.
  if (inputs_grad_ret != nullptr) {
    tape_->set_output(inputs_grad_ret);
    return;
  }
  // Gradients wrt weights.
  if (weights_grad_ret != nullptr) {
    tape_->set_output(weights_grad_ret);
    return;
  }
  // grad_all_inputs, grad_weights and get_by_position are all false.
  AnfNodePtr tape_output = nullptr;
  if (cell_inputs_.empty()) {
    // If no input nodes, return empty tuple.
    tape_output = tape_->NewCNode({NewValueNode(prim::kPrimMakeTuple)});
    abstract::AbstractBasePtrList abs{};
    tape_output->set_abstract(std::make_shared<abstract::AbstractTuple>(abs));
  } else {
    // If there are input nodes, return gradient of first input node.
    tape_output = GetGradNodeByIndex(cell_inputs_, 0);
  }
  tape_->set_output(tape_output);
}

void AutoGradCellImpl::AddUser(const AnfNodePtr &node, const CNodePtr &user, size_t index) {
  if (users_.find(node) == users_.end()) {
    users_[node] = {};
  }
  (void)users_[node].emplace_back(make_pair(user, index));
}

void AutoGradCellImpl::Replace(const AnfNodePtr &old_node, const AnfNodePtr &new_node) {
  if (users_.find(old_node) == users_.end()) {
    return;
  }
  auto &old_node_users = users_[old_node];
  for (const auto &pair_node : old_node_users) {
    auto cnode = pair_node.first;
    size_t index = pair_node.second;
    if (index >= cnode->size()) {
      MS_LOG(EXCEPTION) << "exception for index:" << index << "greater than cnode size:" << cnode->size();
    }
    cnode->set_input(index, new_node);
  }
}

void AutoGradCellImpl::ElimateTupleGetItem() {
  for (auto iter = users_.begin(); iter != users_.end(); iter++) {
    auto old_node = iter->first;
    if (!old_node->isa<CNode>()) {
      continue;
    }
    auto old_cnode = old_node->cast<CNodePtr>();
    if (IsPrimitiveCNode(old_cnode, prim::kPrimTupleGetItem)) {
      auto tuple_node = old_cnode->input(1);
      if (!tuple_node->isa<CNode>() || !IsPrimitiveCNode(tuple_node->cast<CNodePtr>(), prim::kPrimMakeTuple)) {
        continue;
      }
      auto index_value = GetValueNode<Int64ImmPtr>(old_cnode->input(2));
      size_t index = LongToSize(index_value->value());
      auto tuple_cnode = tuple_node->cast<CNodePtr>();
      Replace(old_node, tuple_cnode->input(index + 1));
    }
  }
}

void AutoGradCellImpl::ReplacePrimalParameter(const AnfNodePtrList &weights, bool has_sens_arg) {
  const auto &parameters = tape_->parameters();
  auto cell_inputs_size = cell_inputs_.size();
  if (need_do_manager_replace_) {
    MS_LOG(DEBUG) << "Do parameter replace by manager";
    auto mng = MakeManager({tape_}, false);
    auto tr = mng->Transact();

    for (size_t i = 0; i < cell_inputs_size; ++i) {
      (void)tr.Replace(cell_inputs_[i], parameters[i]);
    }
    // (Inputs, sens, weights) or (Inputs, weights)
    size_t weight_offset = cell_inputs_size;
    if (has_sens_arg) {
      weight_offset = weight_offset + 1;
    }
    for (size_t i = 0; i < weights.size(); ++i) {
      (void)tr.Replace(weights[i], parameters[weight_offset + i]);
    }
    tr.Commit();
    need_do_manager_replace_ = false;
  } else {
    for (size_t i = 0; i < cell_inputs_size; ++i) {
      Replace(cell_inputs_[i], parameters[i]);
    }
    size_t weight_offset = cell_inputs_size;
    if (has_sens_arg) {
      weight_offset = weight_offset + 1;
    }
    for (size_t i = 0; i < weights.size(); ++i) {
      Replace(weights[i], parameters[weight_offset + i]);
    }
  }

  for (auto &weight : weights_) {
    if (need_grad_weights_.find(weight) == need_grad_weights_.end()) {
      auto parameter = weight->cast<ParameterPtr>();
      const auto &input_value = parameter->default_param();
      MS_EXCEPTION_IF_NULL(input_value);
      auto value_node = NewValueNode(input_value);
      value_node->set_abstract(input_value->ToAbstract()->Broaden());
      Replace(weight, value_node);
    }
  }
  ElimateTupleGetItem();
}

AutoGradCellImplPtr GradPynativeCellBegin(const AnfNodePtrList &cell_inputs,
                                          const std::vector<ValuePtr> &input_param_values) {
  auto abstract_are_set = std::all_of(cell_inputs.cbegin(), cell_inputs.cend(),
                                      [](const AnfNodePtr &node) { return node->abstract() != nullptr; });
  if (!abstract_are_set) {
    MS_LOG(EXCEPTION) << "Not all abstract_value in cell_inputs are set";
  }
  if (cell_inputs.size() != input_param_values.size()) {
    MS_LOG(EXCEPTION) << "The size of cell inputs " << cell_inputs.size()
                      << " is not equal to the size of input parameter values " << input_param_values.size();
  }
  return std::make_shared<AutoGradCellImpl>(cell_inputs, input_param_values);
}

FuncGraphPtr GradPynativeCellEnd(const AutoGradCellImplPtr &auto_grad_cell, const AnfNodePtrList &weights,
                                 const std::vector<size_t> &grad_position, const GradAttr &grad_attr) {
  return auto_grad_cell->Finish(weights, grad_position, grad_attr);
}
}  // namespace ad
}  // namespace mindspore
