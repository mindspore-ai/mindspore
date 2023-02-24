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

#include "pipeline/pynative/grad/auto_grad.h"
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include "mindspore/core/ops/core_ops.h"
#include "ir/anf.h"
#include "ir/func_graph_cloner.h"
#include "frontend/optimizer/ad/adjoint.h"
#include "utils/info.h"
#include "pipeline/jit/debug/trace.h"
#include "pipeline/pynative/grad/bprop_expander/bprop.h"
#include "pipeline/pynative/pynative_utils.h"
#include "utils/profile.h"
#include "include/common/utils/primitive_utils.h"
#include "pipeline/jit/pass.h"
namespace mindspore {
namespace pynative {
namespace autograd {
namespace {
enum class SpecialType { kZerosLikeType = 0, kOnesLikeType = 1 };
const std::map<SpecialType, std::shared_ptr<Primitive>> kValueType{{SpecialType::kZerosLikeType, prim::kPrimZerosLike},
                                                                   {SpecialType::kOnesLikeType, prim::kPrimOnesLike}};
const size_t kContainerRatio = 2;
const PrimitiveSet kGradBlackList{
  prim::kPrimMakeTuple,           prim::kPrimTupleGetItem,      prim::kPrimStopGradient,       prim::kPrimUpdateState,
  prim::kPrimNPUAllocFloatStatus, prim::kPrimNPUGetFloatStatus, prim::kPrimNPUClearFloatStatus};

const PrimitiveSet kMonadPrim = {prim::kPrimLoad, prim::kPrimDepend, prim::kPrimUpdateState};

mindspore::HashMap<std::string, FuncGraphPtr> pass_grad_graph_;

void ClearDeviceAddress(const ValuePtr &value) {
  std::vector<tensor::TensorPtr> tensors;
  TensorValueToTensor(value, &tensors);
  for (auto tensor : tensors) {
    tensor->set_device_address(nullptr);
  }
}

inline bool IsPrimNeedGrad(const PrimitivePtr &prim) { return kGradBlackList.find(prim) == kGradBlackList.end(); }

ValueNodePtr CreateValueNodeByClonedValue(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  auto cloned_v = ShallowCopyTensorValue(v);
  ClearDeviceAddress(cloned_v);
  auto v_node = NewValueNode(cloned_v);
  v_node->set_abstract(PyNativeAlgo::Common::SetAbstractValueToAnyValue(cloned_v->ToAbstract()));
  return v_node;
}

ValueNodePtr GetSparseTensorShapeNode(const ShapeVector &shape) {
  auto value_shape = NewValueNode(shape);
  std::vector<abstract::AbstractBasePtr> abstract_shape;
  (void)std::transform(
    shape.begin(), shape.end(), std::back_inserter(abstract_shape),
    [](auto shp) -> abstract::AbstractScalarPtr { return std::make_shared<abstract::AbstractScalar>(shp); });
  auto abs_shape = std::make_shared<abstract::AbstractTuple>(abstract_shape);
  value_shape->set_abstract(abs_shape);
  return value_shape;
}

AnfNodePtr ZerosLike(const FuncGraphPtr &tape, const AbstractBasePtr &abstract) {
  MS_EXCEPTION_IF_NULL(abstract);
  auto abs = PyNativeAlgo::Common::SetAbstractValueToAnyValue(abstract);
  auto output = tape->NewCNode({NewValueNode(prim::kPrimZerosLike), NewValueNode(std::make_shared<tensor::Tensor>(0))});
  output->set_abstract(abs);
  return output;
}

AnfNodePtr BuildSpecialLikeSparseTensor(const FuncGraphPtr &tape, const ValuePtr &sparse_value,
                                        const AnfNodePtr &dout_value_node) {
  MS_EXCEPTION_IF_NULL(tape);
  MS_EXCEPTION_IF_NULL(sparse_value);
  if (sparse_value->isa<tensor::CSRTensor>()) {
    auto csr_tensor = sparse_value->cast<tensor::CSRTensorPtr>();
    MS_EXCEPTION_IF_NULL(csr_tensor);
    auto indptr_node = CreateValueNodeByClonedValue(csr_tensor->GetIndptr());
    auto indices_node = CreateValueNodeByClonedValue(csr_tensor->GetIndices());
    auto value_shape = GetSparseTensorShapeNode(csr_tensor->shape());
    auto special_like_csr_node =
      tape->NewCNode({NewValueNode(prim::kPrimMakeTuple), indptr_node, indices_node, dout_value_node, value_shape});
    special_like_csr_node->set_abstract(sparse_value->ToAbstract()->Broaden());
    return special_like_csr_node;
  } else if (sparse_value->isa<tensor::COOTensor>()) {
    auto coo_tensor = sparse_value->cast<tensor::COOTensorPtr>();
    MS_EXCEPTION_IF_NULL(coo_tensor);
    auto indices_node = CreateValueNodeByClonedValue(coo_tensor->GetIndices());
    auto value_shape = GetSparseTensorShapeNode(coo_tensor->shape());
    auto special_like_coo_node =
      tape->NewCNode({NewValueNode(prim::kPrimMakeTuple), indices_node, dout_value_node, value_shape});
    special_like_coo_node->set_abstract(sparse_value->ToAbstract()->Broaden());
    return special_like_coo_node;
  }
  MS_LOG(EXCEPTION) << "Get invalid sparse tensor";
}

AnfNodePtr BuildSpecialLikeValue(const FuncGraphPtr &tape, const ValuePtr &value, const SpecialType &type) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>() || value->isa<Scalar>()) {
    auto vlaue_node = NewValueNode(value);
    auto value_abs = PyNativeAlgo::Common::SetAbstractValueToAnyValue(value->ToAbstract());
    vlaue_node->set_abstract(value_abs);
    auto primitive = kValueType.at(type);
    MS_EXCEPTION_IF_NULL(primitive);
    auto special_like_value = tape->NewCNode({NewValueNode(primitive), vlaue_node});
    special_like_value->set_abstract(value_abs);
    return special_like_value;
  } else if (value->isa<tensor::CSRTensor>()) {
    auto csr_tensor = value->cast<tensor::CSRTensorPtr>();
    MS_EXCEPTION_IF_NULL(csr_tensor);
    auto data = csr_tensor->GetValues();
    auto cloned_data = ShallowCopyTensorValue(data);
    ClearDeviceAddress(cloned_data);
    return BuildSpecialLikeValue(tape, cloned_data, type);
  } else if (value->isa<tensor::COOTensor>()) {
    auto coo_tensor = value->cast<tensor::COOTensorPtr>();
    MS_EXCEPTION_IF_NULL(coo_tensor);
    auto data = coo_tensor->GetValues();
    auto cloned_data = ShallowCopyTensorValue(data);
    ClearDeviceAddress(cloned_data);
    return BuildSpecialLikeValue(tape, cloned_data, type);
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
    auto value_abs = PyNativeAlgo::Common::SetAbstractValueToAnyValue(value->ToAbstract());
    special_like_value->set_abstract(value_abs);
    return special_like_value;
  } else if (value->isa<ValueDictionary>()) {
    const auto &dic_v = value->cast<ValueDictionaryPtr>()->value();
    std::vector<ValuePtr> v_list;
    std::transform(dic_v.begin(), dic_v.end(), std::back_inserter(v_list),
                   [](const std::pair<ValuePtr, ValuePtr> &elem) { return elem.second; });
    return BuildSpecialLikeValue(tape, std::make_shared<ValueTuple>(v_list), type);
  } else if (value->isa<None>() || value->isa<Type>()) {
    return BuildSpecialLikeValue(tape, MakeValue(0), type);
  } else {
    MS_EXCEPTION(TypeError) << "For value " << value->ToString() << ", the type is not support now";
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
  pynative::PyNativeAlgo::Common::DumpGraphIR("bprop_builder_before_opt.ir", bprop_func_graph);
  pipeline::ResourcePtr resource = std::make_shared<pipeline::Resource>();
  resource->set_func_graph(bprop_func_graph);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(bprop_func_graph);
  auto after_opt_bg = pipeline::MsFunctionBpropGraphPass(resource, true);
  pynative::PyNativeAlgo::Common::DumpGraphIR("bprop_builder_after_opt.ir", after_opt_bg);
  return after_opt_bg;
}

// Handle bprob of op which input dtype is real number and output dtype is complex number.
// If the dtype of a gradient(din) is complex number and the input of that is real number,
// only the real part of the gradient make sense in back propagate. So we handle it by
// insert a Real() ops after the gradient.
// input: AnfNode with input of op which input dtype is real number and output dtype is complex number.
// din: CNodePtr with gradient of input.
// tape: Funcgraph witch input and din belong to.
// return: New din with inserted real op if necessarily.
AnfNodePtr HandleRealToComplex(const AnfNodePtr &input, const AnfNodePtr &din, const FuncGraphPtr &tape) {
  MS_EXCEPTION_IF_NULL(din);
  TypePtr din_type = din->Type();
  if (din_type == nullptr || !din_type->isa<TensorType>()) {
    return din;
  }
  din_type = din_type->cast_ptr<TensorType>()->element();
  MS_EXCEPTION_IF_NULL(din_type);
  if (din_type->type_id() != kNumberTypeComplex64 && din_type->type_id() != kNumberTypeComplex128) {
    return din;
  }

  MS_EXCEPTION_IF_NULL(input);
  TypePtr input_type = input->Type();
  if (input_type == nullptr || !input_type->isa<TensorType>()) {
    return din;
  }
  input_type = input_type->cast_ptr<TensorType>()->element();
  MS_EXCEPTION_IF_NULL(input_type);
  if (input_type->type_id() == kNumberTypeComplex64 || input_type->type_id() == kNumberTypeComplex128) {
    return din;
  }

  AnfNodePtr new_din = tape->NewCNode({NewValueNode(prim::kPrimReal), din});
  AbstractBasePtr abs = std::make_shared<abstract::AbstractTensor>(
    abstract::AbstractTensor(input_type, input->abstract()->GetShapeTrack()));
  new_din->set_abstract(abs);
  return new_din;
}

bool IsOutputBothEmpty(const AnfNodePtr &inputs_grad, const AnfNodePtr &weights_grad) {
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

AnfNodePtr GenerateEmptyTupleValue() {
  std::vector<ValuePtr> value_list;
  auto inputs_value = std::make_shared<ValueTuple>(value_list);
  auto weights_value = std::make_shared<ValueTuple>(value_list);
  std::vector<ValuePtr> tuple_list{inputs_value, weights_value};
  auto tuple_value = std::make_shared<ValueTuple>(tuple_list);
  auto tuple_value_node = NewValueNode(tuple_value);
  tuple_value_node->set_abstract(tuple_value->ToAbstract());
  return tuple_value_node;
}

bool IsValidTensorInput(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  return abs->isa<abstract::AbstractTensor>() || abs->isa<abstract::AbstractSparseTensor>();
}

bool IsMonadPrim(const PrimitivePtr &prim, const CNodePtr &cnode, const GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(prim);
  if (kMonadPrim.find(prim) != kMonadPrim.end()) {
    MS_LOG(DEBUG) << "Get monad cnode " << cnode->DebugString();
    return true;
  }
  if (prim->HasAttr(GRAPH_FLAG_SIDE_EFFECT_MEM) || prim->HasAttr(GRAPH_FLAG_SIDE_EFFECT_IO)) {
    std::vector<AnfNodePtr> inputs{cnode->inputs().begin(), cnode->inputs().end() - 1};
    cnode->set_inputs(inputs);
  }
  MS_EXCEPTION_IF_NULL(grad_param);
  // Ms function graph contain monad op
  if (grad_param->is_ms_function_graph) {
    for (size_t i = 1; i < cnode->size(); ++i) {
      cnode->set_input(i, common::AnfAlgo::VisitKernelWithReturnType(cnode->input(i), 0, false,
                                                                     {prim::kPrimTupleGetItem, prim::kPrimMakeTuple})
                            .first);
    }
  }
  return false;
}

AnfNodePtr GetTupleItemNodeInput(const FuncGraphPtr &tape, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(tape);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  AnfNodePtr new_cnode = nullptr;
  if (IsPrimitive(cnode->input(kIndex1), prim::kPrimTupleGetItem)) {
    auto inner_cnode = cnode->input(kIndex1)->cast<CNodePtr>();
    new_cnode = tape->NewCNode(
      {inner_cnode->input(kIndex0), GetTupleItemNodeInput(tape, inner_cnode), inner_cnode->input(kIndex2)});
  } else {
    AnfNodePtrList new_inputs{cnode->inputs().begin(), cnode->inputs().end()};
    new_cnode = tape->NewCNode(new_inputs);
  }
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->set_abstract(cnode->abstract());
  return new_cnode;
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
    add_result->set_abstract(right_node->abstract());
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
    MS_LOG(EXCEPTION) << "Unknown cnode type" << left_node->DebugString();
  }
}

void FunctionNode::AddNextEdge(const AnfNodePtr &next_node, const AnfNodePtr &din) {
  MS_EXCEPTION_IF_NULL(next_node);
  MS_EXCEPTION_IF_NULL(din);
  // next_node and its corresponding din
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

AnfNodePtr VariableAdjoint::RealDout() {
  MS_EXCEPTION_IF_NULL(out_value_);
  auto &tape = fn()->tape();
  if (MS_UNLIKELY(IsZerosLikeNode(fn()->accumulate_dout()))) {
    fn()->set_accumulate_dout(BuildZerosLikeNode(fn()->tape(), out_value_));
  }
  const auto &accumulate_dout = fn()->accumulate_dout();
  const auto &dout_abs = accumulate_dout->abstract();
  MS_EXCEPTION_IF_NULL(dout_abs);
  // For input, if it is a sparsetensor, we need return a sparsetensor.
  if (out_value_->isa<tensor::Tensor>() || dout_abs->isa<abstract::AbstractSparseTensor>()) {
    return accumulate_dout;
  } else if (out_value_->isa<tensor::MetaSparseTensor>()) {
    return BuildSpecialLikeSparseTensor(tape, out_value_, accumulate_dout);
  }
  return accumulate_dout;
}

AutoGradCellImpl::AutoGradCellImpl(const AnfNodePtrList &cell_inputs, const std::vector<ValuePtr> &input_param_values,
                                   const AbstractBasePtrList &abs_list, size_t op_num_in_bprop_graph)
    : ad_param_(std::make_shared<AdParam>()), cell_inputs_(cell_inputs), op_num_in_bprop_graph_(op_num_in_bprop_graph) {
  ad_param()->tape_->debug_info()->set_name("grad_top");
  MS_LOG(DEBUG) << "Start AutoGradCellImpl, cell_inputs size: " << cell_inputs.size();
  if (op_num_in_bprop_graph_ != 0) {
    size_t estimate_element_num = op_num_in_bprop_graph_ * kContainerRatio;
    ad_param()->anfnode_to_variable_adjoint_.reserve(estimate_element_num);
    weights_used_in_graph_.reserve(estimate_element_num);
  }
  for (size_t i = 0; i < cell_inputs.size(); ++i) {
    TraceGuard trace_guard(std::make_shared<TraceCopy>(cell_inputs[i]->debug_info()));
    auto parameter = ad_param()->tape_->add_parameter();
    parameter->set_abstract(abs_list[i]);
    auto zeros_like_dout = ZerosLike(ad_param()->tape_, abs_list[i]);
    auto func_node = std::make_shared<FunctionNode>(ad_param()->tape_, zeros_like_dout);
    auto input_adjoint = std::make_shared<VariableAdjoint>(func_node, input_param_values[i]);
    (void)ad_param()->anfnode_to_variable_adjoint_.insert(std::make_pair(cell_inputs[i], input_adjoint));
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

  // construct zeroslike placeholder, if need use in bprop, we replace it in backprogate.
  AnfNodePtr dout = ZerosLike(ad_param()->tape_, grad_param->out->ToAbstract());
  auto fn = std::make_shared<FunctionNode>(ad_param()->tape_, dout);
  auto variable_adjoint = std::make_shared<VariableAdjoint>(fn, cloned_value);
  // Custom forward cnode no need record in bprop graph, because it is a flag cnode for run python. So just create
  // bprop_cut grad op is ok
  bool is_custom_prim =
    IsPrimitiveEquals(prim, prim::kPrimHookBackward) || IsPrimitiveEquals(prim, prim::kPrimCellBackwardHook);
  if (!grad_param->grad_by_value && !is_custom_prim) {
    variable_adjoint->set_k_node(BuildKNode(grad_param, true));
    need_do_manager_replace_ = true;
  }
  CNodePtr input_node = ConstructBpropGraphInput(grad_param, dout, variable_adjoint, is_custom_prim);
  MS_LOG(DEBUG) << "Construct input cnode: " << input_node->DebugString();
  // Gradient outputs
  std::vector<CNodePtr> outputs;
  if (is_custom_prim) {
    BuildBPropCutCNode(input_node, prim, &outputs);
  } else {
    auto ret = BpropExpander(&outputs, &ad_param()->users_).Run(input_node);
    if (!ret || outputs.empty()) {
      MS_LOG(DEBUG) << "Expander has no bprop of this node: " << grad_param->cnode->DebugString();
      BuildCustomBpropCNode(input_node, prim, &outputs);
    }
  }
  if (outputs.empty()) {
    MS_LOG(DEBUG) << "This op has not custom bprop: " << grad_param->cnode->DebugString();
    BuildFakeBpropCNode(input_node, &outputs);
    variable_adjoint->set_is_fake_bprop(true);
    variable_adjoint->set_fake_prim_name(prim->name());
  }
  UpdateNextEdges(variable_adjoint, grad_param->cnode, outputs, grad_param->op_args);
  (void)ad_param()->anfnode_to_variable_adjoint_.insert(std::make_pair(grad_param->cnode, variable_adjoint));
  // record last_node for brackpropagate
  ad_param()->last_node_ = grad_param->cnode;
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
      MS_EXCEPTION_IF_NULL(input_node->abstract());
      if (input_node->isa<Parameter>()) {
        (void)args_node_list.emplace_back(input_node);
        continue;
      }
      // Valuenode, cnode
      auto v_node = NewValueNode(grad_param->op_args[i]);
      v_node->set_abstract(input_node->abstract()->Clone());
      (void)args_node_list.emplace_back(v_node);
    }
    bprop_cnode = GetBpropGraphCNode(grad_param, args_node_list, &dout);
  } else {
    k_node = BuildKNode(grad_param, false);
    BuildKNodeListFromPrimalCNode(grad_param->cnode, grad_param->op_args, &args_node_list);
    bprop_cnode = GetBpropGraphCNode(grad_param, args_node_list, &dout);
  }
  auto fn = std::make_shared<FunctionNode>(ad_param()->tape_, dout);
  auto variable_adjoint = std::make_shared<VariableAdjoint>(fn, grad_param->out);
  variable_adjoint->set_k_node(k_node);
  std::vector<CNodePtr> outputs;
  for (size_t i = 1; i < grad_param->cnode->size(); ++i) {
    CNodePtr din = nullptr;
    if (grad_param->is_not_support_by_expander) {
      // bprop_app[0] env
      din =
        ad_param()->tape_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), bprop_cnode, NewValueNode(SizeToLong(i))});
    } else {
      din = ad_param()->tape_->NewCNode(
        {NewValueNode(prim::kPrimTupleGetItem), bprop_cnode, NewValueNode(SizeToLong(i - 1))});
    }
    MS_EXCEPTION_IF_NULL(grad_param->cnode->input(i)->abstract());
    din->set_abstract(grad_param->cnode->input(i)->abstract());
    (void)outputs.emplace_back(din);
  }
  UpdateNextEdges(variable_adjoint, grad_param->cnode, outputs, grad_param->op_args);
  (void)ad_param()->anfnode_to_variable_adjoint_.insert(std::make_pair(grad_param->cnode, variable_adjoint));
  need_do_manager_replace_ = true;
  return true;
}

CNodePtr AutoGradCellImpl::GetBpropGraphCNode(const GradParamPtr &grad_param, const AnfNodePtrList &args,
                                              AnfNodePtr *const tape_dout) {
  MS_EXCEPTION_IF_NULL(grad_param);
  if (grad_param->is_not_support_by_expander) {
    MS_LOG(DEBUG) << "Get control flow graph or op not support by expander";
    return GetBPropFromFProp(grad_param, args, tape_dout);
  }
  return GetBPropFromExpander(grad_param, args, tape_dout);
}

CNodePtr AutoGradCellImpl::GetBPropFromExpander(const GradParamPtr &grad_param, const AnfNodePtrList &args,
                                                AnfNodePtr *const tape_dout) {
  auto ad_graph = GradFuncGraph(grad_param);
  AnfNodePtrList bprop_inputs(args.begin(), args.end());

  // Call by tape_
  MS_EXCEPTION_IF_NULL(tape_dout);
  *tape_dout = BuildZerosLikeNode(ad_param()->tape_, grad_param->out);
  (void)bprop_inputs.emplace_back(*tape_dout);
  (void)bprop_inputs.insert(bprop_inputs.cbegin(), NewValueNode(ad_graph));
  auto get_bprop = ad_param()->tape_->NewCNode(bprop_inputs);
  get_bprop->set_abstract(ad_graph->output()->abstract());
  // tape_dout is set by next op
  AddUser(*tape_dout, get_bprop, bprop_inputs.size() - 1);
  return get_bprop;
}

CNodePtr AutoGradCellImpl::GetBPropFromFProp(const GradParamPtr &grad_param, const AnfNodePtrList &args,
                                             AnfNodePtr *const tape_dout) {
  MS_EXCEPTION_IF_NULL(grad_param);
  // Wrap tuple_getitem(fprop_app, 1) in a FuncGraph and optimize it;
  auto bprop_builder = std::make_shared<FuncGraph>();
  bprop_builder->debug_info()->set_name("bprop_builder");

  AnfNodePtrList fprop_app_inputs{NewValueNode(grad_param->fg)};
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

  // Get bprop from fprop_fg, it is 2th output of fprop_fg
  AnfNodePtrList node_list{get_bprop};
  auto dout = bprop_builder->add_parameter();
  dout->set_abstract(grad_param->cnode->abstract());
  (void)node_list.emplace_back(dout);
  auto call_bprop = bprop_builder->NewCNode(node_list);
  bprop_builder->set_output(call_bprop);

  // Call pass for optimize graph, such as inline
  auto after_opt_fg = OptimizeBpropBuilder(bprop_builder);

  // Call by tape_
  MS_EXCEPTION_IF_NULL(tape_dout);
  *tape_dout = ZerosLike(ad_param()->tape_, grad_param->cnode->abstract());
  (void)bprop_builder_inputs.emplace_back(*tape_dout);
  (void)bprop_builder_inputs.insert(bprop_builder_inputs.cbegin(), NewValueNode(after_opt_fg));
  get_bprop = ad_param()->tape_->NewCNode(bprop_builder_inputs);
  get_bprop->set_abstract(after_opt_fg->output()->abstract());
  // tape_dout is set by next op
  AddUser(*tape_dout, get_bprop, bprop_builder_inputs.size() - 1);
  return get_bprop;
}

FuncGraphPtr AutoGradCellImpl::GradFuncGraph(const GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(grad_param);
  // Find ad graph in cache
  if (!grad_param->use_dynamic_shape_process) {
    const auto it = pass_grad_graph_.find(grad_param->graph_cache_key);
    if (it != pass_grad_graph_.end()) {
      MS_LOG(DEBUG) << "Get ad grad graph by cache";
      return BasicClone(it->second);
    }
  } else {
    pass_grad_graph_.clear();
  }

  // Create new ad param for graph ad
  pynative::PyNativeAlgo::Common::DumpGraphIR("ad_input_graph.ir", grad_param->fg);
  auto current_ad_param = ad_param_;
  ad_param_ = std::make_shared<AdParam>();
  ad_param()->tape_->debug_info()->set_name("ad_graph");

  GradGraphByExpander(grad_param);

  if (ad_param()->last_node_ != nullptr) {
    // Set dout parameter
    if (kMonadPrim.find(GetCNodePrimitive(ad_param()->last_node_)) != kMonadPrim.end()) {
      ad_param()->last_node_ = common::AnfAlgo::VisitKernelWithReturnType(
                                 ad_param()->last_node_, 0, false, {prim::kPrimTupleGetItem, prim::kPrimMakeTuple})
                                 .first;
    }
    auto ad_graph_dout = ad_param()->tape_->add_parameter();
    ad_graph_dout->set_abstract(ad_param()->last_node_->abstract());
    ad_param()->anfnode_to_variable_adjoint_.at(ad_param()->last_node_)->fn()->UpdateAccumulativeDout(ad_graph_dout);

    (void)BackPropagate();
  }

  AnfNodePtrList outputs{NewValueNode(prim::kPrimMakeTuple)};
  abstract::AbstractBasePtrList out_abs_list;
  for (const auto &node : grad_param->fg->parameters()) {
    (void)outputs.emplace_back(ad_param()->anfnode_to_variable_adjoint_.at(node)->RealDout());
    (void)out_abs_list.emplace_back(outputs.back()->abstract());
  }
  auto ad_graph_out = ad_param()->tape_->NewCNode(outputs);
  ad_graph_out->set_abstract(std::make_shared<abstract::AbstractTuple>(out_abs_list));
  ad_param()->tape_->set_output(ad_graph_out);
  auto ad_graph = ad_param()->tape_;
  pynative::PyNativeAlgo::Common::DumpGraphIR("ad_output_graph.ir", ad_graph);

  // Save ad graph in cache
  if (!grad_param->use_dynamic_shape_process) {
    pass_grad_graph_[grad_param->graph_cache_key] = BasicClone(ad_graph);
  }
  // Replace cnode with valuenode for reduce compute
  bool ms_function_by_value = grad_param->is_ms_function_graph && grad_param->grad_by_value;
  if (ms_function_by_value) {
    pynative::PyNativeAlgo::Common::ReplaceCNodeWithValueNode(ad_graph);
  }
  // Restore ad param
  ad_param_ = current_ad_param;
  return ad_graph;
}

void AutoGradCellImpl::GradGraphByExpander(const GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(grad_param);
  // First handle parameters
  CreateParameterAdjoint(grad_param);
  bool ms_function_by_value = grad_param->is_ms_function_graph && grad_param->grad_by_value;

  const auto &order = TopoSort(grad_param->fg->output());
  for (const auto &node : order) {
    if (node == nullptr || !node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto prim = GetCNodePrimitive(cnode);
    if (prim == nullptr) {
      MS_LOG(EXCEPTION) << "Should be primitive, but: " << cnode->DebugString();
    }
    ad_param()->last_node_ = node;
    if (IsMonadPrim(prim, cnode, grad_param) || IsPrimitiveEquals(prim, prim::kPrimStopGradient)) {
      continue;
    }
    MS_LOG(DEBUG) << "Get cnode " << cnode->DebugString() << ", " << cnode->fullname_with_scope();
    if (IsPrimitiveEquals(prim, prim::kPrimMakeTuple)) {
      (void)BuildKNodeForMakeTuple(cnode);
      continue;
    } else if (IsPrimitiveEquals(prim, prim::kPrimTupleGetItem)) {
      (void)BuildKNodeForTupleGetItem(cnode);
      continue;
    }

    std::vector<AnfNodePtr> cnode_inputs{std::make_shared<ValueNode>(prim)};
    auto op_args = GetInputArgs(grad_param, cnode, &cnode_inputs);
    AnfNodePtr k_node = nullptr;
    if (IsPrimitiveEquals(prim, prim::kPrimMirror)) {
      k_node = ad_param()->anfnode_to_variable_adjoint_.at(cnode->input(kIndex1))->k_node();
    } else {
      auto c_k_node = ad_param()->tape_->NewCNode(cnode_inputs);
      c_k_node->set_abstract(cnode->abstract());
      // In ms function, copy forward graph cnode info to bprop graph
      if (ms_function_by_value && cnode->forward().first != nullptr) {
        auto new_v_node = NewValueNode(cnode->forward().first->value());
        new_v_node->set_abstract(cnode->forward().first->abstract());
        c_k_node->set_forward(new_v_node, cnode->forward().second);
        ad_param()->tape_->set_used_forward_nodes({c_k_node});
      }
      k_node = c_k_node;
    }
    MS_LOG(DEBUG) << "Build knode " << k_node->DebugString();
    // Set out
    auto out = pynative::PyNativeAlgo::Common::CreatOutputTensorValueByAbstract(cnode->abstract());
    (void)cnode_inputs.emplace_back(k_node);
    // Set dout
    AnfNodePtr dout = ZerosLike(ad_param()->tape_, out->ToAbstract());
    (void)cnode_inputs.emplace_back(dout);
    auto input_node = ad_param()->tape_->NewCNode(cnode_inputs);
    input_node->set_abstract(cnode->abstract());

    auto fn = std::make_shared<FunctionNode>(ad_param()->tape_, dout);
    auto variable_adjoint = std::make_shared<VariableAdjoint>(fn, out);
    variable_adjoint->set_k_node(k_node);

    std::vector<CNodePtr> outputs;
    auto ret = BpropExpander(&outputs, &ad_param()->users_).Run(input_node);
    if (!ret || outputs.empty()) {
      MS_LOG(DEBUG) << "Expander has no bprop of this node: " << input_node->DebugString();
      BuildCustomBpropCNode(input_node, prim, &outputs);
    }
    if (outputs.empty()) {
      MS_LOG(DEBUG) << "Build fake bprop for this node: " << input_node->DebugString();
      BuildFakeBpropCNode(input_node, &outputs);
      variable_adjoint->set_is_fake_bprop(true);
      variable_adjoint->set_fake_prim_name(prim->name());
    }
    // Create current op node din edge
    UpdateNextEdges(variable_adjoint, cnode, outputs, op_args);
    (void)ad_param()->anfnode_to_variable_adjoint_.insert(std::make_pair(node, variable_adjoint));
  }
}

void AutoGradCellImpl::CreateParameterAdjoint(const GradParamPtr &grad_param) {
  auto &graph_parameters = grad_param->fg->parameters();
  if (graph_parameters.size() != grad_param->op_args.size()) {
    MS_LOG(EXCEPTION) << "Parameters size " << graph_parameters.size() << " is not equal to graph input size "
                      << grad_param->op_args.size();
  }
  for (size_t i = 0; i < graph_parameters.size(); ++i) {
    MS_LOG(DEBUG) << "Get param " << graph_parameters[i]->DebugString();
    ParameterPtr param = ad_param()->tape_->add_parameter();
    auto tensor = pynative::PyNativeAlgo::Common::GetTensorFromParam(graph_parameters[i]);
    // Weight parameter
    if (tensor != nullptr) {
      const auto &param_info = tensor->param_info();
      MS_EXCEPTION_IF_NULL(param_info);
      const auto &param_name = param_info->name();
      param->set_name(param_name);
      param->debug_info()->set_name(param_name);
      param->set_default_param(tensor);
    }
    param->set_abstract(graph_parameters[i]->abstract());
    auto zeros_like_dout = ZerosLike(ad_param()->tape_, graph_parameters[i]->abstract());
    auto func_node = std::make_shared<FunctionNode>(ad_param()->tape_, zeros_like_dout);
    auto adjoint = std::make_shared<VariableAdjoint>(func_node, grad_param->op_args[i]);
    adjoint->set_k_node(param);
    (void)ad_param()->anfnode_to_variable_adjoint_.insert(std::make_pair(graph_parameters[i], adjoint));
  }
}

ValuePtrList AutoGradCellImpl::GetInputArgs(const GradParamPtr &grad_param, const CNodePtr &cnode,
                                            std::vector<AnfNodePtr> *cnode_inputs) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(cnode_inputs);
  ValuePtrList op_args;
  for (size_t i = 1; i < cnode->size(); ++i) {
    const auto &input_node = cnode->input(i);
    const auto it = ad_param()->anfnode_to_variable_adjoint_.find(input_node);
    if (it != ad_param()->anfnode_to_variable_adjoint_.end()) {
      (void)cnode_inputs->emplace_back(it->second->k_node());
      (void)op_args.emplace_back(it->second->out_value());
      continue;
    }
    if (input_node->isa<ValueNode>()) {
      auto v_node = input_node->cast<ValueNodePtr>();
      // In case of ms function forward graph and pynative bprop graph used same valuenode
      auto new_v_node = NewValueNode(v_node->value());
      new_v_node->set_abstract(v_node->abstract());
      (void)cnode_inputs->emplace_back(new_v_node);
      op_args.emplace_back(v_node->value());
    } else {
      MS_LOG(EXCEPTION) << "Get input node " << input_node->DebugString();
    }
  }
  return op_args;
}

void AutoGradCellImpl::UpdateOutputNodeOfTopCell(const AnfNodePtr &output_node, const ValuePtr &sens_out) {
  MS_EXCEPTION_IF_NULL(output_node);
  MS_EXCEPTION_IF_NULL(sens_out);
  MS_LOG(DEBUG) << "Real output node of top cell is " << output_node->DebugString();
  ad_param()->last_node_ = output_node;
  sens_value_ = sens_out;
}

FuncGraphPtr AutoGradCellImpl::Finish(const AnfNodePtrList &weights, const std::vector<size_t> &grad_position,
                                      const GradAttr &grad_attr) {
  // Set sens node and weights node
  SetSensAndWeights(weights, grad_attr.has_sens);

  // BackPropagate sensitivity, except when the last node is a valuenode which may be obtained by constant folding;
  if (!ad_param()->last_node_->isa<ValueNode>() && !ad_param()->last_node_->isa<Parameter>()) {
    (void)BackPropagate();
  }
  SetOutput(weights, grad_position, grad_attr);
  // Replace Parameter of primal funcgraph with parameter of ad_param()->tape_;
  ReplacePrimalParameter(weights, grad_attr.has_sens);
  pynative::PyNativeAlgo::Common::DumpGraphIR("before_final_opt.ir", ad_param()->tape_);
  return ad_param()->tape_;
}

CNodePtr AutoGradCellImpl::ConstructBpropGraphInput(const GradParamPtr &grad_param, const AnfNodePtr &dout,
                                                    const VariableAdjointPtr &variable_adjoint, bool is_custom_prim) {
  MS_EXCEPTION_IF_NULL(grad_param);
  std::vector<AnfNodePtr> node_list;
  (void)node_list.emplace_back(grad_param->cnode->input(0));
  if (grad_param->grad_by_value || is_custom_prim) {
    for (size_t i = 0; i < grad_param->op_args.size(); ++i) {
      auto node = grad_param->cnode->input(i + 1);
      if (node->isa<Parameter>()) {
        (void)node_list.emplace_back(node);
        continue;
      }
      // TupleGetItem need replace
      if (node->has_user_data(kParamterIsSequence)) {
        (void)node_list.emplace_back(GetTupleItemNodeInput(ad_param()->tape_, node));
        need_do_manager_replace_ = true;
        continue;
      }
      auto v_node = NewValueNode(grad_param->op_args[i]);
      // Node abstract obj may free, so v node abstract will be not correct
      v_node->set_abstract(node->abstract()->Clone());
      (void)node_list.emplace_back(v_node);
    }
    // Set out
    auto out_node = NewValueNode(grad_param->out);
    out_node->set_abstract(grad_param->cnode->abstract());
    (void)node_list.emplace_back(out_node);
  } else {
    // Input is a Parameter or cnode, not a value node
    BuildKNodeListFromPrimalCNode(grad_param->cnode, grad_param->op_args, &node_list);
    // Set out
    MS_EXCEPTION_IF_NULL(variable_adjoint);
    (void)node_list.emplace_back(variable_adjoint->k_node());
  }
  // Set dout
  (void)node_list.emplace_back(dout);
  auto input_node = ad_param()->tape_->NewCNode(node_list);
  input_node->set_abstract(grad_param->cnode->abstract());
  return input_node;
}

void AutoGradCellImpl::BuildKNodeListFromPrimalCNode(const CNodePtr &cnode, const ValuePtrList &op_args,
                                                     std::vector<AnfNodePtr> *const node_list) {
  MS_EXCEPTION_IF_NULL(cnode);
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    if (cnode->input(i)->isa<CNode>()) {
      const auto input_adjoint_iter = ad_param()->anfnode_to_variable_adjoint_.find(cnode->input(i));
      if (input_adjoint_iter == ad_param()->anfnode_to_variable_adjoint_.end()) {
        MS_LOG(EXCEPTION) << "Cannot find input in adjoint map, inp: " << cnode->input(i)->DebugString();
      }
      MS_EXCEPTION_IF_NULL(input_adjoint_iter->second->k_node());
      (void)node_list->emplace_back(input_adjoint_iter->second->k_node());
    } else {
      (void)node_list->emplace_back(cnode->input(i));
    }
    MS_LOG(DEBUG) << "Get knode for node " << cnode->input(i)->DebugString();
  }
}

AnfNodePtr AutoGradCellImpl::BuildKNode(const GradParamPtr &grad_param, bool from_single_op) {
  MS_EXCEPTION_IF_NULL(grad_param);
  AnfNodePtrList node_list;
  for (size_t i = 0; i < grad_param->cnode->inputs().size(); ++i) {
    (void)node_list.emplace_back(BuildKNodeForCNodeInput(grad_param->cnode->input(i)));
  }
  auto k_node = ad_param()->tape_->NewCNode(node_list);
  k_node->set_abstract(grad_param->cnode->abstract());
  if (from_single_op && grad_param->out_used_in_bporp_graph) {
    auto v_node = NewValueNode(grad_param->out);
    v_node->set_abstract(grad_param->cnode->abstract());
    k_node->set_forward(v_node, "");
    ad_param()->tape_->set_used_forward_nodes({k_node});
  }
  MS_LOG(DEBUG) << "Build knode " << k_node->DebugString();
  return k_node;
}

AnfNodePtr AutoGradCellImpl::BuildKNodeForCNodeInput(const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(input_node);
  if (input_node->isa<CNode>()) {
    const auto input_adjoint_iter = ad_param()->anfnode_to_variable_adjoint_.find(input_node);
    if (input_adjoint_iter == ad_param()->anfnode_to_variable_adjoint_.end()) {
      if (IsPrimitiveCNode(input_node, prim::kPrimMakeTuple)) {
        return BuildKNodeForMakeTuple(input_node);
      } else if (IsPrimitiveCNode(input_node, prim::kPrimTupleGetItem)) {
        return BuildKNodeForTupleGetItem(input_node);
      }
      MS_LOG(EXCEPTION) << "Can not find input in adjoint map, inp: " << input_node->DebugString();
    }
    return input_adjoint_iter->second->k_node();
  } else {
    // Tuple sens will come in
    if (input_node->isa<Parameter>()) {
      const auto input_adjoint_iter = ad_param()->anfnode_to_variable_adjoint_.find(input_node);
      if (input_adjoint_iter != ad_param()->anfnode_to_variable_adjoint_.end() &&
          input_adjoint_iter->second->k_node() != nullptr) {
        return input_adjoint_iter->second->k_node();
      }
    }
    return input_node;
  }
}

AnfNodePtr AutoGradCellImpl::BuildKNodeForMakeTuple(const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(input_node);
  MS_LOG(DEBUG) << "Build knode for MakeTuple " << input_node->DebugString();
  const auto &cnode = input_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimMakeTuple)};
  ValuePtrList op_args;
  for (size_t i = 1; i < cnode->size(); ++i) {
    (void)inputs.emplace_back(BuildKNodeForCNodeInput(cnode->input(i)));
    if (cnode->input(i)->isa<CNode>() || cnode->input(i)->isa<Parameter>()) {
      const auto input_adjoint_iter = ad_param()->anfnode_to_variable_adjoint_.find(cnode->input(i));
      if (input_adjoint_iter == ad_param()->anfnode_to_variable_adjoint_.end()) {
        MS_LOG(EXCEPTION) << "Cannot find input in adjoint map, inp: " << cnode->input(i)->DebugString();
      }
      (void)op_args.emplace_back(input_adjoint_iter->second->out_value());
    } else {
      auto value_node = cnode->input(i)->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      (void)op_args.emplace_back(value_node->value());
    }
  }
  auto out_value = MakeValue(op_args);
  AnfNodePtr dout = BuildSpecialLikeValue(ad_param()->tape_, out_value, SpecialType::kZerosLikeType);
  auto fn = std::make_shared<FunctionNode>(ad_param()->tape_, dout);
  auto variable_adjoint = std::make_shared<VariableAdjoint>(fn, out_value);
  auto k_node = ad_param()->tape_->NewCNode(inputs);
  k_node->set_abstract(input_node->abstract());
  variable_adjoint->set_k_node(k_node);
  (void)ad_param()->anfnode_to_variable_adjoint_.insert(std::make_pair(input_node, variable_adjoint));

  // Create dout for maketuple
  std::vector<CNodePtr> make_tuple_dout;
  for (size_t i = 1; i < cnode->size(); ++i) {
    auto d =
      ad_param()->tape_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), dout, NewValueNode(SizeToLong(i - 1))});
    d->set_abstract(cnode->input(i)->abstract());
    (void)make_tuple_dout.emplace_back(d);
    AddUser(dout, d, 1);
  }
  UpdateNextEdges(variable_adjoint, cnode, make_tuple_dout, op_args);
  return k_node;
}

AnfNodePtr AutoGradCellImpl::BuildKNodeForTupleGetItem(const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(input_node);
  MS_LOG(DEBUG) << "Build knode for TupleGetItem " << input_node->DebugString();
  const auto &tuple_item_cnode = input_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(tuple_item_cnode);
  // Find make tuple or sens(tuple) node for get out value
  const auto input_adjoint_iter = ad_param()->anfnode_to_variable_adjoint_.find(tuple_item_cnode->input(1));
  if (input_adjoint_iter == ad_param()->anfnode_to_variable_adjoint_.end()) {
    MS_LOG(EXCEPTION) << "Cannot find input in adjoint map, inp: " << tuple_item_cnode->input(1)->DebugString();
  }
  const auto &v_tuple = input_adjoint_iter->second->out_value()->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(v_tuple);
  auto index_value = GetValueNode<Int64ImmPtr>(tuple_item_cnode->input(2));
  auto index_value_int = LongToSize(index_value->value());
  auto out_value = (*v_tuple)[index_value_int];
  MS_EXCEPTION_IF_NULL(out_value);
  AnfNodePtr dout = BuildSpecialLikeValue(ad_param()->tape_, out_value, SpecialType::kZerosLikeType);
  auto fn = std::make_shared<FunctionNode>(ad_param()->tape_, dout);
  auto variable_adjoint = std::make_shared<VariableAdjoint>(fn, out_value);

  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimTupleGetItem)};
  // Get make tuple knode
  (void)inputs.emplace_back(BuildKNodeForCNodeInput(tuple_item_cnode->input(1)));
  // Get index knode
  (void)inputs.emplace_back(BuildKNodeForCNodeInput(tuple_item_cnode->input(2)));
  auto k_node = ad_param()->tape_->NewCNode(inputs);
  k_node->set_abstract(input_node->abstract());
  variable_adjoint->set_k_node(k_node);
  (void)ad_param()->anfnode_to_variable_adjoint_.insert(std::make_pair(input_node, variable_adjoint));

  // Create dout for tuplegetitem
  std::vector<AnfNodePtr> tuple_getitem_dout{NewValueNode(prim::kPrimMakeTuple)};
  for (size_t i = 0; i < v_tuple->size(); ++i) {
    const auto &v = v_tuple->value()[i];
    if (i == index_value_int) {
      (void)tuple_getitem_dout.emplace_back(dout);
    } else {
      (void)tuple_getitem_dout.emplace_back(BuildSpecialLikeValue(ad_param()->tape_, v, SpecialType::kZerosLikeType));
    }
  }
  CNodePtr tuple_getitem_dout_value = ad_param()->tape_->NewCNode(tuple_getitem_dout);
  tuple_getitem_dout_value->set_abstract(tuple_item_cnode->input(1)->abstract());
  CNodePtr index_dout_value =
    BuildSpecialLikeValue(ad_param()->tape_, index_value, SpecialType::kZerosLikeType)->cast<CNodePtr>();
  UpdateNextEdges(variable_adjoint, tuple_item_cnode, {tuple_getitem_dout_value, index_dout_value},
                  {v_tuple, index_value});
  AddUser(dout, tuple_getitem_dout_value, index_value_int + 1);
  return k_node;
}

bool GradPynativeOp(const AutoGradCellImplPtr &k_cell, const GradParamPtr &grad_param) {
  return k_cell->KPynativeOp(grad_param);
}

void AutoGradCellImpl::UpdateNextEdges(const VariableAdjointPtr &variable, const CNodePtr &cnode,
                                       const std::vector<CNodePtr> &dins, const ValuePtrList &op_args) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (dins.size() != op_args.size()) {
    MS_LOG(EXCEPTION) << "The size of dins is not same as op_args, cnode: " << cnode->DebugString();
  }
  const auto &fn = variable->fn();
  for (size_t i = 0; i < op_args.size(); ++i) {
    const auto &node = cnode->input(i + 1);
    const auto &din = dins[i];
    MS_LOG(DEBUG) << "Node " << node->DebugString() << ", din " << din->DebugString();
    UpdateNextEdge(fn, node, din, op_args[i]);
  }
  if (fn->next_edges().empty()) {
    variable->set_is_need_grad(false);
  }
}

void AutoGradCellImpl::UpdateNextEdge(const FunctionNodePtr &fn, const AnfNodePtr &input_node, const AnfNodePtr &din,
                                      const ValuePtr &input_arg) {
  MS_EXCEPTION_IF_NULL(input_node);
  MS_EXCEPTION_IF_NULL(din);
  MS_EXCEPTION_IF_NULL(input_arg);
  const auto it = ad_param()->anfnode_to_variable_adjoint_.find(input_node);
  if (it != ad_param()->anfnode_to_variable_adjoint_.end()) {
    if (!it->second->is_need_grad()) {
      return;
    }
    auto real_din = HandleRealToComplex(input_node, din, fn->tape());
    auto new_din = TraceShape(fn, it->second->out_value(), input_arg, real_din);
    fn->AddNextEdge(input_node, new_din);
  } else if (input_node->isa<CNode>()) {
    const auto &cnode = input_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (IsPrimitiveCNode(cnode, prim::kPrimStopGradient) || IsPrimitiveCNode(cnode, prim::kPrimUpdateState)) {
      return;
    }
    if (IsPrimitiveCNode(cnode, prim::kPrimMakeTuple)) {
      if (!input_arg->isa<ValueSequence>()) {
        MS_LOG(EXCEPTION) << "op_arg type is not valuesequence";
      }
      auto value_seq = input_arg->cast<ValueSequencePtr>();
      for (size_t i = 0; i < value_seq->value().size(); ++i) {
        auto input = cnode->input(i + 1);
        auto sub_value = value_seq->value()[i];
        CNodePtr new_din =
          ad_param()->tape_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), din, NewValueNode(SizeToLong(i))});
        new_din->set_abstract(input->abstract());
        if (din == fn->fake_dout()) {
          // The new_din's index input is fn->fake_dout()
          AddUser(fn->fake_dout(), new_din, 1);
        }
        // Add next edge to fn
        UpdateNextEdge(fn, input, new_din, sub_value);
      }
    } else if (IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem)) {
      auto src_node = cnode->input(1);
      auto index_value = GetValueNode<Int64ImmPtr>(cnode->input(2));
      if (index_value == nullptr) {
        MS_LOG(EXCEPTION) << "CNode input 2 should be a Int64Imm, CNode: " << cnode->DebugString();
      }
      UpdateNextEdge(fn, src_node, din, input_arg);
    } else {
      MS_LOG(EXCEPTION) << "Cnode should be tuplegetitem or maketuple " << cnode->DebugString();
    }
  } else if (input_node->isa<Parameter>()) {
    auto param = input_node->cast<ParameterPtr>();
    auto tensor = param->default_param();
    MS_EXCEPTION_IF_NULL(tensor);
    AddParameterNode(param, tensor);
    UpdateNextEdge(fn, input_node, din, input_arg);
  } else {
    MS_LOG(DEBUG) << "It is not a cnode or parameter: " << input_node->DebugString();
    return;
  }
}

void AutoGradCellImpl::BuildForwardLastNode() {
  MS_EXCEPTION_IF_NULL(ad_param()->last_node_);
  if (ad_param()->last_node_->isa<ValueNode>() ||
      ad_param()->anfnode_to_variable_adjoint_.find(ad_param()->last_node_) !=
        ad_param()->anfnode_to_variable_adjoint_.end()) {
    return;
  }
  MS_LOG(DEBUG) << "Process last node info " << ad_param()->last_node_->DebugString();
  auto zeros_like_node = BuildZerosLikeNode(ad_param()->tape_, sens_value_);
  auto fn = std::make_shared<FunctionNode>(ad_param()->tape_, zeros_like_node);
  // If last_node is a maketuple or tuplegetitem, need update next edges,
  // if last_node is parameter, not need to update next edges.
  if (ad_param()->last_node_->isa<CNode>()) {
    UpdateNextEdge(fn, ad_param()->last_node_, zeros_like_node, sens_value_);
  }
  auto input_adjoint = std::make_shared<VariableAdjoint>(fn, sens_value_);
  (void)ad_param()->anfnode_to_variable_adjoint_.insert(std::make_pair(ad_param()->last_node_, input_adjoint));
}

void AutoGradCellImpl::AddParameterNode(const AnfNodePtr &parameter, const ValuePtr &tensor) {
  MS_EXCEPTION_IF_NULL(parameter);
  MS_EXCEPTION_IF_NULL(tensor);
  auto zeros_like_dout = ZerosLike(ad_param()->tape_, tensor->ToAbstract());
  auto func_node = std::make_shared<FunctionNode>(ad_param()->tape_, zeros_like_dout);
  auto input_adjoint = std::make_shared<VariableAdjoint>(func_node, tensor);
  (void)ad_param()->anfnode_to_variable_adjoint_.insert(std::make_pair(parameter, input_adjoint));
  (void)weights_used_in_graph_.emplace_back(parameter);
}

AnfNodePtr AutoGradCellImpl::TraceShape(const FunctionNodePtr &fn, const ValuePtr &out_value, const ValuePtr &input_arg,
                                        const AnfNodePtr &din) {
  MS_EXCEPTION_IF_NULL(out_value);
  MS_EXCEPTION_IF_NULL(input_arg);
  MS_EXCEPTION_IF_NULL(din);
  const auto &out_value_id = pynative::PyNativeAlgo::Common::GetIdByValue(out_value);
  const auto &input_arg_id = pynative::PyNativeAlgo::Common::GetIdByValue(input_arg);
  // The node corresponding output tensor is the same as the currently used tensor
  if (out_value_id == input_arg_id) {
    return din;
  } else if (out_value->isa<tensor::Tensor>()) {
    // out_value is be used, may be it is one of multiple output
    return BuildZerosLikeNode(ad_param()->tape_, out_value);
  } else if (out_value->isa<ValueSequence>()) {
    // Input is scalar tuple, list
    if (out_value_id.find('T') == std::string::npos) {
      return BuildZerosLikeNode(ad_param()->tape_, out_value);
    }
    // The corresponding output of node is ValueSequence, but used one of it
    std::vector<AnfNodePtr> inputs;
    if (out_value->isa<ValueTuple>()) {
      (void)inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    } else {
      (void)inputs.emplace_back(NewValueNode(prim::kPrimMakeList));
    }
    auto value_seq = out_value->cast<ValueSequencePtr>();
    int index = -1;
    for (const auto &value : value_seq->value()) {
      // Find the value's din, if value equal to sub_value, means value be used, is it will get din; Otherwise value's
      // din is zero , which set by second branch condition above
      auto new_din = TraceShape(fn, value, input_arg, din);
      (void)inputs.emplace_back(new_din);

      // if exist din == fake_dout, we record it in user vector
      if (din == fn->fake_dout() && new_din == din) {
        index = static_cast<int>(inputs.size()) - 1;
      }
    }
    auto new_din = ad_param()->tape_->NewCNode(inputs);
    new_din->set_abstract(PyNativeAlgo::Common::SetAbstractValueToAnyValue(out_value->ToAbstract()));
    if (index != -1) {
      AddUser(fn->fake_dout(), new_din, index);
    }
    return new_din;
  }
  MS_LOG(DEBUG) << "Get non tensor input " << out_value->ToString();
  return BuildZerosLikeNode(ad_param()->tape_, out_value);
}

void AutoGradCellImpl::BuildBPropCutCNode(const CNodePtr &cnode, const PrimitivePtr &prim,
                                          std::vector<CNodePtr> *outputs) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_py = prim->cast<PrimitivePyPtr>();
  MS_EXCEPTION_IF_NULL(prim_py);
  auto bprop_cut = std::make_shared<PrimitivePy>("bprop_cut");
  bprop_cut->CopyHookFunction(prim_py);
  prim_py->AddBpropCutPrim(bprop_cut);
  if (prim->HasAttr("cell_id")) {
    auto cell_id = GetValue<std::string>(prim->GetAttr("cell_id"));
    if (!cell_id.empty()) {
      (void)bprop_cut->AddAttr("cell_hook", MakeValue(true));
      (void)bprop_cut->AddAttr("cell_id", MakeValue(cell_id));
    }
  }
  if (prim->HasAttr("custom_op_bprop")) {
    (void)bprop_cut->AddAttr("custom_op_bprop", MakeValue(true));
  }
  // Create gradient outputs cnode
  std::vector<AnfNodePtr> inputs{NewValueNode(bprop_cut)};
  // Get input, get output, get dout
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    (void)inputs.emplace_back(cnode->input(i));
  }
  auto bprop_cut_cnode = ad_param()->tape_->NewCNode(inputs);

  size_t input_num = cnode->size() - 2;
  AbstractBasePtrList abs_list;
  for (size_t i = 1; i < cnode->size(); ++i) {
    // bprop_cut_cnode ith input used cnode->input(i)
    AddUser(cnode->input(i), bprop_cut_cnode, i);
    if (i < input_num) {
      auto din = ad_param()->tape_->NewCNode(
        {NewValueNode(prim::kPrimTupleGetItem), bprop_cut_cnode, NewValueNode(static_cast<int64_t>(i - 1))});
      MS_EXCEPTION_IF_NULL(cnode->input(i)->abstract());
      din->set_abstract(cnode->input(i)->abstract());
      abs_list.emplace_back(cnode->input(i)->abstract());
      (void)outputs->emplace_back(din);
    }
  }
  bprop_cut_cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abs_list));
}

void AutoGradCellImpl::BuildCustomBpropCNode(const CNodePtr &cnode, const PrimitivePtr &prim,
                                             std::vector<CNodePtr> *outputs) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_LOG(DEBUG) << "Try build custom bprop: " << prim->name();
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
    auto prim_py = prim->cast<PrimitivePyPtr>();
    MS_EXCEPTION_IF_NULL(prim_py);
    prim_py->AddBackwardHookFn(0, fn);
    prim_py->AddAttr("custom_op_bprop", MakeValue(True));
    prim_py->AddAttr("custom_op_name", MakeValue(cnode->fullname_with_scope()));
  }
  BuildBPropCutCNode(cnode, prim, outputs);
}

void AutoGradCellImpl::BuildFakeBpropCNode(const CNodePtr &cnode, std::vector<CNodePtr> *outputs) {
  auto prim = GetCNodePrimitive(cnode);
  if (prim == nullptr) {
    MS_LOG(EXCEPTION) << "Should be primitive, but: " << cnode->DebugString();
  }
  size_t dout_index = cnode->size() - 1;
  const auto &dout = cnode->input(dout_index);
  const auto &dout_cnode = dout->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(dout_cnode);
  // Size is same as op_arg size
  size_t input_size = cnode->size() - 2;
  for (size_t i = 1; i < input_size; ++i) {
    (void)outputs->emplace_back(dout_cnode);
  }
}

void AutoGradCellImpl::SetSensAndWeights(const AnfNodePtrList &weights, bool has_sens_arg) {
  BuildForwardLastNode();

  // Update dout for dout
  MS_EXCEPTION_IF_NULL(ad_param()->last_node_);
  if (ad_param()->anfnode_to_variable_adjoint_.find(ad_param()->last_node_) !=
      ad_param()->anfnode_to_variable_adjoint_.end()) {
    const auto &variable = ad_param()->anfnode_to_variable_adjoint_.at(ad_param()->last_node_);
    if (has_sens_arg) {
      ParameterPtr sens_param = ad_param()->tape_->add_parameter();
      sens_param->debug_info()->set_name("sens");
      sens_param->set_abstract(ad_param()->last_node_->abstract());
      variable->fn()->UpdateAccumulativeDout(sens_param);
    } else {
      variable->fn()->UpdateAccumulativeDout(BuildOnesLikeNode(ad_param()->tape_, sens_value_));
    }
  }

  // Add weights parameter
  need_grad_weights_.clear();
  for (const auto &weight : weights) {
    TraceGuard trace_guard(std::make_shared<TraceCopy>(weight->debug_info()));
    auto t = pynative::PyNativeAlgo::Common::GetTensorFromParam(weight);
    (void)need_grad_weights_.emplace(t->id());
    auto p = ad_param()->tape_->add_parameter();
    auto param = weight->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
    p->set_name(param->name());
    p->set_default_param(t);
    MS_EXCEPTION_IF_NULL(weight->abstract());
    p->set_abstract(weight->abstract());
  }
}

OrderedMap<AnfNodePtr, VariableAdjointPtr>::reverse_iterator AutoGradCellImpl::GetLastNodeReverseIter() {
  for (auto iter = ad_param()->anfnode_to_variable_adjoint_.rbegin();
       iter != ad_param()->anfnode_to_variable_adjoint_.rend(); ++iter) {
    if (!iter->first->isa<CNode>()) {
      continue;
    }
    if (iter->first->cast<CNodePtr>() == ad_param()->last_node_) {
      auto &variable = ad_param()->anfnode_to_variable_adjoint_[ad_param()->last_node_];
      variable->set_is_need_propagate(true);
      return iter;
    }
  }
  return ad_param()->anfnode_to_variable_adjoint_.rend();
}

void AutoGradCellImpl::BackPropagate() {
  const auto &last_node_reverse_iter = GetLastNodeReverseIter();
  bool has_primc = false;
  for (auto iter = last_node_reverse_iter; iter != ad_param()->anfnode_to_variable_adjoint_.rend(); ++iter) {
    MS_LOG(DEBUG) << "BackPropagate cnode: " << iter->first->DebugString();
    const auto &variable = iter->second;
    if (!variable->is_need_propagate() || !variable->is_need_grad()) {
      MS_LOG(DEBUG) << "No need grad";
      continue;
    }
    if (variable->is_fake_bprop()) {
      MS_LOG(EXCEPTION) << "Illegal primitive " << variable->fake_prim_name() << "'s bprop not defined";
    }
    if (!has_primc && iter->first->isa<CNode>() && GetCNodePrimitive(iter->first) != nullptr) {
      has_primc = true;
    }
    const auto &fn = variable->fn();
    // If zeroslike not used in funcgraph, we need replace the zeroslike placeholder with real zeroslike value.
    if (MS_UNLIKELY(IsZerosLikeNode(fn->accumulate_dout()))) {
      fn->set_accumulate_dout(BuildZerosLikeNode(fn->tape(), variable->out_value()));
    }
    // Replace real dout to fake dout, update replace result to eliminate tuplegetitem
    // when accumulate_dout is tuplegetitem
    Replace(fn->fake_dout(), fn->accumulate_dout(), true);
    // replace edges which exist fake dout
    fn->ReplaceEdges();

    const auto &next_edges = fn->next_edges();
    for (const auto &next_edge : next_edges) {
      const auto &node = next_edge.first;
      const auto &din = next_edge.second;
      if (ad_param()->anfnode_to_variable_adjoint_.find(node) == ad_param()->anfnode_to_variable_adjoint_.end()) {
        MS_LOG(EXCEPTION) << "Current node not find corresponding node";
      }
      auto last_variable = ad_param()->anfnode_to_variable_adjoint_[node];
      last_variable->fn()->UpdateAccumulativeDout(din);
      last_variable->set_is_need_propagate(true);
    }
  }
  ad_param()->tape_->set_flag(kPrimCPrimPyMixed, has_primc && need_do_manager_replace_);
}

AnfNodePtr AutoGradCellImpl::GetGradNodeByIndex(const AnfNodePtr &grad_node) {
  MS_EXCEPTION_IF_NULL(grad_node);
  const auto &input_adjoint_iter = ad_param()->anfnode_to_variable_adjoint_.find(grad_node);
  if (input_adjoint_iter == ad_param()->anfnode_to_variable_adjoint_.end()) {
    // If weight is not used in the forward network, just return zeros_like() as dout.
    if (grad_node->isa<Parameter>()) {
      MS_LOG(INFO) << "Weight does not participate in forward calculation, weight: " << grad_node->DebugString();
      auto w = grad_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(w);
      auto default_param = w->default_param();
      MS_EXCEPTION_IF_NULL(default_param);
      return BuildZerosLikeNode(ad_param()->tape_, default_param);
    }
    // If input is not used in the forward network, just return zeros_like() as dout.
    MS_LOG(EXCEPTION) << "Input does not participate in forward calculation, input: " << grad_node->DebugString();
    return nullptr;
  }
  return input_adjoint_iter->second->RealDout();
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
      if (index >= cell_inputs_.size()) {
        MS_LOG(EXCEPTION) << "Position index " << index << " is exceed input size.";
      }
      // Tuple, List, scalar will be ignore
      if (!IsValidTensorInput(cell_inputs_[index]->abstract())) {
        MS_LOG(DEBUG) << "Get input node is not tensor " << cell_inputs_[index]->DebugString() << ", abs "
                      << cell_inputs_[index]->abstract()->ToString();
        continue;
      }
      auto intput_grad_node = GetGradNodeByIndex(cell_inputs_[index]);
      MS_EXCEPTION_IF_NULL(intput_grad_node);
      (void)inputs_grad_list.emplace_back(intput_grad_node);
      (void)inputs_grad_spec.emplace_back(intput_grad_node->abstract());
    }
    constexpr size_t single_pos_size = 1;
    if (get_by_position && inputs_grad_spec.size() == single_pos_size) {
      // First elem is prim
      return inputs_grad_list[single_pos_size];
    }
  }
  auto input_grad_ret = ad_param()->tape_->NewCNode(inputs_grad_list);
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
      auto grad_node = GetGradNodeByIndex(weights[index]);
      MS_EXCEPTION_IF_NULL(grad_node);
      (void)weights_grad_list.emplace_back(grad_node);
      (void)weights_grad_spec.emplace_back(grad_node->abstract());
    }
    auto weight_grad_ret = ad_param()->tape_->NewCNode(weights_grad_list);
    weight_grad_ret->set_abstract(std::make_shared<abstract::AbstractTuple>(weights_grad_spec));
    return weight_grad_ret;
  } else {
    return GetGradNodeByIndex(weights[0]);
  }
}

void AutoGradCellImpl::SetOutput(const AnfNodePtrList &weights, const std::vector<size_t> &grad_position,
                                 const GradAttr &grad_attr) {
  auto inputs_grad_ret = GetInputGrad(grad_attr.grad_all_inputs, grad_attr.get_by_position, grad_position);
  auto weights_grad_ret = GetWeightGrad(grad_attr.grad_weights, weights, grad_attr.weight_param_is_tuple);
  // Gradients wrt inputs and weights.
  if (inputs_grad_ret != nullptr && weights_grad_ret != nullptr) {
    if (IsOutputBothEmpty(inputs_grad_ret, weights_grad_ret)) {
      auto tape_output = GenerateEmptyTupleValue();
      ad_param()->tape_->set_output(tape_output);
    } else {
      auto tape_output =
        ad_param()->tape_->NewCNode({NewValueNode(prim::kPrimMakeTuple), inputs_grad_ret, weights_grad_ret});
      tape_output->set_abstract(std::make_shared<abstract::AbstractTuple>(
        abstract::AbstractBasePtrList{inputs_grad_ret->abstract(), weights_grad_ret->abstract()}));
      ad_param()->tape_->set_output(tape_output);
    }
    return;
  }
  // Gradients wrt inputs.
  if (inputs_grad_ret != nullptr) {
    ad_param()->tape_->set_output(inputs_grad_ret);
    return;
  }
  // Gradients wrt weights.
  if (weights_grad_ret != nullptr) {
    ad_param()->tape_->set_output(weights_grad_ret);
    return;
  }
  // grad_all_inputs, grad_weights and get_by_position are all false.
  AnfNodePtr tape_output = nullptr;
  if (cell_inputs_.empty()) {
    // If no input nodes, return empty tuple.
    tape_output = ad_param()->tape_->NewCNode({NewValueNode(prim::kPrimMakeTuple)});
    abstract::AbstractBasePtrList abs{};
    tape_output->set_abstract(std::make_shared<abstract::AbstractTuple>(abs));
  } else {
    // If there are input nodes, return gradient of first input node.
    // Tuple, List, scalar will be ignore
    if (IsValidTensorInput(cell_inputs_[0]->abstract())) {
      tape_output = GetGradNodeByIndex(cell_inputs_[0]);
    } else {
      MS_LOG(DEBUG) << "Get first input node is not tensor " << cell_inputs_[0]->DebugString() << ", abs "
                    << cell_inputs_[0]->abstract()->ToString();
      tape_output = NewValueNode(kNull);
      tape_output->set_abstract(nullptr);
    }
  }
  ad_param()->tape_->set_output(tape_output);
}

void AutoGradCellImpl::AddUser(const AnfNodePtr &node, const CNodePtr &user, size_t index) {
  MS_EXCEPTION_IF_NULL(ad_param_);
  if (ad_param()->users_.find(node) == ad_param()->users_.end()) {
    ad_param()->users_[node] = {};
  }
  (void)ad_param()->users_[node].emplace_back(user, index);
}

void AutoGradCellImpl::Replace(const AnfNodePtr &old_node, const AnfNodePtr &new_node, bool need_update) {
  MS_EXCEPTION_IF_NULL(ad_param_);
  if (ad_param()->users_.find(old_node) == ad_param()->users_.end()) {
    return;
  }
  auto &old_node_users = ad_param()->users_[old_node];
  for (const auto &pair_node : old_node_users) {
    auto cnode = pair_node.first.lock();
    if (cnode == nullptr) {
      continue;
    }
    size_t index = pair_node.second;
    if (index >= cnode->size()) {
      MS_LOG(EXCEPTION) << "exception for index:" << index << "greater than cnode size:" << cnode->size();
    }
    cnode->set_input(index, new_node);
    if (need_update) {
      AddUser(new_node, cnode, index);
    }
  }
}

void AutoGradCellImpl::ElimateTupleGetItem() {
  for (auto &user : ad_param()->users_) {
    auto old_node = user.first;
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

void AutoGradCellImpl::DoParameterReplaceByManager(const AnfNodePtrList &weights, bool has_sens_arg) {
  const auto &parameters = ad_param()->tape_->parameters();
  auto cell_inputs_size = cell_inputs_.size();
  auto mng = MakeManager({ad_param()->tape_}, false);
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
  for (auto &weight : weights_used_in_graph_) {
    auto t = pynative::PyNativeAlgo::Common::GetTensorFromParam(weight);
    if (need_grad_weights_.find(t->id()) == need_grad_weights_.end()) {
      MS_LOG(DEBUG) << "Convert " << weight->DebugString() << " to value node by manager.";
      auto value_node = NewValueNode(t);
      MS_EXCEPTION_IF_NULL(weight->abstract());
      value_node->set_abstract(weight->abstract());
      (void)tr.Replace(weight, value_node);
    }
  }
  tr.Commit();
}

void AutoGradCellImpl::DoParameterReplaceByUser(const AnfNodePtrList &weights, bool has_sens_arg) {
  const auto &parameters = ad_param()->tape_->parameters();
  auto cell_inputs_size = cell_inputs_.size();
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
  for (auto &weight : weights_used_in_graph_) {
    auto t = pynative::PyNativeAlgo::Common::GetTensorFromParam(weight);
    if (need_grad_weights_.find(t->id()) == need_grad_weights_.end()) {
      MS_LOG(DEBUG) << "Convert " << weight->DebugString() << " to value node by user.";
      auto value_node = NewValueNode(t);
      MS_EXCEPTION_IF_NULL(weight->abstract());
      value_node->set_abstract(weight->abstract());
      Replace(weight, value_node);
    }
  }
}

void AutoGradCellImpl::ReplacePrimalParameter(const AnfNodePtrList &weights, bool has_sens_arg) {
  pynative::PyNativeAlgo::Common::DumpGraphIR("replace_param.ir", ad_param()->tape_);
  if (need_do_manager_replace_) {
    MS_LOG(DEBUG) << "Do parameter replace by manager.";
    DoParameterReplaceByManager(weights, has_sens_arg);
    need_do_manager_replace_ = false;
  } else {
    MS_LOG(DEBUG) << "Do parameter replace by user.";
    DoParameterReplaceByUser(weights, has_sens_arg);
  }
  ElimateTupleGetItem();
}

AutoGradCellImplPtr GradPynativeCellBegin(const AnfNodePtrList &cell_inputs,
                                          const std::vector<ValuePtr> &input_param_values,
                                          const AbstractBasePtrList &abs_list, size_t op_num_in_bprop_graph) {
  auto abstract_are_set = std::all_of(cell_inputs.cbegin(), cell_inputs.cend(),
                                      [](const AnfNodePtr &node) { return node->abstract() != nullptr; });
  if (!abstract_are_set) {
    MS_LOG(EXCEPTION) << "Not all abstract_value in cell_inputs are set";
  }
  if (cell_inputs.size() != input_param_values.size()) {
    MS_LOG(EXCEPTION) << "The size of cell inputs " << cell_inputs.size()
                      << " is not equal to the size of input parameter values " << input_param_values.size();
  }
  return std::make_shared<AutoGradCellImpl>(cell_inputs, input_param_values, abs_list, op_num_in_bprop_graph);
}

FuncGraphPtr GradPynativeCellEnd(const AutoGradCellImplPtr &auto_grad_cell, const AnfNodePtrList &weights,
                                 const std::vector<size_t> &grad_position, const GradAttr &grad_attr) {
  return auto_grad_cell->Finish(weights, grad_position, grad_attr);
}

void ClearPyNativeAutoGradStaticRes() { pass_grad_graph_.clear(); }
}  // namespace autograd
}  // namespace pynative
}  // namespace mindspore
