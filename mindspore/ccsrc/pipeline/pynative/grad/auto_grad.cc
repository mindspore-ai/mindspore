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
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include "mindspore/core/ops/core_ops.h"
#include "ir/anf.h"
#include "ir/func_graph_cloner.h"
#include "frontend/optimizer/ad/dfunctor.h"
#include "frontend/operator/composite/composite.h"
#include "utils/info.h"
#include "pipeline/jit/debug/trace.h"
#include "frontend/expander/bprop/bprop.h"
#include "pipeline/pynative/pynative_utils.h"
#include "utils/profile.h"
#include "include/common/utils/primitive_utils.h"
#include "pipeline/jit/pass.h"
#include "pybind_api/gil_scoped_long_running.h"
namespace mindspore {
namespace pynative {
namespace autograd {
namespace {
enum class SpecialType { kZerosLikeType = 0, kOnesLikeType = 1 };
const size_t kContainerRatio = 2;
const mindspore::HashSet<std::string> kGradBlackList{kMakeTupleOpName,         kMakeListOpName,
                                                     kTupleGetItemOpName,      kStopGradientOpName,
                                                     kUpdateStateOpName,       kNPUAllocFloatStatusOpName,
                                                     kNPUGetFloatStatusOpName, kNPUClearFloatStatusOpName};

const mindspore::HashSet<std::string> kMonadOp = {kLoadOPName, kDependOpName, kUpdateStateOpName};

const mindspore::HashSet<std::string> kMetaFuncGraphOp{
  kPyExecuteOpName,
  kAttrMutableOpName,
  kMakeDictOpName,
};

mindspore::HashMap<std::string, FuncGraphPtr> pass_grad_graph_;

inline bool IsPrimNeedGrad(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  return kGradBlackList.find(prim->name()) == kGradBlackList.end();
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

AnfNodePtr BuildSparseTensorNode(const FuncGraphPtr &tape, const ValuePtr &sparse_value,
                                 const AnfNodePtr &dout_value_node) {
  MS_EXCEPTION_IF_NULL(tape);
  MS_EXCEPTION_IF_NULL(sparse_value);
  if (sparse_value->isa<tensor::CSRTensor>()) {
    auto csr_tensor = sparse_value->cast<tensor::CSRTensorPtr>();
    MS_EXCEPTION_IF_NULL(csr_tensor);
    auto indptr_node = PyNativeAlgo::Common::CreateValueNodeByValue(csr_tensor->GetIndptr());
    auto indices_node = PyNativeAlgo::Common::CreateValueNodeByValue(csr_tensor->GetIndices());
    auto value_shape = GetSparseTensorShapeNode(csr_tensor->shape());
    auto special_like_csr_node =
      tape->NewCNode({NewValueNode(prim::kPrimMakeTuple), indptr_node, indices_node, dout_value_node, value_shape});
    special_like_csr_node->set_abstract(sparse_value->ToAbstract()->Broaden());
    return special_like_csr_node;
  } else if (sparse_value->isa<tensor::COOTensor>()) {
    auto coo_tensor = sparse_value->cast<tensor::COOTensorPtr>();
    MS_EXCEPTION_IF_NULL(coo_tensor);
    auto indices_node = PyNativeAlgo::Common::CreateValueNodeByValue(coo_tensor->GetIndices());
    auto value_shape = GetSparseTensorShapeNode(coo_tensor->shape());
    auto special_like_coo_node =
      tape->NewCNode({NewValueNode(prim::kPrimMakeTuple), indices_node, dout_value_node, value_shape});
    special_like_coo_node->set_abstract(sparse_value->ToAbstract()->Broaden());
    return special_like_coo_node;
  }
  MS_LOG(EXCEPTION) << "Get invalid sparse tensor";
}

AnfNodePtr BuildSpecialNode(const FuncGraphPtr &tape, const ValuePtr &value, const abstract::AbstractBasePtr &abs,
                            const SpecialType &type) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>() || value->isa<Scalar>()) {
    auto prim_node =
      (type == SpecialType::kZerosLikeType ? NewValueNode(prim::kPrimZerosLike) : NewValueNode(prim::kPrimOnesLike));
    auto value_node = PyNativeAlgo::Common::CreateValueNodeByValue(value, abs);
    auto special_like_value = tape->NewCNode({prim_node, value_node});
    special_like_value->set_abstract(value_node->abstract());
    return special_like_value;
  } else if (value->isa<tensor::CSRTensor>()) {
    auto csr_tensor = value->cast<tensor::CSRTensorPtr>();
    MS_EXCEPTION_IF_NULL(csr_tensor);
    auto data = csr_tensor->GetValues();
    return BuildSpecialNode(tape, data, nullptr, type);
  } else if (value->isa<tensor::COOTensor>()) {
    auto coo_tensor = value->cast<tensor::COOTensorPtr>();
    MS_EXCEPTION_IF_NULL(coo_tensor);
    auto data = coo_tensor->GetValues();
    return BuildSpecialNode(tape, data, nullptr, type);
  } else if (value->isa<ValueSequence>()) {
    auto tuple = value->cast<ValueSequencePtr>();
    abstract::AbstractSequencePtr abs_seq;
    if (abs == nullptr) {
      abs_seq =
        PyNativeAlgo::Common::SetAbstractValueToAnyValue(value->ToAbstract())->cast<abstract::AbstractSequencePtr>();
    } else {
      abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    }
    std::vector<AnfNodePtr> args{NewValueNode(prim::kPrimMakeTuple)};
    for (size_t i = 0; i < tuple->size(); ++i) {
      AnfNodePtr special_like_value = BuildSpecialNode(tape, tuple->value()[i], abs_seq->elements()[i], type);
      (void)args.emplace_back(special_like_value);
    }
    auto special_like_value = tape->NewCNode(args);
    special_like_value->set_abstract(abs_seq);
    return special_like_value;
  } else if (value->isa<ValueDictionary>()) {
    const auto &dic_v = value->cast<ValueDictionaryPtr>()->value();
    std::vector<ValuePtr> v_list;
    std::transform(dic_v.begin(), dic_v.end(), std::back_inserter(v_list),
                   [](const std::pair<ValuePtr, ValuePtr> &elem) { return elem.second; });
    MS_EXCEPTION_IF_NULL(abs);
    const auto &abs_dict = abs->cast<abstract::AbstractDictionaryPtr>();
    MS_EXCEPTION_IF_NULL(abs_dict);
    abstract::AbstractBasePtrList abs_list;
    std::transform(abs_dict->elements().begin(), abs_dict->elements().end(), std::back_inserter(abs_list),
                   [](const auto &elem) { return elem.second; });
    return BuildSpecialNode(tape, std::make_shared<ValueTuple>(v_list),
                            std::make_shared<abstract::AbstractTuple>(abs_list), type);
  } else if (value->isa<None>() || value->isa<Type>()) {
    return BuildSpecialNode(tape, MakeValue(0), nullptr, type);
  } else {
    MS_EXCEPTION(TypeError) << "For value " << value->ToString() << ", the type is not support now";
  }
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
    return std::all_of(cnode->inputs().begin() + 1, cnode->inputs().end(),
                       [](const auto &node) { return IsZerosLikeNode(node) == true; });
  } else {
    return false;
  }
}

bool IsConstant(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>()) {
    const auto &tensor = value->cast<tensor::TensorPtr>();
    auto auto_grad_meta_data = tensor->auto_grad_meta_data();
    MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
    if (auto_grad_meta_data->grad_type() == TensorGradType::kParameter ||
        auto_grad_meta_data->grad_type() == TensorGradType::kInput) {
      return false;
    }
    auto k_node = auto_grad_meta_data->k_node();
    if (k_node != nullptr) {
      return false;
    }
    return true;
  } else if (value->isa<tensor::COOTensor>()) {
    auto coo_tensor = value->cast<tensor::COOTensorPtr>();
    return IsConstant(coo_tensor->GetIndices());
  } else if (value->isa<tensor::CSRTensor>()) {
    auto csr_tensor = value->cast<tensor::CSRTensorPtr>();
    return IsConstant(csr_tensor->GetIndices());
  } else if (value->isa<ValueSequence>()) {
    auto val_seq = value->cast<ValueSequencePtr>();
    return std::all_of(val_seq->value().begin(), val_seq->value().end(),
                       [](const ValuePtr &value) { return IsConstant(value); });
  }
  return true;
}

FuncGraphPtr OptimizeBpropBuilder(const FuncGraphPtr &bprop_func_graph) {
  PyNativeAlgo::Common::DumpGraphIR("bprop_builder_before_opt.ir", bprop_func_graph);
  pipeline::ResourcePtr resource = std::make_shared<pipeline::Resource>();
  resource->set_func_graph(bprop_func_graph);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(bprop_func_graph);
  auto after_opt_bg = pipeline::MsFunctionBpropGraphPass(resource, true);
  PyNativeAlgo::Common::DumpGraphIR("bprop_builder_after_opt.ir", after_opt_bg);
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
AnfNodePtr HandleRealToComplex(const TensorPtr &input, const AnfNodePtr &din, const FuncGraphPtr &tape) {
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
  TypePtr input_type = input->Dtype();
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
    abstract::AbstractTensor(input_type, input->ToAbstract()->GetShapeTrack()));
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
  return PyNativeAlgo::Common::CreateValueNodeByValue(tuple_value);
}

bool IsValidTensorInput(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  return abs->isa<abstract::AbstractTensor>() || abs->isa<abstract::AbstractSparseTensor>();
}

bool ProcessMonadNode(const PrimitivePtr &prim, const CNodePtr &cnode, const GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(prim);
  if (kMonadOp.find(prim->name()) != kMonadOp.end()) {
    MS_LOG(DEBUG) << "Get monad cnode " << cnode->DebugString();
    return true;
  }
  if ((prim->HasAttr(GRAPH_FLAG_SIDE_EFFECT_MEM) || prim->HasAttr(GRAPH_FLAG_SIDE_EFFECT_IO)) &&
      (cnode->inputs().back()->abstract()->isa<abstract::AbstractMonad>())) {
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

void SetGradInfoForInputs(const ValuePtr &value, const VariableAdjointPtr &variable, const ParameterPtr &param) {
  if (value->isa<tensor::Tensor>()) {
    auto input_tensor = value->cast<tensor::TensorPtr>();
    auto auto_grad_meta_data = input_tensor->auto_grad_meta_data();
    MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
    auto_grad_meta_data->set_variable(variable);
    auto_grad_meta_data->set_parameter(param);
    auto_grad_meta_data->set_grad_type(TensorGradType::kInput);
  } else if (value->isa<tensor::COOTensor>()) {
    auto coo_tensor = value->cast<tensor::COOTensorPtr>();
    auto indices_tensor = coo_tensor->GetIndices();
    auto auto_grad_meta_data = std::make_shared<AutoGradMetaData>(variable, param, TensorGradType::kInput);
    indices_tensor->set_auto_grad_meta_data(auto_grad_meta_data);
  } else if (value->isa<tensor::CSRTensor>()) {
    auto csr_tensor = value->cast<tensor::CSRTensorPtr>();
    auto indices_tensor = csr_tensor->GetIndices();
    auto auto_grad_meta_data = std::make_shared<AutoGradMetaData>(variable, param, TensorGradType::kInput);
    indices_tensor->set_auto_grad_meta_data(auto_grad_meta_data);
  }
}

void SetGradMetaData(const ValuePtr &value, const VariableAdjointPtr &variable, const ParameterPtr &param = nullptr) {
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    auto auto_grad_meta_data = tensor->auto_grad_meta_data();
    if (auto_grad_meta_data == nullptr) {
      MS_LOG(DEBUG) << "tensor has no auto_grad_meta_data";
      auto_grad_meta_data = std::make_shared<AutoGradMetaData>();
      tensor->set_auto_grad_meta_data(auto_grad_meta_data);
    }
    auto_grad_meta_data->set_variable(variable);
    if (param != nullptr) {
      auto_grad_meta_data->set_parameter(param);
      auto_grad_meta_data->set_grad_type(TensorGradType::kParameter);
    }
  } else if (value->isa<ValueSequence>()) {
    auto value_sequence = value->cast<ValueSequencePtr>();
    for (auto val : value_sequence->value()) {
      SetGradMetaData(val, variable);
    }
  }
}

void ClearGradMetaData(const ValuePtr &value) {
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    tensor->set_auto_grad_meta_data(nullptr);
  } else if (value->isa<ValueSequence>()) {
    auto value_sequence = value->cast<ValueSequencePtr>();
    for (auto val : value_sequence->value()) {
      ClearGradMetaData(val);
    }
  }
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

void FunctionNode::AddNextEdge(const VariableAdjointPtr &next_variable, const AnfNodePtr &din) {
  MS_EXCEPTION_IF_NULL(next_variable);
  MS_EXCEPTION_IF_NULL(din);
  // next_node and its corresponding din
  (void)next_edges_.emplace_back(std::make_pair(next_variable, din));
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
  if (static_cast<bool>(MS_UNLIKELY(IsZerosLikeNode(fn()->accumulate_dout())))) {
    fn()->set_accumulate_dout(
      BuildSpecialNode(fn()->tape(), out_value_, fn()->accumulate_dout()->abstract(), SpecialType::kZerosLikeType));
  }
  const auto &accumulate_dout = fn()->accumulate_dout();
  const auto &dout_abs = accumulate_dout->abstract();
  MS_EXCEPTION_IF_NULL(dout_abs);
  // For input, if it is a sparsetensor, we need return a sparsetensor.
  if (out_value_->isa<tensor::Tensor>() || dout_abs->isa<abstract::AbstractSparseTensor>()) {
    return accumulate_dout;
  } else if (out_value_->isa<tensor::MetaSparseTensor>()) {
    return BuildSparseTensorNode(fn()->tape(), out_value_, accumulate_dout);
  }
  return accumulate_dout;
}

std::string VariableAdjoint::ToString() const {
  std::ostringstream buf;
  buf << "Variable id: " << PyNativeAlgo::Common::GetIdByValue(out_value_) << " is_need_grad: " << is_need_grad_
      << ", is_need_propagate " << is_need_propagate_ << " is_leaf: " << is_leaf_ << "   ";
  for (size_t i = 0; i < fn()->next_edges().size(); ++i) {
    auto last_variable = fn()->next_edges()[i].first;
    auto din = fn()->next_edges()[i].second;
    buf << "last variable id: " << PyNativeAlgo::Common::GetIdByValue(last_variable->out_value_)
        << " din: " << din->DebugString() << "   ";
  }
  return buf.str();
}

AutoGradCellImpl::AutoGradCellImpl(const std::vector<ValuePtr> &input_param_values, const AbstractBasePtrList &abs_list,
                                   size_t op_num_in_bprop_graph)
    : ad_param_(std::make_shared<AdParam>()), op_num_in_bprop_graph_(op_num_in_bprop_graph) {
  ad_param()->tape_->debug_info()->set_name("grad_top");
  MS_LOG(DEBUG) << "Start AutoGradCellImpl, input size: " << input_param_values.size();
  if (op_num_in_bprop_graph_ != 0) {
    size_t estimate_element_num = op_num_in_bprop_graph_ * kContainerRatio;
    ad_param()->variable_adjoint_set_.reserve(estimate_element_num);
    weights_used_in_graph_.reserve(estimate_element_num);
  }
  for (size_t i = 0; i < input_param_values.size(); ++i) {
    auto input_parameter = ad_param()->fg_->add_parameter();
    input_parameter->set_abstract(abs_list[i]);
    TraceGuard trace_guard(std::make_shared<TraceCopy>(input_parameter->debug_info()));
    auto tape_parameter = ad_param()->tape_->add_parameter();
    tape_parameter->set_abstract(abs_list[i]);

    auto zeros_like_dout = BuildSpecialNode(ad_param()->tape_, MakeValue(0), abs_list[i], SpecialType::kZerosLikeType);
    auto func_node = std::make_shared<FunctionNode>(ad_param()->tape_, zeros_like_dout);
    auto input_adjoint = std::make_shared<VariableAdjoint>(func_node, input_param_values[i], true);
    if (!input_param_values[i]->isa<ValueSequence>()) {
      SetGradInfoForInputs(input_param_values[i], input_adjoint, input_parameter);
    } else {
      input_adjoint->set_is_need_grad(false);
    }
    (void)cell_inputs_.emplace_back(std::make_pair(input_parameter, input_adjoint));
    ad_param()->variable_adjoint_set_.insert(input_adjoint);
  }
}

bool AutoGradCellImpl::KPynativeOp(const GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(grad_param);

  auto &prim = grad_param->prim;
  if (!IsPrimNeedGrad(prim)) {
    MS_LOG(DEBUG) << "Prim " << prim->name() << " not need do op grad";
    return true;
  }

  // construct zeroslike placeholder, if need use in bprop, we replace it in backprogate.
  AnfNodePtr dout = BuildSpecialNode(ad_param()->tape_, MakeValue(0), grad_param->out_abs, SpecialType::kZerosLikeType);
  auto fn = std::make_shared<FunctionNode>(ad_param()->tape_, dout);
  auto variable_adjoint = std::make_shared<VariableAdjoint>(fn, grad_param->out);
  // Custom forward cnode no need record in bprop graph, because it is a flag cnode for run python. So just create
  // bprop_cut grad op is ok
  bool is_custom_prim =
    IsPrimitiveEquals(prim, prim::kPrimHookBackward) || IsPrimitiveEquals(prim, prim::kPrimCellBackwardHook);
  if (!grad_param->grad_by_value && !is_custom_prim) {
    auto k_node = BuildKNode(NewValueNode(prim), grad_param, true);
    variable_adjoint->set_k_node(k_node);
    SetKNodeInfo(grad_param->out, k_node);
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
      MS_LOG(DEBUG) << "Expander has no bprop of this prim: " << prim->name();
      BuildCustomBpropCNode(input_node, prim, &outputs);
    }
  }
  if (outputs.empty()) {
    MS_LOG(DEBUG) << "This op has not custom bprop: " << prim->name();
    BuildFakeBpropCNode(input_node, &outputs);
    variable_adjoint->set_is_fake_bprop(true);
    variable_adjoint->set_fake_prim_name(prim->name());
  }
  UpdateNextEdges(variable_adjoint, outputs, grad_param->op_args);
  MS_LOG(DEBUG) << "Finish update next edges, "
                << "prim is: " << prim->name() << " variable is: " << variable_adjoint->ToString();
  (void)ad_param()->variable_adjoint_set_.insert(variable_adjoint);
  SetGradMetaData(grad_param->out, variable_adjoint);
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
      if (PyNativeAlgo::Common::IsParamTensor(grad_param->op_args_grad_type[i])) {
        auto parameter = MapParameter(grad_param->op_args[i]);
        if (parameter != nullptr) {
          (void)args_node_list.emplace_back(parameter);
          continue;
        }
      }
      // Valuenode, cnode
      (void)args_node_list.emplace_back(
        PyNativeAlgo::Common::CreateValueNodeByValue(grad_param->op_args[i], grad_param->input_abs[i]->Clone()));
    }
    bprop_cnode = GetBpropGraphCNode(grad_param, args_node_list, &dout);
  } else {
    k_node = BuildKNode(NewValueNode(grad_param->source_fg), grad_param, false);
    BuildKNodeListFromPrimalCNode(grad_param->op_args, &args_node_list);
    bprop_cnode = GetBpropGraphCNode(grad_param, args_node_list, &dout);
  }
  auto fn = std::make_shared<FunctionNode>(ad_param()->tape_, dout);
  auto variable_adjoint = std::make_shared<VariableAdjoint>(fn, grad_param->out);
  variable_adjoint->set_k_node(k_node);
  std::vector<CNodePtr> outputs;
  for (size_t i = 0; i < grad_param->op_args.size(); ++i) {
    CNodePtr din = nullptr;
    if (grad_param->is_not_support_by_expander) {
      // bprop_app[0] env
      din = ad_param()->tape_->NewCNode(
        {NewValueNode(prim::kPrimTupleGetItem), bprop_cnode, NewValueNode(SizeToLong(i + 1))});
    } else {
      din =
        ad_param()->tape_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), bprop_cnode, NewValueNode(SizeToLong(i))});
    }
    din->set_abstract(grad_param->input_abs[i]);
    (void)outputs.emplace_back(din);
  }
  UpdateNextEdges(variable_adjoint, outputs, grad_param->op_args);
  (void)ad_param()->variable_adjoint_set_.insert(variable_adjoint);
  SetGradMetaData(grad_param->out, variable_adjoint);
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
  *tape_dout = BuildSpecialNode(ad_param()->tape_, grad_param->out, grad_param->out_abs, SpecialType::kZerosLikeType);
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
  auto get_bprop = bprop_builder->NewCNode(
    {NewValueNode(prim::kPrimTupleGetItem), fprop_app, NewValueNode(static_cast<int64_t>(kIndex1))});

  // Get bprop from fprop_fg, it is 2th output of fprop_fg
  AnfNodePtrList node_list{get_bprop};
  auto dout = bprop_builder->add_parameter();
  dout->set_abstract(grad_param->out_abs);
  (void)node_list.emplace_back(dout);
  auto call_bprop = bprop_builder->NewCNode(node_list);
  bprop_builder->set_output(call_bprop);

  // Call pass for optimize graph, such as inline
  auto after_opt_fg = OptimizeBpropBuilder(bprop_builder);
  ad_param()->tape_->set_flag(kFlagMSFunctionGraph, true);
  // Call by tape_
  MS_EXCEPTION_IF_NULL(tape_dout);
  *tape_dout = BuildSpecialNode(ad_param()->tape_, MakeValue(0), grad_param->out_abs, SpecialType::kZerosLikeType);
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
  PyNativeAlgo::Common::DumpGraphIR("ad_input_graph.ir", grad_param->fg);
  auto current_ad_param = ad_param_;
  ad_param_ = std::make_shared<AdParam>();
  ad_param()->tape_->debug_info()->set_name("ad_graph");

  GradGraphByExpander(grad_param);

  if (ad_param()->last_node_ != nullptr) {
    // Set dout parameter
    const auto last_prim = GetCNodePrimitive(ad_param()->last_node_);
    if (kMonadOp.find(last_prim->name()) != kMonadOp.end()) {
      ad_param()->last_node_ = common::AnfAlgo::VisitKernelWithReturnType(
                                 ad_param()->last_node_, 0, false, {prim::kPrimTupleGetItem, prim::kPrimMakeTuple})
                                 .first;
    }
    if (!ad_param()->anfnode_to_variable_adjoint_.count(ad_param()->last_node_)) {
      MS_LOG(EXCEPTION) << "Can not find last node" << ad_param()->last_node_->DebugString();
    }
    ad_param()->last_variable_ = ad_param()->anfnode_to_variable_adjoint_[ad_param()->last_node_];
    auto ad_graph_dout = ad_param()->tape_->add_parameter();
    ad_graph_dout->set_abstract(ad_param()->last_variable_->out_value()->ToAbstract());
    ad_param()->last_variable_->fn()->UpdateAccumulativeDout(ad_graph_dout);
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
  PyNativeAlgo::Common::DumpGraphIR("ad_output_graph.ir", ad_graph);

  // Save ad graph in cache
  if (!grad_param->use_dynamic_shape_process) {
    pass_grad_graph_[grad_param->graph_cache_key] = BasicClone(ad_graph);
  }
  // Replace cnode with valuenode for reduce compute
  bool ms_function_by_value = grad_param->is_ms_function_graph && grad_param->grad_by_value;
  if (ms_function_by_value) {
    PyNativeAlgo::Common::ReplaceCNodeWithValueNode(ad_graph);
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
    if (ProcessMonadNode(prim, cnode, grad_param) || IsPrimitiveEquals(prim, prim::kPrimStopGradient)) {
      continue;
    }
    MS_LOG(DEBUG) << "Get cnode " << cnode->DebugString() << ", " << cnode->fullname_with_scope();
    if (IsPrimitiveEquals(prim, prim::kPrimMakeTuple) || IsPrimitiveEquals(prim, prim::kPrimMakeList)) {
      (void)BuildKNodeForMakeTuple(cnode);
      continue;
    } else if (IsPrimitiveEquals(prim, prim::kPrimTupleGetItem)) {
      (void)BuildKNodeForTupleGetItem(cnode);
      continue;
    }

    std::vector<AnfNodePtr> cnode_inputs{std::make_shared<ValueNode>(prim)};
    auto op_args = GetInputArgs(cnode, &cnode_inputs);
    AnfNodePtr k_node = nullptr;
    if (IsPrimitiveEquals(prim, prim::kPrimMirror)) {
      k_node = ad_param()->anfnode_to_variable_adjoint_.at(cnode->input(kIndex1))->k_node();
    } else {
      auto c_k_node = ad_param()->tape_->NewCNode(cnode_inputs);
      c_k_node->set_abstract(cnode->abstract());
      // In ms function, copy forward graph cnode info to bprop graph
      if (ms_function_by_value && cnode->forward().first != nullptr) {
        auto new_v_node = PyNativeAlgo::Common::CreateValueNodeByValue(cnode->forward().first->value(),
                                                                       cnode->forward().first->abstract());
        c_k_node->set_forward(new_v_node, cnode->forward().second);
        ad_param()->tape_->set_used_forward_nodes({c_k_node});
      }
      k_node = c_k_node;
    }
    MS_LOG(DEBUG) << "Build knode " << k_node->DebugString();
    // Set out
    auto out = PyNativeAlgo::Common::CreatOutputTensorValueByAbstract(cnode->abstract());
    (void)cnode_inputs.emplace_back(k_node);
    // Set dout
    AnfNodePtr dout = BuildSpecialNode(ad_param()->tape_, MakeValue(0), cnode->abstract(), SpecialType::kZerosLikeType);
    (void)cnode_inputs.emplace_back(dout);
    auto input_node = ad_param()->tape_->NewCNode(cnode_inputs);
    input_node->set_abstract(cnode->abstract());

    std::vector<CNodePtr> outputs;
    // Get bprop by expander
    auto ret = BpropExpander(&outputs, &ad_param()->users_).Run(input_node);
    if (!ret || outputs.empty()) {
      // Get bprop by meta graph
      if (grad_param->is_ms_function_graph && kMetaFuncGraphOp.find(prim->name()) != kMetaFuncGraphOp.end()) {
        ProcessMetaFuncGraphOp(grad_param, prim, cnode, op_args, out);
        continue;
      } else {
        // Get bprop by python custom
        MS_LOG(DEBUG) << "Expander has no bprop of this node: " << input_node->DebugString();
        BuildCustomBpropCNode(input_node, prim, &outputs);
      }
    }
    auto fn = std::make_shared<FunctionNode>(ad_param()->tape_, dout);
    auto variable_adjoint = std::make_shared<VariableAdjoint>(fn, out);
    variable_adjoint->set_k_node(k_node);
    // Get bprop by fake bprop
    if (outputs.empty()) {
      MS_LOG(DEBUG) << "Build fake bprop for this node: " << input_node->DebugString();
      BuildFakeBpropCNode(input_node, &outputs);
      variable_adjoint->set_is_fake_bprop(true);
      variable_adjoint->set_fake_prim_name(prim->name());
    }
    // Create current op node din edge
    UpdateNextEdges(variable_adjoint, outputs, op_args);
    SetGradMetaData(out, variable_adjoint);
    (void)ad_param()->anfnode_to_variable_adjoint_.insert(std::make_pair(node, variable_adjoint));
    (void)ad_param()->variable_adjoint_set_.insert(variable_adjoint);
  }
}

void AutoGradCellImpl::ProcessMetaFuncGraphOp(const GradParamPtr &grad_param, const PrimitivePtr &prim,
                                              const CNodePtr &cnode, const ValuePtrList &op_args, const ValuePtr &out) {
  MS_EXCEPTION_IF_NULL(grad_param);
  MS_EXCEPTION_IF_NULL(prim);
  static auto pyexecute_ins = std::make_shared<prim::PyExecuteGradient>("PyExecuteGradient");
  static auto mutable_ins = std::make_shared<prim::MutableGradient>("MutableGradient");
  static auto make_dict_ins = std::make_shared<prim::MakeDictGradient>("make_dict_gradient");
  AbstractBasePtrList args_abs_list;
  for (size_t i = 1; i < cnode->size(); ++i) {
    (void)args_abs_list.emplace_back(cnode->input(i)->abstract());
  }
  FuncGraphPtr grad_func_graph = nullptr;
  if (prim->name() == kPyExecuteOpName) {
    grad_func_graph = pyexecute_ins->GenerateFuncGraph(args_abs_list);
  } else if (prim->name() == kAttrMutableOpName) {
    grad_func_graph = mutable_ins->GenerateFuncGraph(args_abs_list);
  } else if (prim->name() == kMakeDictOpName) {
    grad_func_graph = make_dict_ins->GenerateFuncGraph(args_abs_list);
  }
  MS_EXCEPTION_IF_NULL(grad_func_graph);
  auto out_abs = PyNativeAlgo::Common::SetAbstractValueToAnyValue(out->ToAbstract());
  auto meta_graph_grad_param =
    std::make_shared<GradParam>(prim, op_args, args_abs_list, out, out_abs, grad_func_graph, nullptr,
                                grad_param->grad_by_value, grad_param->use_dynamic_shape_process);
  meta_graph_grad_param->is_not_support_by_expander = true;
  meta_graph_grad_param->is_ms_function_graph = true;
  meta_graph_grad_param->graph_cache_key = grad_param->graph_cache_key;
  if (!KPynativeWithFProp(meta_graph_grad_param)) {
    MS_LOG(EXCEPTION) << "Failed to make meta graph, cnode info: " << cnode->DebugString();
  }
  PyNativeAlgo::Common::GetPyNativeExecutor()->grad_executor()->top_cell()->set_need_do_final_opt(true);
}

void AutoGradCellImpl::CreateParameterAdjoint(const GradParamPtr &grad_param) const {
  auto &graph_parameters = grad_param->fg->parameters();
  if (graph_parameters.size() != grad_param->op_args.size()) {
    MS_LOG(EXCEPTION) << "Parameters size " << graph_parameters.size() << " is not equal to graph input size "
                      << grad_param->op_args.size();
  }
  for (size_t i = 0; i < graph_parameters.size(); ++i) {
    MS_LOG(DEBUG) << "Get param " << graph_parameters[i]->DebugString();
    ParameterPtr param = ad_param()->tape_->add_parameter();
    auto tensor = PyNativeAlgo::Common::GetTensorFromParam(graph_parameters[i]);
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
    auto zeros_like_dout =
      BuildSpecialNode(ad_param()->tape_, MakeValue(0), graph_parameters[i]->abstract(), SpecialType::kZerosLikeType);
    auto func_node = std::make_shared<FunctionNode>(ad_param()->tape_, zeros_like_dout);
    // Copy to avoid corrupt real input grad info.
    auto op_arg = PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(grad_param->op_args[i]);
    ClearGradMetaData(op_arg);
    auto adjoint = std::make_shared<VariableAdjoint>(func_node, op_arg, true);
    adjoint->set_k_node(param);
    SetGradMetaData(op_arg, adjoint, graph_parameters[i]->cast<ParameterPtr>());
    (void)ad_param()->variable_adjoint_set_.insert(adjoint);
    (void)ad_param()->anfnode_to_variable_adjoint_.insert(std::make_pair(graph_parameters[i], adjoint));
  }
}

ValuePtrList AutoGradCellImpl::GetInputArgs(const CNodePtr &cnode, std::vector<AnfNodePtr> *cnode_inputs) const {
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
      (void)PyNativeAlgo::Common::SetValueGradInfo(v_node->value(), nullptr, TensorGradType::kConstant);
      // In case of ms function forward graph and pynative bprop graph used same valuenode
      auto new_v_node = PyNativeAlgo::Common::CreateValueNodeByValue(v_node->value(), v_node->abstract());
      (void)cnode_inputs->emplace_back(new_v_node);
      op_args.emplace_back(v_node->value());
    } else {
      // Make Fake value
      auto v = MakeValue(0);
      auto new_v_node = NewValueNode(v);
      new_v_node->set_abstract(input_node->abstract());
      (void)cnode_inputs->emplace_back(new_v_node);
      (void)op_args.emplace_back(v);
      MS_LOG(DEBUG) << "Get input node " << input_node->DebugString();
    }
  }
  return op_args;
}

void AutoGradCellImpl::UpdateOutputNodeOfTopCell(const ValuePtr &sens_out) {
  MS_EXCEPTION_IF_NULL(sens_out);
  MS_LOG(DEBUG) << "Real output of top cell is " << PyNativeAlgo::Common::GetIdByValue(sens_out);
  ad_param()->sens_value_ = sens_out;
}

FuncGraphPtr AutoGradCellImpl::Finish(const tensor::TensorPtrList &weights, const std::vector<size_t> &grad_position,
                                      const GradAttr &grad_attr) {
  // Set sens node and weights node
  SetSensAndWeights(weights, grad_attr.has_sens);

  // BackPropagate sensitivity, except when the last node is a valuenode which may be obtained by constant folding;
  if (ad_param()->last_variable_->is_need_grad() && !ad_param()->last_variable_->is_leaf()) {
    (void)BackPropagate();
  }
  SetOutput(weights, grad_position, grad_attr);
  // Replace Parameter of primal func graph with parameter of ad_param()->tape_;
  AnfNodePtrList params = ExtractParamters(weights, ad_param()->fg_);
  ReplacePrimalParameter(params, grad_attr.has_sens);
  PyNativeAlgo::Common::DumpGraphIR("before_final_opt.ir", ad_param()->tape_);
  // Clear weights grad info
  for (auto weight : weights) {
    weight->set_auto_grad_meta_data(nullptr);
  }
  return ad_param()->tape_;
}

CNodePtr AutoGradCellImpl::ConstructBpropGraphInput(const GradParamPtr &grad_param, const AnfNodePtr &dout,
                                                    const VariableAdjointPtr &variable_adjoint, bool is_custom_prim) {
  MS_EXCEPTION_IF_NULL(grad_param);
  std::vector<AnfNodePtr> node_list;
  (void)node_list.emplace_back(NewValueNode(grad_param->prim));
  if (grad_param->grad_by_value || is_custom_prim) {
    for (size_t i = 0; i < grad_param->op_args.size(); ++i) {
      if (PyNativeAlgo::Common::IsParamTensor(grad_param->op_args_grad_type[i])) {
        // To solve the input is a tuple like (parameter, ...)
        auto parameter = MapParameter(grad_param->op_args[i]);
        MS_EXCEPTION_IF_NULL(parameter);
        (void)node_list.emplace_back(parameter);
        continue;
      }
      // Node abstract obj may free, so v node abstract will be not correct
      (void)node_list.emplace_back(
        PyNativeAlgo::Common::CreateValueNodeByValue(grad_param->op_args[i], grad_param->input_abs[i]->Clone()));
    }
    // Set out
    (void)node_list.emplace_back(PyNativeAlgo::Common::CreateValueNodeByValue(grad_param->out, grad_param->out_abs));
  } else {
    // Input is a Parameter or cnode, not a value node
    BuildKNodeListFromPrimalCNode(grad_param->op_args, &node_list);
    // Set out
    MS_EXCEPTION_IF_NULL(variable_adjoint);
    (void)node_list.emplace_back(variable_adjoint->k_node());
  }
  // Set dout
  (void)node_list.emplace_back(dout);
  auto input_node = ad_param()->tape_->NewCNode(node_list);
  return input_node;
}

void AutoGradCellImpl::BuildKNodeListFromPrimalCNode(const ValuePtrList &op_args,
                                                     std::vector<AnfNodePtr> *const node_list) {
  for (size_t i = 0; i < op_args.size(); ++i) {
    (void)node_list->emplace_back(BuildKNodeForCNodeInput(op_args[i]));
    MS_LOG(DEBUG) << "Get knode for input:  " << PyNativeAlgo::Common::GetIdByValue(op_args[i]);
  }
}

void AutoGradCellImpl::SetKNodeInfo(const ValuePtr &value, const AnfNodePtr &k_node) {
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    auto auto_grad_meta_data = tensor->auto_grad_meta_data();
    MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
    auto_grad_meta_data->set_k_node(k_node);
    (void)k_nodes_used_in_graph_.emplace_back(k_node);
  } else if (value->isa<ValueSequence>()) {
    auto value_sequence = value->cast<ValueSequencePtr>();
    for (size_t i = 0; i < value_sequence->value().size(); ++i) {
      auto sub_k_node = ad_param()->tape_->NewCNode(
        {NewValueNode(prim::kPrimTupleGetItem), k_node, NewValueNode(static_cast<int64_t>(i))});
      auto sub_abstract = PyNativeAlgo::Common::SetAbstractValueToAnyValue(value_sequence->value()[i]->ToAbstract());
      sub_k_node->set_abstract(sub_abstract);
      SetKNodeInfo(value_sequence->value()[i], sub_k_node);
    }
  }
}

AnfNodePtr AutoGradCellImpl::BuildKNode(const AnfNodePtr &prim, const GradParamPtr &grad_param, bool from_single_op) {
  MS_EXCEPTION_IF_NULL(grad_param);
  AnfNodePtrList node_list;
  (void)node_list.emplace_back(prim);
  for (size_t i = 0; i < grad_param->op_args.size(); ++i) {
    (void)node_list.emplace_back(BuildKNodeForCNodeInput(grad_param->op_args[i]));
  }
  auto k_node = ad_param()->tape_->NewCNode(node_list);
  k_node->set_abstract(grad_param->out_abs);
  if (from_single_op && grad_param->out_used_in_bporp_graph) {
    auto v_node = PyNativeAlgo::Common::CreateValueNodeByValue(grad_param->out, grad_param->out_abs);
    k_node->set_forward(v_node, "");
    ad_param()->tape_->set_used_forward_nodes({k_node});
  }
  MS_LOG(DEBUG) << "Build knode " << k_node->DebugString();
  return k_node;
}

AnfNodePtr AutoGradCellImpl::BuildKNodeForCNodeInput(const ValuePtr &input) {
  MS_EXCEPTION_IF_NULL(input);
  if (input->isa<tensor::Tensor>()) {
    auto tensor = input->cast<tensor::TensorPtr>();
    auto auto_grad_meta_data = tensor->auto_grad_meta_data();
    MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
    auto k_node = auto_grad_meta_data->k_node();
    if (k_node != nullptr) {
      return k_node;
    }
    if (auto_grad_meta_data->grad_type() == TensorGradType::kParameter ||
        auto_grad_meta_data->grad_type() == TensorGradType::kInput) {
      return MapParameter(input);
    }
  } else if (input->isa<ValueSequence>() && !IsConstant(input)) {
    std::vector<AnfNodePtr> inputs;
    (void)inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    auto val_sequence = input->cast<ValueSequencePtr>();
    for (auto value : val_sequence->value()) {
      (void)inputs.emplace_back(BuildKNodeForCNodeInput(value));
    }
    auto k_node = ad_param()->tape_->NewCNode(inputs);
    k_node->set_abstract(PyNativeAlgo::Common::SetAbstractValueToAnyValue(input->ToAbstract()));
    return k_node;
  }
  auto value_node = NewValueNode(input);
  value_node->set_abstract(PyNativeAlgo::Common::SetAbstractValueToAnyValue(input->ToAbstract()));
  return value_node;
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
  AnfNodePtr dout = BuildSpecialNode(ad_param()->tape_, out_value, input_node->abstract(), SpecialType::kZerosLikeType);
  auto fn = std::make_shared<FunctionNode>(ad_param()->tape_, dout);
  auto variable_adjoint = std::make_shared<VariableAdjoint>(fn, out_value);
  auto k_node = ad_param()->tape_->NewCNode(inputs);
  k_node->set_abstract(input_node->abstract());
  variable_adjoint->set_k_node(k_node);
  // Create dout for maketuple
  std::vector<CNodePtr> make_tuple_dout;
  for (size_t i = 1; i < cnode->size(); ++i) {
    auto d =
      ad_param()->tape_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), dout, NewValueNode(SizeToLong(i - 1))});
    d->set_abstract(cnode->input(i)->abstract());
    (void)make_tuple_dout.emplace_back(d);
    AddUser(dout, d, 1);
  }
  UpdateNextEdges(variable_adjoint, make_tuple_dout, op_args);
  (void)ad_param()->anfnode_to_variable_adjoint_.insert(std::make_pair(input_node, variable_adjoint));
  (void)ad_param()->variable_adjoint_set_.insert(variable_adjoint);
  return k_node;
}

AnfNodePtr AutoGradCellImpl::BuildKNodeForTupleGetItem(const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(input_node);
  MS_LOG(DEBUG) << "Build knode for TupleGetItem " << input_node->DebugString();
  const auto &tuple_item_cnode = input_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(tuple_item_cnode);
  // Find make tuple or sens(tuple) node for get out value
  const auto input_adjoint_iter = ad_param()->anfnode_to_variable_adjoint_.find(tuple_item_cnode->input(kIndex1));
  if (input_adjoint_iter == ad_param()->anfnode_to_variable_adjoint_.end()) {
    MS_LOG(EXCEPTION) << "Cannot find input in adjoint map, inp: " << tuple_item_cnode->input(kIndex1)->DebugString();
  }
  const auto &v_tuple = input_adjoint_iter->second->out_value()->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(v_tuple);
  auto index_value = GetValueNode<Int64ImmPtr>(tuple_item_cnode->input(kIndex2));
  auto index_value_int = LongToSize(index_value->value());
  auto out_value = (*v_tuple)[index_value_int];
  MS_EXCEPTION_IF_NULL(out_value);
  AnfNodePtr dout = BuildSpecialNode(ad_param()->tape_, out_value, input_node->abstract(), SpecialType::kZerosLikeType);
  auto fn = std::make_shared<FunctionNode>(ad_param()->tape_, dout);
  auto variable_adjoint = std::make_shared<VariableAdjoint>(fn, out_value);

  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimTupleGetItem)};
  // Get make tuple knode
  (void)inputs.emplace_back(BuildKNodeForCNodeInput(tuple_item_cnode->input(kIndex1)));
  // Get index knode
  (void)inputs.emplace_back(BuildKNodeForCNodeInput(tuple_item_cnode->input(kIndex2)));
  auto k_node = ad_param()->tape_->NewCNode(inputs);
  k_node->set_abstract(input_node->abstract());
  variable_adjoint->set_k_node(k_node);
  // Create dout for tuplegetitem
  std::vector<AnfNodePtr> tuple_getitem_dout{NewValueNode(prim::kPrimMakeTuple)};
  const auto &abs_tuple = tuple_item_cnode->input(kIndex1)->abstract()->cast<abstract::AbstractSequencePtr>();
  for (size_t i = 0; i < v_tuple->size(); ++i) {
    const auto &v = v_tuple->value()[i];
    if (i == index_value_int) {
      (void)tuple_getitem_dout.emplace_back(dout);
    } else {
      (void)tuple_getitem_dout.emplace_back(
        BuildSpecialNode(ad_param()->tape_, v, abs_tuple->elements()[i], SpecialType::kZerosLikeType));
    }
  }
  CNodePtr tuple_getitem_dout_value = ad_param()->tape_->NewCNode(tuple_getitem_dout);
  tuple_getitem_dout_value->set_abstract(tuple_item_cnode->input(kIndex1)->abstract());
  CNodePtr index_dout_value =
    BuildSpecialNode(ad_param()->tape_, index_value, tuple_item_cnode->input(kIndex1)->abstract(),
                     SpecialType::kZerosLikeType)
      ->cast<CNodePtr>();
  UpdateNextEdges(variable_adjoint, {tuple_getitem_dout_value, index_dout_value}, {v_tuple, index_value});
  AddUser(dout, tuple_getitem_dout_value, index_value_int + 1);
  (void)ad_param()->anfnode_to_variable_adjoint_.insert(std::make_pair(input_node, variable_adjoint));
  (void)ad_param()->variable_adjoint_set_.insert(variable_adjoint);
  return k_node;
}

void AutoGradCellImpl::UpdateNextEdges(const VariableAdjointPtr &variable, const std::vector<CNodePtr> &dins,
                                       const ValuePtrList &op_args) {
  if (dins.size() != op_args.size()) {
    MS_LOG(EXCEPTION) << "The size of dins is not same as op_args";
  }
  const auto &fn = variable->fn();
  MS_LOG(DEBUG) << "Begin update next edges for variable: " << variable->ToString();
  for (size_t i = 0; i < op_args.size(); ++i) {
    const auto &din = dins[i];
    MS_LOG(DEBUG) << "Input arg id: " << PyNativeAlgo::Common::GetIdByValue(op_args[i]) << ", din "
                  << din->DebugString();
    UpdateNextEdge(fn, din, op_args[i]);
  }
  if (fn->next_edges().empty()) {
    variable->set_is_need_grad(false);
  }
}

void AutoGradCellImpl::UpdateNextEdge(const FunctionNodePtr &fn, const AnfNodePtr &din, const ValuePtr &input_arg) {
  MS_EXCEPTION_IF_NULL(din);
  MS_EXCEPTION_IF_NULL(input_arg);
  if (input_arg->isa<tensor::Tensor>() || input_arg->isa<tensor::COOTensor>() || input_arg->isa<tensor::CSRTensor>()) {
    TensorPtr input_tensor = nullptr;
    if (input_arg->isa<tensor::Tensor>()) {
      input_tensor = input_arg->cast<tensor::TensorPtr>();
    } else if (input_arg->isa<tensor::COOTensor>()) {
      input_tensor = input_arg->cast<tensor::COOTensorPtr>()->GetIndices();
    } else {
      input_tensor = input_arg->cast<tensor::CSRTensorPtr>()->GetIndices();
    }
    auto auto_grad_meta_data = input_tensor->auto_grad_meta_data();
    MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
    auto variable = auto_grad_meta_data->variable();
    if (variable == nullptr) {
      return;
    }
    if (!variable->is_need_grad()) {
      return;
    }
    auto real_din = HandleRealToComplex(input_tensor, din, fn->tape());
    auto new_din =
      TraceShape(fn, variable->out_value(), variable->fn()->accumulate_dout()->abstract(), input_arg, real_din);
    fn->AddNextEdge(variable, new_din);
  } else if (input_arg->isa<ValueSequence>()) {
    auto value_seq = input_arg->cast<ValueSequencePtr>();
    for (size_t i = 0; i < value_seq->value().size(); ++i) {
      auto sub_value = value_seq->value()[i];
      CNodePtr new_din =
        ad_param()->tape_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), din, NewValueNode(SizeToLong(i))});
      new_din->set_abstract(PyNativeAlgo::Common::SetAbstractValueToAnyValue(sub_value->ToAbstract()));
      if (din == fn->fake_dout()) {
        // The new_din's index input is fn->fake_dout()
        AddUser(fn->fake_dout(), new_din, 1);
      }
      // Add next edge to fn
      UpdateNextEdge(fn, new_din, sub_value);
    }
  } else {
    MS_LOG(DEBUG) << "It is not tensor, not need derivation " << input_arg->ToString();
    return;
  }
}

void AutoGradCellImpl::BuildForwardLastNode() {
  MS_LOG(DEBUG) << "Process last node info " << PyNativeAlgo::Common::GetIdByValue(ad_param()->sens_value_);
  auto zeros_like_node =
    BuildSpecialNode(ad_param()->tape_, ad_param()->sens_value_, nullptr, SpecialType::kZerosLikeType);
  auto fn = std::make_shared<FunctionNode>(ad_param()->tape_, zeros_like_node);
  auto input_adjoint = std::make_shared<VariableAdjoint>(fn, ad_param()->sens_value_);
  if (ad_param()->sens_value_->isa<tensor::Tensor>()) {
    const auto &sens_tensor = ad_param()->sens_value_->cast<tensor::TensorPtr>();
    const auto &auto_grad_meta_data = sens_tensor->auto_grad_meta_data();
    MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
    if (PyNativeAlgo::Common::IsConstantTensor(auto_grad_meta_data->grad_type())) {
      input_adjoint->set_is_need_grad(false);
    } else {
      if (PyNativeAlgo::Common::IsParamTensor(auto_grad_meta_data->grad_type())) {
        auto_grad_meta_data->set_variable(input_adjoint);
      } else {
        UpdateNextEdge(fn, zeros_like_node, ad_param()->sens_value_);
      }
    }
  } else {
    UpdateNextEdge(fn, zeros_like_node, ad_param()->sens_value_);
  }
  (void)ad_param()->variable_adjoint_set_.insert(input_adjoint);
  ad_param()->last_variable_ = input_adjoint;
}

ParameterPtr AutoGradCellImpl::NewWeightParameter(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  auto param = ad_param()->fg_->add_parameter();
  param->set_abstract(PyNativeAlgo::Common::SetAbstractValueToAnyValue(tensor->ToAbstract()));
  param->set_default_param(tensor);
  auto auto_grad_meta_data = tensor->auto_grad_meta_data();
  if (auto_grad_meta_data == nullptr) {
    auto_grad_meta_data = std::make_shared<AutoGradMetaData>();
    tensor->set_auto_grad_meta_data(auto_grad_meta_data);
  }
  auto_grad_meta_data->set_grad_type(TensorGradType::kParameter);
  auto_grad_meta_data->set_parameter(param);
  return param;
}

ParameterPtr AutoGradCellImpl::AddParameterNode(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  auto param = NewWeightParameter(tensor);
  auto zeros_like_dout =
    BuildSpecialNode(ad_param()->tape_, MakeValue(0), param->abstract(), SpecialType::kZerosLikeType);
  auto func_node = std::make_shared<FunctionNode>(ad_param()->tape_, zeros_like_dout);
  auto input_adjoint = std::make_shared<VariableAdjoint>(func_node, tensor, true);
  (void)ad_param()->variable_adjoint_set_.insert(input_adjoint);
  auto auto_grad_meta_data = tensor->auto_grad_meta_data();
  MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
  auto_grad_meta_data->set_variable(input_adjoint);
  (void)weights_used_in_graph_.emplace_back(param);
  return param;
}

std::vector<AnfNodePtr> AutoGradCellImpl::ExtractParamters(const tensor::TensorPtrList weights,
                                                           const FuncGraphPtr &fg) {
  std::vector<AnfNodePtr> params;
  for (auto weight : weights) {
    auto parameter = ExtractParameter(weight);
    MS_EXCEPTION_IF_NULL(parameter);
    (void)params.emplace_back(std::move(parameter));
  }
  return params;
}

AnfNodePtr AutoGradCellImpl::MapParameter(const ValuePtr &value) {
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    auto auto_grad_meta_data = tensor->auto_grad_meta_data();
    MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
    auto param = auto_grad_meta_data->parameter();
    if (param != nullptr) {
      return param;
    }
    if (auto_grad_meta_data->grad_type() == TensorGradType::kParameter) {
      return AddParameterNode(tensor);
    }
    auto v = NewValueNode(value);
    v->set_abstract(PyNativeAlgo::Common::SetAbstractValueToAnyValue(value->ToAbstract()));
    return v;
  } else if (value->isa<tensor::COOTensor>()) {
    auto coo_tensor = value->cast<tensor::COOTensorPtr>();
    return MapParameter(coo_tensor->GetIndices());
  } else if (value->isa<tensor::CSRTensor>()) {
    auto csr_tensor = value->cast<tensor::CSRTensorPtr>();
    return MapParameter(csr_tensor->GetIndices());
  } else if (value->isa<ValueSequence>()) {
    const auto &val_seq = value->cast<ValueSequencePtr>();
    std::vector<AnfNodePtr> inputs;
    (void)inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    for (const auto &val : val_seq->value()) {
      (void)inputs.emplace_back(MapParameter(val));
    }
    auto cnode = ad_param()->tape_->NewCNode(inputs);
    cnode->set_abstract(PyNativeAlgo::Common::SetAbstractValueToAnyValue(value->ToAbstract()));
    return cnode;
  } else {
    auto v = NewValueNode(value);
    v->set_abstract(PyNativeAlgo::Common::SetAbstractValueToAnyValue(value->ToAbstract()));
    return v;
  }
}

ParameterPtr AutoGradCellImpl::ExtractParameter(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  auto auto_grad_meta_data = tensor->auto_grad_meta_data();
  if (auto_grad_meta_data != nullptr && (auto_grad_meta_data->grad_type() == TensorGradType::kParameter ||
                                         auto_grad_meta_data->grad_type() == TensorGradType::kInput)) {
    return auto_grad_meta_data->parameter();
  }
  return nullptr;
}

AnfNodePtr AutoGradCellImpl::TraceShape(const FunctionNodePtr &fn, const ValuePtr &out_value,
                                        const abstract::AbstractBasePtr &out_abs, const ValuePtr &input_arg,
                                        const AnfNodePtr &din) {
  MS_EXCEPTION_IF_NULL(out_value);
  MS_EXCEPTION_IF_NULL(input_arg);
  MS_EXCEPTION_IF_NULL(din);
  const auto &out_value_id = PyNativeAlgo::Common::GetIdByValue(out_value);
  const auto &input_arg_id = PyNativeAlgo::Common::GetIdByValue(input_arg);
  // The node corresponding output tensor is the same as the currently used tensor
  if (out_value_id == input_arg_id) {
    return din;
  } else if (out_value->isa<tensor::Tensor>()) {
    // out_value is be used, may be it is one of multiple output
    return BuildSpecialNode(ad_param()->tape_, out_value, out_abs, SpecialType::kZerosLikeType);
  } else if (out_value->isa<ValueSequence>()) {
    // Input is scalar tuple, list
    if (out_value_id.find('T') == std::string::npos) {
      return BuildSpecialNode(ad_param()->tape_, out_value, out_abs, SpecialType::kZerosLikeType);
    }
    // The corresponding output of node is ValueSequence, but used one of it
    std::vector<AnfNodePtr> inputs;
    if (out_value->isa<ValueTuple>()) {
      (void)inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    } else {
      (void)inputs.emplace_back(NewValueNode(prim::kPrimMakeList));
    }
    auto value_seq = out_value->cast<ValueSequencePtr>();
    auto abs_seq = out_abs->cast<abstract::AbstractSequencePtr>();
    int index = -1;
    for (size_t i = 0; i < value_seq->size(); ++i) {
      // Find the value's din, if value equal to sub_value, means value be used, is it will get din; Otherwise value's
      // din is zero , which set by second branch condition above
      auto new_din = TraceShape(fn, value_seq->value()[i], abs_seq->elements()[i], input_arg, din);
      (void)inputs.emplace_back(new_din);

      // if exist din == fake_dout, we record it in user vector
      if (din == fn->fake_dout() && new_din == din) {
        index = static_cast<int>(inputs.size()) - 1;
      }
    }
    auto new_din = ad_param()->tape_->NewCNode(inputs);
    new_din->set_abstract(out_abs);
    if (index != -1) {
      AddUser(fn->fake_dout(), new_din, index);
    }
    return new_din;
  }
  MS_LOG(DEBUG) << "Get non tensor input " << out_value->ToString();
  return BuildSpecialNode(ad_param()->tape_, out_value, out_abs, SpecialType::kZerosLikeType);
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
  // Only custom op need add this attr, hook function not need.
  if (prim->HasAttr("custom_op_bprop")) {
    (void)bprop_cut->AddAttr("custom_op_bprop", MakeValue(true));
  }
  (void)bprop_cut->AddAttr("custom_op_name", MakeValue(prim->name()));
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
    auto prim_py = prim->cast<PrimitivePyPtr>();
    if (prim_py == nullptr) {
      MS_LOG(DEBUG) << "Prim is not PrimitivePy, can not find python bprop";
      return;
    }
    py::function fn = prim_py->GetBpropFunction();
    if (py::isinstance<py::none>(fn)) {
      fn = GetBpropFunction(prim->name());
    }
    if (!fn || py::isinstance<py::none>(fn)) {
      MS_LOG(INFO) << "Can not find bprop function for " << prim->name() << ". fn: " << py::str(fn);
      return;
    }
    (void)prim_py->AddBackwardHookFn(0, fn);
    prim_py->AddAttr("custom_op_bprop", MakeValue(true));
  }
  BuildBPropCutCNode(cnode, prim, outputs);
}

void AutoGradCellImpl::BuildFakeBpropCNode(const CNodePtr &cnode, std::vector<CNodePtr> *outputs) const {
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

void AutoGradCellImpl::SetSensAndWeights(const tensor::TensorPtrList &weights, bool has_sens_arg) {
  BuildForwardLastNode();
  ParameterPtr sens_param = nullptr;
  if (has_sens_arg) {
    sens_param = ad_param()->tape_->add_parameter();
    sens_param->debug_info()->set_name("sens");
    sens_param->set_abstract(ad_param()->sens_value_->ToAbstract());
  }
  // Update dout for dout
  MS_EXCEPTION_IF_NULL(ad_param()->last_variable_);
  if (ad_param()->last_variable_->is_need_grad()) {
    if (has_sens_arg) {
      ad_param()->last_variable_->fn()->UpdateAccumulativeDout(sens_param);
    } else {
      ad_param()->last_variable_->fn()->UpdateAccumulativeDout(
        BuildSpecialNode(ad_param()->tape_, ad_param()->sens_value_, nullptr, SpecialType::kOnesLikeType));
    }
  }
  // Add weights parameter
  need_grad_weights_.clear();
  for (const auto &weight_tensor : weights) {
    (void)need_grad_weights_.emplace(weight_tensor->id());
    auto p = ad_param()->tape_->add_parameter();
    auto param = ExtractParameter(weight_tensor);
    if (param == nullptr) {
      param = NewWeightParameter(weight_tensor);
    }
    MS_EXCEPTION_IF_NULL(param);
    const auto &param_info = weight_tensor->param_info();
    MS_EXCEPTION_IF_NULL(param_info);
    const auto &param_name = param_info->name();
    p->set_name(param_name);
    p->debug_info()->set_name(param_name);
    TraceGuard trace_guard(std::make_shared<TraceCopy>(p->debug_info()));
    p->set_default_param(weight_tensor);
    p->set_abstract(PyNativeAlgo::Common::SetAbstractValueToAnyValue(weight_tensor->ToAbstract()));
  }
}

OrderedSet<VariableAdjointPtr>::reverse_iterator AutoGradCellImpl::GetLastNodeReverseIter() {
  for (auto iter = ad_param()->variable_adjoint_set_.rbegin(); iter != ad_param()->variable_adjoint_set_.rend();
       ++iter) {
    if (*iter == ad_param()->last_variable_) {
      ad_param()->last_variable_->set_is_need_propagate(true);
      return iter;
    }
  }
  return ad_param()->variable_adjoint_set_.rend();
}

void AutoGradCellImpl::BackPropagate() {
  const auto &last_node_reverse_iter = GetLastNodeReverseIter();
  for (auto iter = last_node_reverse_iter; iter != ad_param()->variable_adjoint_set_.rend(); ++iter) {
    // MS_LOG(DEBUG) << "BackPropagate cnode: " << iter->first->DebugString();
    const auto &variable = *iter;
    if (!variable->is_need_propagate() || !variable->is_need_grad()) {
      MS_LOG(DEBUG) << "No need grad, variable is: " << variable->ToString();
      continue;
    }
    if (variable->is_fake_bprop()) {
      MS_LOG(EXCEPTION) << "Illegal primitive " << variable->fake_prim_name() << "'s bprop not defined";
    }

    const auto &fn = variable->fn();
    // If zeroslike not used in funcgraph, we need replace the zeroslike placeholder with real zeroslike value.
    if (static_cast<bool>(MS_UNLIKELY(IsZerosLikeNode(fn->accumulate_dout())))) {
      fn->set_accumulate_dout(BuildSpecialNode(fn->tape(), variable->out_value(), fn->accumulate_dout()->abstract(),
                                               SpecialType::kZerosLikeType));
    }
    // Replace real dout to fake dout, update replace result to eliminate tuplegetitem
    // when accumulate_dout is tuplegetitem
    Replace(fn->fake_dout(), fn->accumulate_dout(), true);
    // replace edges which exist fake dout
    fn->ReplaceEdges();
    MS_LOG(DEBUG) << "Begin backpropagate: " << variable->ToString();
    const auto &next_edges = fn->next_edges();
    for (const auto &next_edge : next_edges) {
      const auto &last_variable = next_edge.first;
      const auto &din = next_edge.second;
      last_variable->fn()->UpdateAccumulativeDout(din);
      last_variable->set_is_need_propagate(true);
    }
  }
}

AnfNodePtr AutoGradCellImpl::GetGradNodeByIndex(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  auto auto_grad_meta_data = tensor->auto_grad_meta_data();
  MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
  auto variable = auto_grad_meta_data->variable();
  if (variable != nullptr && variable->is_need_grad()) {
    // If weight used in the forward network, but requires_grad is false, return zero like.
    if (tensor->param_info() != nullptr && !tensor->param_info()->requires_grad()) {
      MS_LOG(INFO) << "weight participate in forward calculation, but requires_grad is false";
      return BuildSpecialNode(ad_param()->tape_, tensor, nullptr, SpecialType::kZerosLikeType);
    }
    return variable->RealDout();
  }
  MS_LOG(INFO) << "parameter does not need grad, tensor: " << PyNativeAlgo::Common::GetIdByValue(tensor);
  return BuildSpecialNode(ad_param()->tape_, tensor, nullptr, SpecialType::kZerosLikeType);
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
      if (!IsValidTensorInput(cell_inputs_[index].second->out_value()->ToAbstract())) {
        MS_LOG(DEBUG) << "Get input node is not tensor "
                      << ", abs " << cell_inputs_[index].second->out_value()->ToAbstract()->ToString();
        continue;
      }
      auto real_dout = cell_inputs_[index].second->RealDout();
      MS_EXCEPTION_IF_NULL(real_dout);
      (void)inputs_grad_list.emplace_back(real_dout);
      (void)inputs_grad_spec.emplace_back(real_dout->abstract());
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

AnfNodePtr AutoGradCellImpl::GetWeightGrad(bool grad_weights, const tensor::TensorPtrList &weights,
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

void AutoGradCellImpl::SetOutput(const tensor::TensorPtrList &weights, const std::vector<size_t> &grad_position,
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
    if (IsValidTensorInput(cell_inputs_[0].second->out_value()->ToAbstract())) {
      tape_output = cell_inputs_[0].second->RealDout();
    } else {
      MS_LOG(DEBUG) << "Get first input node is not tensor " << cell_inputs_[0].second->out_value()->ToString();
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
      auto tuple_node = old_cnode->input(kIndex1);
      if (!tuple_node->isa<CNode>() || !IsPrimitiveCNode(tuple_node->cast<CNodePtr>(), prim::kPrimMakeTuple)) {
        continue;
      }
      auto index_value = GetValueNode<Int64ImmPtr>(old_cnode->input(kIndex2));
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
    (void)tr.Replace(cell_inputs_[i].first, parameters[i]);
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
    auto t = PyNativeAlgo::Common::GetTensorFromParam(weight);
    if (need_grad_weights_.find(t->id()) == need_grad_weights_.end()) {
      MS_LOG(DEBUG) << "Convert " << weight->DebugString() << " to value node by manager.";
      (void)tr.Replace(weight, PyNativeAlgo::Common::CreateValueNodeByValue(t, weight->abstract()));
    }
  }
  tr.Commit();
}

void AutoGradCellImpl::DoParameterReplaceByUser(const AnfNodePtrList &weights, bool has_sens_arg) {
  const auto &parameters = ad_param()->tape_->parameters();
  auto cell_inputs_size = cell_inputs_.size();
  for (size_t i = 0; i < cell_inputs_size; ++i) {
    Replace(cell_inputs_[i].first, parameters[i]);
  }
  size_t weight_offset = cell_inputs_size;
  if (has_sens_arg) {
    weight_offset = weight_offset + 1;
  }
  for (size_t i = 0; i < weights.size(); ++i) {
    Replace(weights[i], parameters[weight_offset + i]);
  }
  for (auto &weight : weights_used_in_graph_) {
    auto t = PyNativeAlgo::Common::GetTensorFromParam(weight);
    MS_EXCEPTION_IF_NULL(t);
    if (need_grad_weights_.find(t->id()) == need_grad_weights_.end()) {
      MS_LOG(DEBUG) << "Convert " << weight->DebugString() << " to value node by user.";
      Replace(weight, PyNativeAlgo::Common::CreateValueNodeByValue(t, weight->abstract()));
    }
  }
}

void AutoGradCellImpl::ReplacePrimalParameter(const AnfNodePtrList &weights, bool has_sens_arg) {
  PyNativeAlgo::Common::DumpGraphIR("replace_param.ir", ad_param()->tape_);
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

void ClearPyNativeAutoGradStaticRes() { pass_grad_graph_.clear(); }
}  // namespace autograd
}  // namespace pynative
}  // namespace mindspore
