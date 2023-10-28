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
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "frontend/expander/bprop/bprop.h"
#include "frontend/operator/composite/composite.h"
#include "frontend/optimizer/ad/dfunctor.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/primitive_utils.h"
#include "include/common/profiler.h"
#include "ir/anf.h"
#include "ir/func_graph_cloner.h"
#include "ops/array_ops.h"
#include "ops/framework_ops.h"
#include "ops/math_ops.h"
#include "ops/other_ops.h"
#include "ops/sequence_ops.h"
#include "ops/structure_ops.h"
#include "pipeline/jit/ps/action.h"
#include "pipeline/jit/ps/pass.h"
#include "pipeline/pynative/grad/jit/jit_call_graph.h"
#include "pipeline/pynative/pynative_utils.h"
#include "pipeline/pynative/grad/bprop_pass.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "utils/info.h"
#include "utils/profile.h"

namespace mindspore {
namespace pynative {
namespace autograd {
namespace {
enum class SpecialType { kZerosLikeType = 0, kOnesLikeType = 1 };
const mindspore::HashSet<std::string> kGradBlackList{kMakeTupleOpName,         kMakeListOpName,
                                                     kTupleGetItemOpName,      kStopGradientOpName,
                                                     kUpdateStateOpName,       kNPUAllocFloatStatusOpName,
                                                     kNPUGetFloatStatusOpName, kNPUClearFloatStatusOpName};

const mindspore::HashSet<std::string> kMonadOp = {kLoadOpName, kDependOpName, kUpdateStateOpName};

const mindspore::HashSet<std::string> kMetaFuncGraphOp{
  kPyExecuteOpName,
  kAttrMutableOpName,
  kMakeDictOpName,
};

static constexpr int kInputNum1 = 0;
static constexpr int kInputNum2 = 1;
static constexpr int kInputNum3 = 2;
mindspore::HashMap<std::string, FuncGraphPtr> pass_grad_graph_;
mindspore::HashMap<std::string, pipeline::ResourcePtr> jit_call_graph_compile_cache_;

inline bool IsPrimNeedGrad(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  return kGradBlackList.find(prim->name()) == kGradBlackList.end();
}

bool NeedGrad(const std::vector<ValuePtr> &input_values) {
  for (auto &input_arg : input_values) {
    MS_EXCEPTION_IF_NULL(input_arg);
    if (input_arg->isa<tensor::Tensor>()) {
      const auto &input_tensor = input_arg->cast<tensor::TensorPtr>();
      auto auto_grad_meta_data = input_tensor->auto_grad_meta_data();
      MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
      if (PyNativeAlgo::Common::IsParam(auto_grad_meta_data->grad_type())) {
        return true;
      }
      auto variable = auto_grad_meta_data->variable();
      if (variable != nullptr) {
        return true;
      }
    } else if (input_arg->isa<ValueSequence>()) {
      auto value_seq = input_arg->cast<ValueSequencePtr>()->value();
      if (NeedGrad(value_seq)) {
        return true;
      }
    } else if (input_arg->isa<tensor::COOTensor>() || input_arg->isa<tensor::CSRTensor>()) {
      return true;
    }
  }
  return false;
}

ValuePtr GetFakeZeroTensor() {
  static ValuePtr fake_v = std::make_shared<tensor::Tensor>(0);
  return fake_v;
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

AnfNodePtr BuildSparseTensorNode(const KernelGraphPtr &tape, const ValuePtr &sparse_value,
                                 const AnfNodePtr &dout_value_node) {
  MS_EXCEPTION_IF_NULL(tape);
  MS_EXCEPTION_IF_NULL(sparse_value);
  if (sparse_value->isa<tensor::CSRTensor>()) {
    auto csr_tensor = sparse_value->cast<tensor::CSRTensorPtr>();
    MS_EXCEPTION_IF_NULL(csr_tensor);
    auto indptr_node = PyNativeAlgo::Common::CreateValueNodeByValue(csr_tensor->GetIndptr());
    auto indices_node = PyNativeAlgo::Common::CreateValueNodeByValue(csr_tensor->GetIndices());
    auto value_shape = GetSparseTensorShapeNode(csr_tensor->shape());
    auto special_like_csr_node = tape->FuncGraph::NewCNode(
      {NewValueNode(prim::kPrimMakeTuple), indptr_node, indices_node, dout_value_node, value_shape});
    special_like_csr_node->set_abstract(sparse_value->ToAbstract()->Broaden());
    return special_like_csr_node;
  } else if (sparse_value->isa<tensor::COOTensor>()) {
    auto coo_tensor = sparse_value->cast<tensor::COOTensorPtr>();
    MS_EXCEPTION_IF_NULL(coo_tensor);
    auto indices_node = PyNativeAlgo::Common::CreateValueNodeByValue(coo_tensor->GetIndices());
    auto value_shape = GetSparseTensorShapeNode(coo_tensor->shape());
    auto special_like_coo_node =
      tape->FuncGraph::NewCNode({NewValueNode(prim::kPrimMakeTuple), indices_node, dout_value_node, value_shape});
    special_like_coo_node->set_abstract(sparse_value->ToAbstract()->Broaden());
    return special_like_coo_node;
  }
  MS_LOG(EXCEPTION) << "Get invalid sparse tensor";
}

AnfNodePtr BuildSpecialNode(const KernelGraphPtr &tape, const ValuePtr &value, const abstract::AbstractBasePtr &abs,
                            const SpecialType &type) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>()) {
    auto prim_node =
      (type == SpecialType::kZerosLikeType ? NewValueNode(std::make_shared<Primitive>(*prim::kPrimZerosLike))
                                           : NewValueNode(std::make_shared<Primitive>(*prim::kPrimOnesLike)));
    auto value_node = PyNativeAlgo::Common::CreateValueNodeByValue(value, abs);
    auto special_like_value = tape->FuncGraph::NewCNode({prim_node, value_node});
    special_like_value->set_abstract(value_node->abstract());
    return special_like_value;
  } else if (value->isa<ValueSequence>()) {
    auto tuple = value->cast<ValueSequencePtr>();
    abstract::AbstractSequencePtr abs_seq;
    if (abs == nullptr) {
      abs_seq =
        PyNativeAlgo::Common::SetAbstractValueToAnyValue(value->ToAbstract())->cast<abstract::AbstractSequencePtr>();
    } else {
      abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    }
    AnfNodePtrList args{NewValueNode(prim::kPrimMakeTuple)};
    for (size_t i = 0; i < tuple->size(); ++i) {
      AnfNodePtr special_like_value = BuildSpecialNode(tape, tuple->value()[i], abs_seq->elements()[i], type);
      (void)args.emplace_back(special_like_value);
    }
    auto special_like_value = tape->FuncGraph::NewCNode(args);
    special_like_value->set_abstract(abs_seq);
    return special_like_value;
  } else if (value->isa<Scalar>()) {
    auto fake_tensor = GetFakeZeroTensor();
    return BuildSpecialNode(tape, fake_tensor, nullptr, type);
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
  } else {
    MS_LOG(INFO) << "For value " << value->ToString() << ", the type is not tensor or scalar";
    return BuildSpecialNode(tape, GetFakeZeroTensor(), nullptr, type);
  }
}

void ClearDeviceAddress(const ValuePtr &value) {
  std::vector<tensor::TensorPtr> tensors;
  TensorValueToTensor(value, &tensors);
  for (const auto &tensor : tensors) {
    tensor->set_device_address(nullptr);
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
  } else if (value->isa<ValueSequence>()) {
    auto val_seq = value->cast<ValueSequencePtr>();
    return std::all_of(val_seq->value().begin(), val_seq->value().end(),
                       [](const ValuePtr &value) { return IsConstant(value); });
  } else if (value->isa<tensor::COOTensor>()) {
    auto coo_tensor = value->cast<tensor::COOTensorPtr>();
    return IsConstant(coo_tensor->GetIndices());
  } else if (value->isa<tensor::CSRTensor>()) {
    auto csr_tensor = value->cast<tensor::CSRTensorPtr>();
    return IsConstant(csr_tensor->GetIndices());
  }
  return true;
}

FuncGraphPtr OptimizeBpropBuilder(const FuncGraphPtr &bprop_func_graph, const GradParamPtr &grad_param) {
  PyNativeAlgo::Common::DumpGraphIR("bprop_builder_before_opt.ir", bprop_func_graph);
  pipeline::ResourcePtr resource = std::make_shared<pipeline::Resource>();
  resource->set_func_graph(bprop_func_graph);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(bprop_func_graph);
  auto after_opt_bg = pipeline::JitBpropGraphPass(resource, true);
  auto is_dynamic_shape_control_flow =
    grad_param->is_jit_graph && grad_param->use_dynamic_shape_process && grad_param->is_control_flow;
  if (is_dynamic_shape_control_flow) {
    for (const auto &g : manager->func_graphs()) {
      g->set_flag(kFlagJitCallGraph, true);
    }
  }
  auto abs_seq = after_opt_bg->parameters().empty()
                   ? nullptr
                   : after_opt_bg->parameters().back()->abstract()->cast<abstract::AbstractSequencePtr>();
  if (abs_seq != nullptr && !abs_seq->dynamic_len() && grad_param->is_jit_graph &&
      grad_param->use_dynamic_shape_process) {
    PyNativeAlgo::Common::ProcessTupleParam(after_opt_bg, after_opt_bg->parameters().size() - kIndex1);
  }
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
AnfNodePtr HandleRealToComplex(const TensorPtr &input, const AbstractBasePtr &abs, const AnfNodePtr &din,
                               const KernelGraphPtr &tape) {
  MS_EXCEPTION_IF_NULL(din);
  TypePtr din_type = din->Type();
  if (din_type == nullptr || !din_type->isa<TensorType>()) {
    return din;
  }
  din_type = din_type->cast_ptr<TensorType>()->element();
  MS_EXCEPTION_IF_NULL(din_type);
  // cppcheck-suppress unreadVariable
  if (MS_LIKELY(din_type->type_id() != kNumberTypeComplex64 && din_type->type_id() != kNumberTypeComplex128)) {
    return din;
  }

  MS_EXCEPTION_IF_NULL(input);
  TypePtr input_type = input->Dtype();
  if (input_type == nullptr) {
    return din;
  }
  if (input_type->type_id() == kNumberTypeComplex64 || input_type->type_id() == kNumberTypeComplex128) {
    return din;
  }

  AnfNodePtr new_din = tape->FuncGraph::NewCNode({NewValueNode(prim::kPrimReal), din});
  AbstractBasePtr real_abs =
    std::make_shared<abstract::AbstractTensor>(abstract::AbstractTensor(input_type, abs->GetShapeTrack()));
  new_din->set_abstract(real_abs);
  return new_din;
}

void SetJitCallGraph(const CNodePtr &cnode, const FuncGraphPtr &call_graph, const std::string &cache_key,
                     bool is_control_flow) {
  MS_EXCEPTION_IF_NULL(cnode);
  common::AnfAlgo::SetNodeAttr(kAttrJitCallNode, MakeValue(true), cnode);
  // kFlagJitCallGraph is set true to avoid compilig call_graph whe compiling the main graph
  call_graph->set_flag(kFlagJitCallGraph, true);
  // call graph not inline to grad top
  call_graph->set_flag(FUNC_GRAPH_FLAG_NO_INLINE, true);
  pipeline::ResourcePtr resource;
  const auto it = jit_call_graph_compile_cache_.find(cache_key);
  bool need_compile = (it == jit_call_graph_compile_cache_.end());
  if (need_compile) {
    resource = std::make_shared<pipeline::Resource>();
    resource->set_func_graph(call_graph);
    (void)jit_call_graph_compile_cache_.emplace(cache_key, resource);
  } else {
    resource = it->second;
  }
  MS_EXCEPTION_IF_NULL(resource);
  auto fn = [resource, need_compile, is_control_flow](const VectorRef &arg_list) -> VectorRef {
    if (need_compile) {
      MS_LOG(DEBUG) << "Start emit action for graph " << resource->func_graph()->ToString();
      auto manager = resource->manager();
      manager->AddFuncGraph(resource->func_graph(), true);
      resource->SetBackendAsync([]() { return compile::CreateBackend(); });
      resource->func_graph()->set_flag(kFlagIsPynativeBpropGraph, true);
      // kFlagJitCallGraph is set false to compile sub graph in control flow
      if (is_control_flow) {
        for (const auto &g : manager->func_graphs()) {
          g->set_flag(kFlagJitCallGraph, false);
        }
      }
      (void)TaskEmitAction(resource);
      (void)ExecuteAction(resource);
    }
    MS_LOG(DEBUG) << "Start execute action for graph " << resource->func_graph()->ToString();
    compile::VmEvalFuncPtr run = resource->GetResult(pipeline::kOutput).cast<compile::VmEvalFuncPtr>();
    return utils::cast<VectorRef>((*run)(arg_list));
  };
  cnode->set_user_data<JitCallGraph>(std::make_shared<JitCallGraph>(fn));
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
    AnfNodePtrList inputs{cnode->inputs().begin(), cnode->inputs().end() - 1};
    cnode->set_inputs(inputs);
  }
  MS_EXCEPTION_IF_NULL(grad_param);
  // Jit graph contain monad op
  if (grad_param->is_jit_graph) {
    for (size_t i = 1; i < cnode->size(); ++i) {
      cnode->set_input(i, common::AnfAlgo::VisitKernelWithReturnType(cnode->input(i), 0, false,
                                                                     {prim::kPrimTupleGetItem, prim::kPrimMakeTuple})
                            .first);
    }
  }
  return false;
}

AnfNodePtr GetTupleItemNodeInput(const KernelGraphPtr &tape, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(tape);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  AnfNodePtr new_cnode = nullptr;
  if (IsPrimitive(cnode->input(kIndex1), prim::kPrimTupleGetItem)) {
    auto inner_cnode = cnode->input(kIndex1)->cast<CNodePtr>();
    new_cnode = tape->FuncGraph::NewCNode(
      {inner_cnode->input(kIndex0), GetTupleItemNodeInput(tape, inner_cnode), inner_cnode->input(kIndex2)});
  } else {
    AnfNodePtrList new_inputs{cnode->inputs().begin(), cnode->inputs().end()};
    new_cnode = tape->FuncGraph::NewCNode(new_inputs);
  }
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->set_abstract(cnode->abstract());
  return new_cnode;
}

void SetGradInfoForInputs(const ValuePtr &value, const VariableAdjointPtr &variable, const ParameterPtr &param) {
  if (value->isa<tensor::Tensor>()) {
    const auto &input_tensor = value->cast<tensor::TensorPtr>();
    const auto &auto_grad_meta_data = input_tensor->auto_grad_meta_data();
    MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
    auto_grad_meta_data->set_variable(variable);
    auto_grad_meta_data->set_parameter(param);
  } else if (value->isa<tensor::COOTensor>()) {
    const auto &coo_tensor = value->cast<tensor::COOTensorPtr>();
    const auto &indices_tensor = coo_tensor->GetIndices();
    SetGradInfoForInputs(indices_tensor, variable, param);
  } else if (value->isa<tensor::CSRTensor>()) {
    const auto &csr_tensor = value->cast<tensor::CSRTensorPtr>();
    const auto &indices_tensor = csr_tensor->GetIndices();
    SetGradInfoForInputs(indices_tensor, variable, param);
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
    for (const auto &val : value_sequence->value()) {
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
    for (const auto &val : value_sequence->value()) {
      ClearGradMetaData(val);
    }
  }
}
}  // namespace

AnfNodePtr FunctionNode::HyperAdd(const AnfNodePtr &left_node, const AnfNodePtr &right_node) {
  MS_EXCEPTION_IF_NULL(left_node);
  MS_EXCEPTION_IF_NULL(right_node);

  if (IsZerosLikeNode(left_node)) {
    return right_node;
  }
  if (IsZerosLikeNode(right_node)) {
    return left_node;
  }
  if (!IsPrimitiveCNode(left_node, prim::kPrimMakeTuple)) {
    auto add_result = tape_->FuncGraph::NewCNode({NewValueNode(prim::kPrimAdd), left_node, right_node});
    add_result->set_abstract(right_node->abstract());
    return add_result;
  } else if (IsPrimitiveCNode(left_node, prim::kPrimMakeTuple) && IsPrimitiveCNode(right_node, prim::kPrimMakeTuple)) {
    auto left_cnode = left_node->cast<CNodePtr>();
    auto right_cnode = right_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(right_cnode);
    AnfNodePtrList inputs = {NewValueNode(prim::kPrimMakeTuple)};
    AbstractBasePtrList abs;
    for (size_t i = 1; i < left_cnode->size(); ++i) {
      auto add_result = HyperAdd(left_cnode->input(i), right_cnode->input(i));
      (void)abs.emplace_back(add_result->abstract());
      (void)inputs.emplace_back(add_result);
    }
    auto add_tuple = tape_->FuncGraph::NewCNode(inputs);
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
  (void)next_edges_.emplace_back(next_variable, din);
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
      << ", is_need_propagate: " << is_need_propagate_ << " is_leaf: " << is_leaf_;
  for (size_t i = 0; i < fn()->next_edges().size(); ++i) {
    auto last_variable = fn()->next_edges()[i].first;
    auto din = fn()->next_edges()[i].second;
    buf << ", next edge variable id: " << PyNativeAlgo::Common::GetIdByValue(last_variable->out_value_)
        << " din: " << din->DebugString();
  }
  return buf.str();
}

AutoGradCellImpl::AutoGradCellImpl(const std::vector<ValuePtr> &input_param_values, const AbstractBasePtrList &abs_list,
                                   size_t op_num_in_bprop_graph, const AsyncHqueuePtr &assist_queue, bool enable_async,
                                   bool grad_by_value)
    : ad_param_(std::make_shared<AdParam>()) {
  ad_param()->tape_->debug_info()->set_name("grad_top");
  MS_LOG(DEBUG) << "Start AutoGradCellImpl, input size: " << input_param_values.size();
  ad_param()->variable_adjoint_set_.reserve(op_num_in_bprop_graph);
  ad_param()->anfnode_to_variable_adjoint_.reserve(op_num_in_bprop_graph);
  ad_param()->users_.dout_user_.reserve(op_num_in_bprop_graph);
  weights_used_in_graph_.reserve(op_num_in_bprop_graph);

  for (size_t i = 0; i < input_param_values.size(); ++i) {
    auto input_parameter = ad_param()->fg_->add_parameter();
    input_parameter->set_abstract(abs_list[i]);
    input_parameter->set_name(input_parameter->UniqueName());
    TraceGuard trace_guard(std::make_shared<TraceCopy>(input_parameter->debug_info()));
    auto tape_parameter = ad_param()->tape_->add_parameter();
    tape_parameter->set_abstract(abs_list[i]);

    auto zeros_like_dout =
      BuildSpecialNode(ad_param()->tape_, GetFakeZeroTensor(), abs_list[i], SpecialType::kZerosLikeType);
    auto func_node = std::make_shared<FunctionNode>(ad_param()->tape_, zeros_like_dout);
    auto input_adjoint = std::make_shared<VariableAdjoint>(func_node, input_param_values[i], true);

    if (!input_param_values[i]->isa<ValueSequence>()) {
      SetGradInfoForInputs(input_param_values[i], input_adjoint, input_parameter);
    } else {
      input_adjoint->set_is_need_grad(false);
    }
    (void)cell_inputs_.emplace_back(input_parameter, input_adjoint);
    (void)ad_param()->variable_adjoint_set_.insert(input_adjoint);
  }

  assist_queue_ = assist_queue;
  enable_async_ = enable_async;
  grad_by_value_ = grad_by_value;
  device_target_ = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
}

bool AutoGradCellImpl::KPynativeOp(const GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(grad_param);

  auto &prim = grad_param->op_grad_info->op_prim;
  if (!IsPrimNeedGrad(prim) || (grad_by_value_ && !NeedGrad(grad_param->op_grad_info->input_value))) {
    MS_LOG(DEBUG) << "Prim " << prim->name() << " does not need to do op grad.";
    return true;
  }

  auto cloned_value = grad_param->op_grad_info->out_value;
  if (grad_param->op_grad_info->out_value->isa<ValueSequence>()) {
    cloned_value = ShallowCopyTensorValue(grad_param->op_grad_info->out_value);
    ClearDeviceAddress(cloned_value);
  }

  // construct zeroslike placeholder, if need use in bprop, we replace it in backprogate.
  AnfNodePtr dout = BuildSpecialNode(ad_param()->tape_, GetFakeZeroTensor(), grad_param->op_grad_info->out_abs,
                                     SpecialType::kZerosLikeType);
  auto fn = std::make_shared<FunctionNode>(ad_param()->tape_, dout);
  auto variable_adjoint = std::make_shared<VariableAdjoint>(fn, cloned_value);
  // Custom forward cnode no need record in bprop graph, because it is a flag cnode for run python. So just create
  // bprop_cut grad op is ok
  bool is_custom_prim =
    IsPrimitiveEquals(prim, prim::kPrimHookBackward) || IsPrimitiveEquals(prim, prim::kPrimCellBackwardHook);
  if (!grad_by_value_ && !is_custom_prim) {
    auto k_node = BuildKNode(NewValueNode(prim), grad_param, true);
    variable_adjoint->set_k_node(k_node);
    SetKNodeInfo(grad_param->op_grad_info->out_value, k_node, grad_param->op_grad_info->out_abs);
    need_do_manager_replace_ = true;
  }
  CNodePtr input_node = ConstructBpropGraphInput(grad_param, dout, variable_adjoint, is_custom_prim);
  MS_LOG(DEBUG) << "Construct input cnode: " << input_node->DebugString();
  // Gradient outputs
  std::vector<CNodePtr> outputs;
  if (!is_custom_prim) {
    auto ret = BpropExpander(&outputs, &ad_param()->users_).Run(input_node, grad_param->op_grad_info->input_value);
    // cppcheck-suppress unreadVariable
    if (MS_UNLIKELY(!ret || outputs.empty())) {
      MS_LOG(DEBUG) << "Expander has no bprop of this prim: " << prim->name();
      BuildCustomBpropCNode(input_node, prim, &outputs);
    }
  } else {
    BuildBPropCutCNode(input_node, prim, &outputs);
  }
  // cppcheck-suppress unreadVariable
  if (MS_UNLIKELY(outputs.empty())) {
    MS_LOG(DEBUG) << "This op has not custom bprop: " << prim->name();
    BuildFakeBpropCNode(input_node, &outputs);
    variable_adjoint->set_is_fake_bprop(true);
    variable_adjoint->set_fake_prim_name(prim->name());
  }
  (void)ad_param()->variable_adjoint_set_.insert(variable_adjoint);
  SetGradMetaData(grad_param->op_grad_info->out_value, variable_adjoint);

  if (enable_async_) {
    UpdateNextEdgesAsync(variable_adjoint, outputs, grad_param);
  } else {
    UpdateNextEdges(variable_adjoint, outputs, grad_param->op_grad_info->input_value,
                    grad_param->op_grad_info->input_abs, grad_by_value_);
  }
  return true;
}

bool AutoGradCellImpl::KPynativeWithFProp(const GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(grad_param);
  AnfNodePtrList args_node_list;
  CNodePtr bprop_cnode = nullptr;
  AnfNodePtr k_node = nullptr;
  AnfNodePtr dout = nullptr;
  if (grad_by_value_) {
    for (size_t i = 0; i < grad_param->input_size; ++i) {
      if (PyNativeAlgo::Common::IsParam(grad_param->op_grad_info->input_value_grad_type[i])) {
        auto parameter = MapParameter(grad_param->op_grad_info->input_value[i], grad_param->op_grad_info->input_abs[i]);
        if (parameter != nullptr) {
          (void)args_node_list.emplace_back(parameter);
          continue;
        }
      }
      // Valuenode, node
      const auto value_node = PyNativeAlgo::Common::CreateValueNodeByValue(
        grad_param->op_grad_info->input_value[i], grad_param->op_grad_info->input_abs[i]->Clone());
      auto cnode = PyNativeAlgo::Common::ConvertValueSequenceToMakeTuple(value_node, ad_param()->tape_);
      (void)args_node_list.emplace_back(cnode);
    }
    bprop_cnode = GetBpropGraphCNode(grad_param, args_node_list, &dout);
  } else {
    k_node = BuildKNode(NewValueNode(grad_param->source_fg), grad_param, false);
    BuildKNodeListForHighOrderGraph(grad_param->op_grad_info->input_value, grad_param->op_grad_info->input_abs,
                                    &args_node_list);
    bprop_cnode = GetBpropGraphCNode(grad_param, args_node_list, &dout);
  }
  auto fn = std::make_shared<FunctionNode>(ad_param()->tape_, dout);
  auto variable_adjoint = std::make_shared<VariableAdjoint>(fn, grad_param->op_grad_info->out_value);
  variable_adjoint->set_k_node(k_node);
  std::vector<CNodePtr> outputs;
  for (size_t i = 0; i < grad_param->input_size; ++i) {
    CNodePtr din = ad_param()->tape_->FuncGraph::NewCNode(
      {NewValueNode(prim::kPrimTupleGetItem), bprop_cnode, NewValueNode(SizeToLong(i))});
    din->set_abstract(grad_param->op_grad_info->input_abs[i]);
    (void)outputs.emplace_back(din);
  }
  UpdateNextEdges(variable_adjoint, outputs, grad_param->op_grad_info->input_value, grad_param->op_grad_info->input_abs,
                  grad_by_value_);
  (void)ad_param()->variable_adjoint_set_.insert(variable_adjoint);
  (void)ad_param()->anfnode_to_variable_adjoint_.insert(std::make_pair(grad_param->cnode, variable_adjoint));
  SetGradMetaData(grad_param->op_grad_info->out_value, variable_adjoint);
  return true;
}

CNodePtr AutoGradCellImpl::GetBPropCNode(const GradParamPtr &grad_param, const AnfNodePtrList &args,
                                         const FuncGraphPtr &bprop_graph, bool cache_hit, AnfNodePtr *const tape_dout) {
  AnfNodePtrList bprop_inputs(args.begin(), args.end());
  bool is_jit_dynamic_shape = grad_param->is_jit_graph && grad_param->use_dynamic_shape_process;
  // Save replace info in first time
  if (!cache_hit && is_jit_dynamic_shape && grad_param->has_added_v) {
    const auto &jit = PyNativeAlgo::Common::GetPyNativeExecutor()->grad_executor()->jit();
    jit->SaveForwardOutputTensorInfoInBpropGraph(bprop_graph);
  }

  // Call by tape_
  MS_EXCEPTION_IF_NULL(tape_dout);
  *tape_dout = BuildSpecialNode(ad_param()->tape_, GetFakeZeroTensor(), grad_param->op_grad_info->out_abs,
                                SpecialType::kZerosLikeType);
  if (is_jit_dynamic_shape && grad_param->op_grad_info->out_abs->isa<abstract::AbstractSequence>()) {
    auto abs_seq = grad_param->op_grad_info->out_abs->cast<abstract::AbstractSequencePtr>();
    // Dynamic len has no size current
    if (!abs_seq->dynamic_len()) {
      for (size_t i = 0; i < abs_seq->size(); ++i) {
        CNodePtr din = ad_param()->tape_->FuncGraph::NewCNode(
          {NewValueNode(prim::kPrimTupleGetItem), *tape_dout, NewValueNode(SizeToLong(i))});
        din->set_abstract(abs_seq->elements()[i]);
        (void)bprop_inputs.emplace_back(din);
        AddUser(*tape_dout, din, kIndex1);
      }
    }
  } else {
    (void)bprop_inputs.emplace_back(*tape_dout);
  }
  (void)bprop_inputs.insert(bprop_inputs.cbegin(), NewValueNode(bprop_graph));
  // get_bprop is a call node
  auto bprop_cnode = ad_param()->tape_->FuncGraph::NewCNode(bprop_inputs);
  bprop_cnode->set_abstract(bprop_graph->output()->abstract());
  if (is_jit_dynamic_shape) {
    SetJitCallGraph(bprop_cnode, bprop_graph, grad_param->graph_cache_key, grad_param->is_control_flow);
    ad_param()->tape_->set_flag(FUNC_GRAPH_FLAG_NO_INLINE, true);
  }
  // For replacing parameter and dout.
  for (size_t i = 1; i < bprop_inputs.size(); ++i) {
    AddUser(bprop_inputs[i], bprop_cnode, i);
  }
  return bprop_cnode;
}

CNodePtr AutoGradCellImpl::GetBpropGraphCNode(const GradParamPtr &grad_param, const AnfNodePtrList &args,
                                              AnfNodePtr *const tape_dout) {
  MS_EXCEPTION_IF_NULL(grad_param);
  if (grad_param->is_control_flow || grad_param->is_jit_self_dynamic_shape) {
    MS_LOG(DEBUG) << "Get control flow graph or dynamic shape";
    need_do_manager_replace_ = true;
    return GetBPropFromFProp(grad_param, args, tape_dout);
  }
  return GetBPropFromExpander(grad_param, args, tape_dout);
}

CNodePtr AutoGradCellImpl::GetBPropFromExpander(const GradParamPtr &grad_param, const AnfNodePtrList &args,
                                                AnfNodePtr *const tape_dout) {
  const auto it = pass_grad_graph_.find(grad_param->graph_cache_key);
  bool cache_hit = (it != pass_grad_graph_.end());
  auto ad_graph = GradFuncGraph(grad_param);
  return GetBPropCNode(grad_param, args, ad_graph, cache_hit, tape_dout);
}

CNodePtr AutoGradCellImpl::GetBPropFromFProp(const GradParamPtr &grad_param, const AnfNodePtrList &args,
                                             AnfNodePtr *const tape_dout) {
  FuncGraphPtr after_opt_fg = nullptr;
  // Find ad graph in cache
  const auto it = pass_grad_graph_.find(grad_param->graph_cache_key);
  bool cache_hit = (it != pass_grad_graph_.end());
  if (cache_hit) {
    MS_LOG(DEBUG) << "Get ad grad graph by cache";
    after_opt_fg = BasicClone(it->second);
  } else {
    MS_EXCEPTION_IF_NULL(grad_param);
    auto bprop_builder = std::make_shared<FuncGraph>();
    bprop_builder->debug_info()->set_name("bprop_builder");

    AnfNodePtrList fprop_app_inputs{NewValueNode(grad_param->fg)};
    for (const auto &arg : args) {
      auto param = bprop_builder->add_parameter();
      param->set_abstract(arg->abstract());
      (void)fprop_app_inputs.emplace_back(param);
    }
    auto fprop_app = bprop_builder->NewCNode(fprop_app_inputs);
    // Get bprop from fprop_fg, it is 2th output of fprop_fg
    auto get_bprop = bprop_builder->NewCNode(
      {NewValueNode(prim::kPrimTupleGetItem), fprop_app, NewValueNode(static_cast<int64_t>(kIndex1))});

    AnfNodePtrList node_list{get_bprop};
    auto dout = bprop_builder->add_parameter();
    dout->set_abstract(grad_param->op_grad_info->out_abs);
    (void)node_list.emplace_back(dout);
    auto call_bprop = bprop_builder->NewCNode(node_list);

    AnfNodePtrList actual_out{NewValueNode(prim::kPrimMakeTuple)};
    for (size_t i = 0; i < grad_param->input_size; ++i) {
      // Index 0 env, skip
      auto out =
        bprop_builder->NewCNode({NewValueNode(prim::kPrimTupleGetItem), call_bprop, NewValueNode(SizeToLong(i + 1))});
      (void)actual_out.emplace_back(out);
    }
    bprop_builder->set_output(bprop_builder->NewCNode(actual_out));
    // Call pass for optimize graph, such as inline
    after_opt_fg = OptimizeBpropBuilder(bprop_builder, grad_param);
    if (grad_param->is_jit_graph || !grad_param->use_dynamic_shape_process) {
      pass_grad_graph_[grad_param->graph_cache_key] = BasicClone(after_opt_fg);
    }
  }
  return GetBPropCNode(grad_param, args, after_opt_fg, cache_hit, tape_dout);
}

FuncGraphPtr AutoGradCellImpl::GradFuncGraph(const GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(grad_param);
  // Find ad graph in cache
  if (grad_param->is_jit_graph || !grad_param->use_dynamic_shape_process) {
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
    if (ad_param()->anfnode_to_variable_adjoint_.count(ad_param()->last_node_) == 0) {
      MS_LOG(EXCEPTION) << "Can not find last node" << ad_param()->last_node_->DebugString();
    }
    ad_param()->last_variable_ = ad_param()->anfnode_to_variable_adjoint_[ad_param()->last_node_];
    auto ad_graph_dout = ad_param()->tape_->add_parameter();
    ad_graph_dout->set_abstract(ad_param()->last_node_->abstract());
    ad_param()->last_variable_->fn()->UpdateAccumulativeDout(ad_graph_dout);
    (void)BackPropagate();
  } else {
    // Just have a return node
    auto ad_graph_dout = ad_param()->tape_->add_parameter();
    ad_graph_dout->set_abstract(grad_param->fg->output()->abstract());
  }

  AnfNodePtrList outputs{NewValueNode(prim::kPrimMakeTuple)};
  abstract::AbstractBasePtrList out_abs_list;
  for (const auto &node : grad_param->fg->parameters()) {
    (void)outputs.emplace_back(ad_param()->anfnode_to_variable_adjoint_.at(node)->RealDout());
    (void)out_abs_list.emplace_back(outputs.back()->abstract());
  }
  auto ad_graph_out = ad_param()->tape_->FuncGraph::NewCNode(outputs);
  ad_graph_out->set_abstract(std::make_shared<abstract::AbstractTuple>(out_abs_list));
  ad_param()->tape_->set_output(ad_graph_out);
  auto ad_graph = ad_param()->tape_;
  auto abs_seq = ad_graph->parameters().empty()
                   ? nullptr
                   : ad_graph->parameters().back()->abstract()->cast<abstract::AbstractSequencePtr>();
  if (abs_seq != nullptr && !abs_seq->dynamic_len() && grad_param->is_jit_graph &&
      grad_param->use_dynamic_shape_process) {
    auto manager = MakeManager();
    MS_EXCEPTION_IF_NULL(manager);
    manager->AddFuncGraph(ad_graph);
    PyNativeAlgo::Common::ProcessTupleParam(ad_graph, ad_graph->parameters().size() - kIndex1);
  }
  PyNativeAlgo::Common::DumpGraphIR("ad_output_graph.ir", ad_graph);

  // Save ad graph in cache
  if (grad_param->is_jit_graph || !grad_param->use_dynamic_shape_process) {
    pass_grad_graph_[grad_param->graph_cache_key] = BasicClone(ad_graph);
  }
  // Replace cnode with valuenode for reduce compute
  bool jit_by_value = grad_param->is_jit_graph && grad_by_value_;
  if (jit_by_value) {
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
  bool jit_by_value = grad_param->is_jit_graph && grad_by_value_;
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
    AnfNodePtrList cnode_inputs{std::make_shared<ValueNode>(prim)};
    auto input_value = GetInputArgs(cnode, &cnode_inputs);
    bprop_pass::ProcessAttrNode(ad_param()->tape_, cnode, &input_value, &cnode_inputs);
    if (IsPrimitiveEquals(prim, prim::kPrimMakeTuple) || IsPrimitiveEquals(prim, prim::kPrimMakeList)) {
      (void)BuildKNodeForMakeTuple(cnode);
      continue;
    } else if (IsPrimitiveEquals(prim, prim::kPrimTupleGetItem)) {
      (void)BuildKNodeForTupleGetItem(cnode);
      continue;
    }

    auto k_node = GetKnode(prim, cnode, cnode_inputs, jit_by_value);
    MS_LOG(DEBUG) << "Build knode " << k_node->DebugString();
    // Set out
    auto out = PyNativeAlgo::Common::CreatOutputTensorValueByAbstract(cnode->abstract());
    (void)cnode_inputs.emplace_back(k_node);
    // Set dout
    AnfNodePtr dout =
      BuildSpecialNode(ad_param()->tape_, GetFakeZeroTensor(), cnode->abstract(), SpecialType::kZerosLikeType);
    (void)cnode_inputs.emplace_back(dout);
    auto input_node = ad_param()->tape_->FuncGraph::NewCNode(cnode_inputs);
    input_node->set_abstract(cnode->abstract());

    std::vector<CNodePtr> outputs;
    // Get bprop by expander
    auto ret = BpropExpander(&outputs, &ad_param()->users_).Run(input_node);
    if (!ret || outputs.empty()) {
      // Get bprop by meta graph
      if (grad_param->is_jit_graph && kMetaFuncGraphOp.find(prim->name()) != kMetaFuncGraphOp.end()) {
        MS_LOG(DEBUG) << "Get bprop graph by meta function graph";
        ProcessMetaFuncGraphOp(grad_param, prim, cnode, input_value, out);
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
    AbstractBasePtrList input_abs;
    for (size_t i = 1; i < cnode->size(); ++i) {
      (void)input_abs.emplace_back(cnode->input(i)->abstract());
    }
    UpdateNextEdges(variable_adjoint, outputs, input_value, input_abs, grad_by_value_);
    SetGradMetaData(out, variable_adjoint);
    (void)ad_param()->anfnode_to_variable_adjoint_.insert(std::make_pair(node, variable_adjoint));
    (void)ad_param()->variable_adjoint_set_.insert(variable_adjoint);
  }
}

AnfNodePtr AutoGradCellImpl::GetKnode(const PrimitivePtr &prim, const CNodePtr &cnode,
                                      const AnfNodePtrList &cnode_inputs, bool jit_by_value) {
  if (IsPrimitiveEquals(prim, prim::kPrimMirror)) {
    return ad_param()->anfnode_to_variable_adjoint_.at(cnode->input(kIndex1))->k_node();
  } else {
    auto c_k_node = ad_param()->tape_->FuncGraph::NewCNode(cnode_inputs);
    c_k_node->set_abstract(cnode->abstract());
    // In jit, copy forward graph cnode info to bprop graph
    if (jit_by_value && cnode->forward().first != nullptr) {
      auto new_v_node = PyNativeAlgo::Common::CreateValueNodeByValue(cnode->forward().first->value(),
                                                                     cnode->forward().first->abstract());
      c_k_node->set_forward(new_v_node, cnode->forward().second);
      ad_param()->tape_->set_used_forward_nodes({c_k_node});
    }
    c_k_node->AddAttr(bprop_pass::kIsKNode, MakeValue(true));
    return c_k_node;
  }
}

void AutoGradCellImpl::ProcessMetaFuncGraphOp(const GradParamPtr &grad_param, const PrimitivePtr &prim,
                                              const CNodePtr &cnode, const ValuePtrList &input_value,
                                              const ValuePtr &out) {
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
  auto op_grad_info = std::make_shared<OpGradInfo>();
  op_grad_info->op_prim = prim;
  op_grad_info->input_value = input_value;
  op_grad_info->input_abs = args_abs_list;
  op_grad_info->out_value = out;
  op_grad_info->out_abs = cnode->abstract();
  op_grad_info->input_value_grad_type = grad_param->op_grad_info->input_value_grad_type;
  auto meta_graph_grad_param = std::make_shared<GradParam>(op_grad_info, grad_param->use_dynamic_shape_process);
  meta_graph_grad_param->is_jit_graph = true;
  // Set to control flow just let it go by ad::Grad, because grad_func_graph with no abstract
  meta_graph_grad_param->is_control_flow = true;
  meta_graph_grad_param->cnode = cnode;
  meta_graph_grad_param->fg = grad_func_graph;
  meta_graph_grad_param->graph_cache_key = grad_param->graph_cache_key;
  if (!KPynativeWithFProp(meta_graph_grad_param)) {
    MS_LOG(EXCEPTION) << "Failed to make meta graph, cnode info: " << cnode->DebugString();
  }
}

void AutoGradCellImpl::CreateParameterAdjoint(const GradParamPtr &grad_param) const {
  auto &graph_parameters = grad_param->fg->parameters();
  if (graph_parameters.size() != grad_param->input_size) {
    MS_LOG(EXCEPTION) << "Parameters size " << graph_parameters.size() << " is not equal to graph input size "
                      << grad_param->input_size;
  }
  for (size_t i = 0; i < graph_parameters.size(); ++i) {
    MS_LOG(DEBUG) << "Get param " << graph_parameters[i]->DebugString();
    ParameterPtr param = ad_param()->tape_->add_parameter();
    param->set_abstract(graph_parameters[i]->abstract());
    auto zeros_like_dout = BuildSpecialNode(ad_param()->tape_, GetFakeZeroTensor(), graph_parameters[i]->abstract(),
                                            SpecialType::kZerosLikeType);
    auto func_node = std::make_shared<FunctionNode>(ad_param()->tape_, zeros_like_dout);
    // Copy to avoid corrupt real input grad info.
    auto op_arg = PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(grad_param->op_grad_info->input_value[i]);
    ClearGradMetaData(op_arg);
    auto adjoint = std::make_shared<VariableAdjoint>(func_node, op_arg, true);
    adjoint->set_k_node(param);
    SetGradMetaData(op_arg, adjoint, graph_parameters[i]->cast<ParameterPtr>());
    (void)ad_param()->variable_adjoint_set_.insert(adjoint);
    (void)ad_param()->anfnode_to_variable_adjoint_.insert(std::make_pair(graph_parameters[i], adjoint));
  }
}

ValuePtrList AutoGradCellImpl::GetInputArgs(const CNodePtr &cnode, AnfNodePtrList *cnode_inputs) const {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(cnode_inputs);
  ValuePtrList input_value;
  for (size_t i = 1; i < cnode->size(); ++i) {
    const auto &input_node = cnode->input(i);
    const auto it = ad_param()->anfnode_to_variable_adjoint_.find(input_node);
    if (it != ad_param()->anfnode_to_variable_adjoint_.end()) {
      (void)cnode_inputs->emplace_back(it->second->k_node());
      (void)input_value.emplace_back(it->second->out_value());
      continue;
    }
    if (input_node->isa<ValueNode>()) {
      auto v_node = input_node->cast<ValueNodePtr>();
      auto v = v_node->value();
      if (v != nullptr && v->isa<tensor::Tensor>()) {
        const auto &t = v->cast<tensor::TensorPtr>();
        const auto &grad_meta = t->auto_grad_meta_data();
        // Jit forward graph has no parameters(input is tuple or constant), so input used in graph as valuenode, but it
        // is used by tape_ as parameter also
        if (grad_meta != nullptr && PyNativeAlgo::Common::IsParam(grad_meta->grad_type())) {
          auto new_tensor = std::make_shared<tensor::Tensor>(t->data_type(), t->shape(), t->data_ptr());
          new_tensor->set_device_address(t->device_address());
          v = new_tensor;
        }
      }
      (void)PyNativeAlgo::Common::SetValueGradInfo(v, nullptr, TensorGradType::kConstant);
      // In case of jit forward graph and pynative bprop graph used same valuenode
      auto new_v_node = PyNativeAlgo::Common::CreateValueNodeByValue(v, v_node->abstract());
      (void)cnode_inputs->emplace_back(new_v_node);
      (void)input_value.emplace_back(v);
    } else {
      // Make Fake value
      auto v = MakeValue(0);
      (void)cnode_inputs->emplace_back(PyNativeAlgo::Common::CreateValueNodeByValue(v, input_node->abstract()));
      (void)input_value.emplace_back(v);
      MS_LOG(DEBUG) << "Get input node " << input_node->DebugString();
    }
  }
  return input_value;
}

void AutoGradCellImpl::UpdateOutputNodeOfTopCell(const ValuePtr &sens_out) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative,
                                     runtime::ProfilerEvent::kPyNativeGradUpdateSens,
                                     runtime::ProfilerRecorder::kNoName, true);
  MS_EXCEPTION_IF_NULL(sens_out);
  MS_LOG(DEBUG) << "Real output of top cell is " << PyNativeAlgo::Common::GetIdByValue(sens_out);
  ad_param()->sens_value_ = sens_out;
  UpdateSensParameter(ad_param()->sens_value_);
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
  ReplacePrimalParameter(grad_attr.has_sens);
  PyNativeAlgo::Common::DumpGraphIR("before_final_opt.ir", ad_param()->tape_);
  // Clear weights grad info
  for (const auto &weight : weights) {
    weight->set_auto_grad_meta_data(nullptr);
  }
  return ad_param()->tape_;
}

CNodePtr AutoGradCellImpl::ConstructBpropGraphInput(const GradParamPtr &grad_param, const AnfNodePtr &dout,
                                                    const VariableAdjointPtr &variable_adjoint, bool is_custom_prim) {
  MS_EXCEPTION_IF_NULL(grad_param);
  AnfNodePtrList node_list;
  (void)node_list.emplace_back(NewValueNode(grad_param->op_grad_info->op_prim));
  if (grad_by_value_ || is_custom_prim) {
    for (size_t i = 0; i < grad_param->input_size; ++i) {
      if (PyNativeAlgo::Common::IsParam(grad_param->op_grad_info->input_value_grad_type[i])) {
        // To solve the input is a tuple like (parameter, ...)
        auto parameter = MapParameter(grad_param->op_grad_info->input_value[i], grad_param->op_grad_info->input_abs[i]);
        MS_EXCEPTION_IF_NULL(parameter);
        (void)node_list.emplace_back(parameter);
        continue;
      }
      // Node abstract obj may free, so v node abstract will be not correct
      (void)node_list.emplace_back(PyNativeAlgo::Common::CreateValueNodeByValue(
        grad_param->op_grad_info->input_value[i], grad_param->op_grad_info->input_abs[i]->Clone()));
    }
    // Set out
    (void)node_list.emplace_back(PyNativeAlgo::Common::CreateValueNodeByValue(grad_param->op_grad_info->out_value,
                                                                              grad_param->op_grad_info->out_abs));
  } else {
    // Input is a Parameter or cnode, not a value node
    BuildKNodeListFromPrimalCNode(grad_param->op_grad_info->input_value, grad_param->op_grad_info->input_abs,
                                  &node_list);
    // Set out
    MS_EXCEPTION_IF_NULL(variable_adjoint);
    (void)node_list.emplace_back(variable_adjoint->k_node());
  }
  // Set dout
  (void)node_list.emplace_back(dout);
  auto input_node = ad_param()->tape_->FuncGraph::NewCNode(node_list);
  return input_node;
}

void AutoGradCellImpl::BuildKNodeListFromPrimalCNode(const ValuePtrList &input_value,
                                                     const abstract::AbstractBasePtrList &input_abs,
                                                     AnfNodePtrList *const node_list) {
  for (size_t i = 0; i < input_value.size(); ++i) {
    (void)node_list->emplace_back(BuildKNodeForCNodeInput(input_value[i], input_abs[i]));
    MS_LOG(DEBUG) << "Get knode for input:  " << PyNativeAlgo::Common::GetIdByValue(input_value[i]);
  }
}

void AutoGradCellImpl::BuildKNodeListForHighOrderGraph(const ValuePtrList &input_value,
                                                       const abstract::AbstractBasePtrList &input_abs,
                                                       AnfNodePtrList *const node_list) {
  for (size_t i = 0; i < input_value.size(); ++i) {
    const auto knode = BuildKNodeForCNodeInput(input_value[i], input_abs[i]);
    // Convert value sequence to make tuple, so that finalpass can elimnate tuplegetitem.
    // BuildKnodeForTuplgeGetItem now do not support input is valuesequence.
    if (knode->isa<ValueNode>()) {
      auto value_node = knode->cast<ValueNodePtr>();
      (void)node_list->emplace_back(
        PyNativeAlgo::Common::ConvertValueSequenceToMakeTuple(value_node, ad_param()->tape_));
    } else {
      (void)node_list->emplace_back(knode);
    }

    MS_LOG(DEBUG) << "Get knode for input:  " << PyNativeAlgo::Common::GetIdByValue(input_value[i]);
  }
}

void AutoGradCellImpl::SetKNodeInfo(const ValuePtr &value, const AnfNodePtr &k_node, const AbstractBasePtr &out_abs) {
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    auto auto_grad_meta_data = tensor->auto_grad_meta_data();
    MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
    auto_grad_meta_data->set_k_node(k_node);
    (void)k_nodes_used_in_graph_.emplace_back(k_node);
  } else if (value->isa<ValueSequence>()) {
    const auto &value_sequence = value->cast<ValueSequencePtr>()->value();
    const auto &abs_seq = out_abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(abs_seq);
    if (value_sequence.size() != abs_seq->size()) {
      MS_LOG(EXCEPTION) << "Get value sequence size " << value_sequence.size() << " not equal to abstract size "
                        << abs_seq->size();
    }
    for (size_t i = 0; i < value_sequence.size(); ++i) {
      auto sub_k_node = ad_param()->tape_->FuncGraph::NewCNode(
        {NewValueNode(prim::kPrimTupleGetItem), k_node, NewValueNode(static_cast<int64_t>(i))});
      sub_k_node->set_abstract(abs_seq->elements()[i]);
      SetKNodeInfo(value_sequence[i], sub_k_node, abs_seq->elements()[i]);
    }
  }
}

AnfNodePtr AutoGradCellImpl::BuildKNode(const AnfNodePtr &prim, const GradParamPtr &grad_param, bool from_single_op) {
  MS_EXCEPTION_IF_NULL(grad_param);
  AnfNodePtrList node_list;
  (void)node_list.emplace_back(prim);
  for (size_t i = 0; i < grad_param->input_size; ++i) {
    (void)node_list.emplace_back(
      BuildKNodeForCNodeInput(grad_param->op_grad_info->input_value[i], grad_param->op_grad_info->input_abs[i]));
  }
  auto k_node = ad_param()->tape_->FuncGraph::NewCNode(node_list);
  k_node->set_abstract(grad_param->op_grad_info->out_abs);
  k_node->AddAttr(bprop_pass::kIsKNode, MakeValue(true));
  if (from_single_op && grad_param->out_used_in_bporp_graph) {
    auto v_node = PyNativeAlgo::Common::CreateValueNodeByValue(grad_param->op_grad_info->out_value,
                                                               grad_param->op_grad_info->out_abs);
    k_node->set_forward(v_node, "");
    ad_param()->tape_->set_used_forward_nodes({k_node});
  }
  MS_LOG(DEBUG) << "Build knode " << k_node->DebugString();
  return k_node;
}

AnfNodePtr AutoGradCellImpl::BuildKNodeForCNodeInput(const ValuePtr &input, const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(input);
  if (input->isa<tensor::Tensor>()) {
    const auto &tensor = input->cast<tensor::TensorPtr>();
    const auto &auto_grad_meta_data = tensor->auto_grad_meta_data();
    MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
    auto k_node = auto_grad_meta_data->k_node();
    if (k_node != nullptr) {
      return k_node;
    }
    if (PyNativeAlgo::Common::IsParam(auto_grad_meta_data->grad_type())) {
      return MapParameter(input, abs);
    }
  } else if (input->isa<ValueSequence>() && !IsConstant(input)) {
    AnfNodePtrList inputs;
    (void)inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    const auto &val_sequence = input->cast<ValueSequencePtr>()->value();
    const auto &abs_sequence = abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(abs_sequence);
    if (val_sequence.size() != abs_sequence->size()) {
      MS_LOG(EXCEPTION) << "Get value sequence size " << val_sequence.size() << " not equal to abstract size "
                        << abs_sequence->size();
    }
    for (size_t i = 0; i < val_sequence.size(); ++i) {
      (void)inputs.emplace_back(BuildKNodeForCNodeInput(val_sequence[i], abs_sequence->elements()[i]));
    }
    auto k_node = ad_param()->tape_->FuncGraph::NewCNode(inputs);
    k_node->set_abstract(abs);
    return k_node;
  }
  auto value_node = NewValueNode(input);
  value_node->set_abstract(abs);
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
  AnfNodePtrList inputs{NewValueNode(prim::kPrimMakeTuple)};
  ValuePtrList input_value;
  AbstractBasePtrList input_abs;
  for (size_t i = 1; i < cnode->size(); ++i) {
    (void)inputs.emplace_back(BuildKNodeForCNodeInput(cnode->input(i)));
    if (cnode->input(i)->isa<CNode>() || cnode->input(i)->isa<Parameter>()) {
      const auto input_adjoint_iter = ad_param()->anfnode_to_variable_adjoint_.find(cnode->input(i));
      if (input_adjoint_iter == ad_param()->anfnode_to_variable_adjoint_.end()) {
        MS_LOG(EXCEPTION) << "Cannot find input in adjoint map, inp: " << cnode->input(i)->DebugString();
      }
      (void)input_value.emplace_back(input_adjoint_iter->second->out_value());
      (void)input_abs.emplace_back(cnode->input(i)->abstract());
    } else {
      auto value_node = cnode->input(i)->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      (void)input_value.emplace_back(value_node->value());
      (void)input_abs.emplace_back(value_node->abstract());
    }
  }
  auto out_value = MakeValue(input_value);
  AnfNodePtr dout = BuildSpecialNode(ad_param()->tape_, out_value, input_node->abstract(), SpecialType::kZerosLikeType);
  auto fn = std::make_shared<FunctionNode>(ad_param()->tape_, dout);
  auto variable_adjoint = std::make_shared<VariableAdjoint>(fn, out_value);
  auto k_node = ad_param()->tape_->FuncGraph::NewCNode(inputs);
  k_node->set_abstract(input_node->abstract());
  variable_adjoint->set_k_node(k_node);
  // Create dout for maketuple
  std::vector<CNodePtr> make_tuple_dout;
  for (size_t i = 1; i < cnode->size(); ++i) {
    auto d = ad_param()->tape_->FuncGraph::NewCNode(
      {NewValueNode(prim::kPrimTupleGetItem), dout, NewValueNode(SizeToLong(i - 1))});
    d->set_abstract(cnode->input(i)->abstract());
    (void)make_tuple_dout.emplace_back(d);
    AddUser(dout, d, 1);
  }
  UpdateNextEdges(variable_adjoint, make_tuple_dout, input_value, input_abs, false);
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

  AnfNodePtrList inputs{NewValueNode(prim::kPrimTupleGetItem)};
  // Get make tuple knode
  (void)inputs.emplace_back(BuildKNodeForCNodeInput(tuple_item_cnode->input(kIndex1)));
  // Get index knode
  (void)inputs.emplace_back(BuildKNodeForCNodeInput(tuple_item_cnode->input(kIndex2)));
  auto k_node = ad_param()->tape_->FuncGraph::NewCNode(inputs);
  k_node->set_abstract(input_node->abstract());
  variable_adjoint->set_k_node(k_node);
  // Create dout for tuplegetitem
  AnfNodePtrList tuple_getitem_dout{NewValueNode(prim::kPrimMakeTuple)};
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
  CNodePtr tuple_getitem_dout_value = ad_param()->tape_->FuncGraph::NewCNode(tuple_getitem_dout);
  tuple_getitem_dout_value->set_abstract(tuple_item_cnode->input(kIndex1)->abstract());
  auto index_dout_value = BuildSpecialNode(ad_param()->tape_, index_value, tuple_item_cnode->input(kIndex1)->abstract(),
                                           SpecialType::kZerosLikeType)
                            ->cast<CNodePtr>();
  UpdateNextEdges(variable_adjoint, {tuple_getitem_dout_value, index_dout_value}, {v_tuple, index_value},
                  {tuple_item_cnode->input(kIndex1)->abstract(), tuple_item_cnode->input(kIndex2)->abstract()}, false);
  AddUser(dout, tuple_getitem_dout_value, index_value_int + 1);
  (void)ad_param()->anfnode_to_variable_adjoint_.insert(std::make_pair(input_node, variable_adjoint));
  (void)ad_param()->variable_adjoint_set_.insert(variable_adjoint);
  return k_node;
}

void AutoGradCellImpl::UpdateNextEdgesAsync(const VariableAdjointPtr &variable, const std::vector<CNodePtr> &dins,
                                            const GradParamPtr &grad_param) {
  auto task = [this, variable, dins, grad_param]() {
    this->UpdateNextEdges(variable, dins, grad_param->op_grad_info->input_value, grad_param->op_grad_info->input_abs,
                          grad_by_value_);
  };
  bool success = assist_queue_->Push(new (std::nothrow) BpropTask(std::move(task)));
  if (!success) {
    assist_queue_->CheckException();
  }
}

void AutoGradCellImpl::UpdateNextEdges(const VariableAdjointPtr &variable, const std::vector<CNodePtr> &dins,
                                       const ValuePtrList &input_value, const abstract::AbstractBasePtrList &abs,
                                       bool grad_by_value) {
  size_t input_size = input_value.size();
  if (dins.size() != input_size) {
    MS_LOG(EXCEPTION) << "The size of dins " << dins.size() << " is not same as input_value " << input_size;
  }
  const auto &fn = variable->fn();
  for (size_t i = 0; i < input_size; ++i) {
    auto din = dins[i];
    MS_LOG(DEBUG) << "Input arg id: " << PyNativeAlgo::Common::GetIdByValue(input_value[i]) << ", din "
                  << din->DebugString();
#ifndef ENABLE_TEST
    // VM no need run pass
    din = bprop_pass::ConvertConstInputToAttr(din, device_target_, false, grad_by_value);
    bprop_pass::ConvertValueNodeValueToTensor(din);
#endif
    UpdateNextEdge(fn, din, input_value[i], abs[i]);
  }
  if (fn->next_edges().empty()) {
    variable->set_is_need_grad(false);
  }
  MS_LOG(DEBUG) << "Finish update next edges for variable: " << variable->ToString();
}

void AutoGradCellImpl::UpdateNextEdge(const FunctionNodePtr &fn, const AnfNodePtr &din, const ValuePtr &input_arg,
                                      const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(din);
  MS_EXCEPTION_IF_NULL(input_arg);
  if (input_arg->isa<tensor::Tensor>()) {
    const auto &input_tensor = input_arg->cast<tensor::TensorPtr>();
    auto auto_grad_meta_data = input_tensor->auto_grad_meta_data();
    MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
    auto variable = auto_grad_meta_data->variable();
    if (variable == nullptr || !variable->is_need_grad()) {
      return;
    }
    auto real_din = HandleRealToComplex(input_tensor, abs, din, fn->tape());
    auto new_din =
      TraceShape(fn, variable->out_value(), variable->fn()->accumulate_dout()->abstract(), input_tensor, real_din);
    fn->AddNextEdge(variable, new_din);
  } else if (input_arg->isa<ValueSequence>()) {
    auto value_seq = input_arg->cast<ValueSequencePtr>()->value();
    const auto &abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(abs_seq);
    if (value_seq.size() != abs_seq->size()) {
      MS_LOG(EXCEPTION) << "Get value sequence size " << value_seq.size() << " not equal to abstract size "
                        << abs_seq->size();
    }
    for (size_t i = 0; i < value_seq.size(); ++i) {
      auto sub_value = value_seq[i];
      CNodePtr new_din = ad_param()->tape_->FuncGraph::NewCNode(
        {NewValueNode(prim::kPrimTupleGetItem), din, NewValueNode(SizeToLong(i))});
      new_din->set_abstract(PyNativeAlgo::Common::SetAbstractValueToAnyValue(abs_seq->elements()[i]));
      if (din == fn->fake_dout()) {
        // The new_din's index input is fn->fake_dout()
        LazyAddUser(fn->fake_dout(), new_din, 1);
      }
      // Add next edge to fn
      UpdateNextEdge(fn, new_din, sub_value, abs_seq->elements()[i]);
    }
  } else if (input_arg->isa<tensor::COOTensor>()) {
    auto input_tensor = input_arg->cast<tensor::COOTensorPtr>()->GetIndices();
    UpdateNextEdge(fn, din, input_tensor, PyNativeAlgo::Common::SetAbstractValueToAnyValue(input_tensor->ToAbstract()));
  } else if (input_arg->isa<tensor::CSRTensor>()) {
    auto input_tensor = input_arg->cast<tensor::CSRTensorPtr>()->GetIndices();
    UpdateNextEdge(fn, din, input_tensor, PyNativeAlgo::Common::SetAbstractValueToAnyValue(input_tensor->ToAbstract()));
  } else {
    MS_LOG(DEBUG) << "It is not tensor, not need derivation " << input_arg->ToString();
    return;
  }
}

AbstractBasePtr AutoGradCellImpl::BuildForwardLastNode() {
  MS_LOG(DEBUG) << "Process last node info " << PyNativeAlgo::Common::GetIdByValue(ad_param()->sens_value_);
  auto zeros_like_node =
    BuildSpecialNode(ad_param()->tape_, ad_param()->sens_value_, nullptr, SpecialType::kZerosLikeType);
  auto fn = std::make_shared<FunctionNode>(ad_param()->tape_, zeros_like_node);
  auto sens_variable = std::make_shared<VariableAdjoint>(fn, ad_param()->sens_value_);
  if (ad_param()->sens_value_->isa<tensor::Tensor>()) {
    const auto &sens_tensor = ad_param()->sens_value_->cast<tensor::TensorPtr>();
    const auto &auto_grad_meta_data = sens_tensor->auto_grad_meta_data();
    MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
    if (PyNativeAlgo::Common::IsConstant(auto_grad_meta_data->grad_type())) {
      sens_variable->set_is_need_grad(false);
    }
  }
  UpdateNextEdge(fn, zeros_like_node, ad_param()->sens_value_, fn->accumulate_dout()->abstract());
  (void)ad_param()->variable_adjoint_set_.insert(sens_variable);
  ad_param()->last_variable_ = sens_variable;
  return fn->accumulate_dout()->abstract();
}

ParameterPtr AutoGradCellImpl::CreateTapeParameter(const tensor::TensorPtr &tensor,
                                                   const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(abs);
  auto param = ad_param()->fg_->add_parameter();
  param->set_abstract(abs);
  if (tensor->is_parameter()) {
    param->set_default_param(tensor);
  }
  auto auto_grad_meta_data = tensor->auto_grad_meta_data();
  if (auto_grad_meta_data == nullptr) {
    auto_grad_meta_data = std::make_shared<AutoGradMetaData>();
    tensor->set_auto_grad_meta_data(auto_grad_meta_data);
  }
  auto_grad_meta_data->set_grad_type(TensorGradType::kParameter);
  auto_grad_meta_data->set_parameter(param);
  return param;
}

ParameterPtr AutoGradCellImpl::AddParameterNode(const tensor::TensorPtr &tensor, const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(tensor);
  auto param = CreateTapeParameter(tensor, abs);
  auto zeros_like_dout =
    BuildSpecialNode(ad_param()->tape_, GetFakeZeroTensor(), param->abstract(), SpecialType::kZerosLikeType);
  auto func_node = std::make_shared<FunctionNode>(ad_param()->tape_, zeros_like_dout);
  auto input_adjoint = std::make_shared<VariableAdjoint>(func_node, tensor, true);
  (void)ad_param()->variable_adjoint_set_.insert(input_adjoint);
  auto auto_grad_meta_data = tensor->auto_grad_meta_data();
  MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
  auto_grad_meta_data->set_variable(input_adjoint);
  (void)weights_used_in_graph_.emplace_back(param);
  return param;
}

AnfNodePtrList AutoGradCellImpl::ExtractParamters(const tensor::TensorPtrList &weights) const {
  AnfNodePtrList params;
  for (const auto &weight : weights) {
    auto parameter = ExtractParameter(weight);
    MS_EXCEPTION_IF_NULL(parameter);
    (void)params.emplace_back(std::move(parameter));
  }
  return params;
}

void AutoGradCellImpl::UpdateSensParameter(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>()) {
    const auto &sens_tensor = value->cast<tensor::TensorPtr>();
    const auto &auto_grad_meta_data = sens_tensor->auto_grad_meta_data();
    MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
    const auto variable = auto_grad_meta_data->variable();
    // Return input parameter or weight parameter for net, if v is parameter just entry once
    if (PyNativeAlgo::Common::IsParam(auto_grad_meta_data->grad_type()) && variable == nullptr) {
      (void)AddParameterNode(sens_tensor, PyNativeAlgo::Common::SetAbstractValueToAnyValue(sens_tensor->ToAbstract()));
    }
  } else if (value->isa<ValueSequence>()) {
    const auto &value_seq = value->cast<ValueSequencePtr>()->value();
    for (const auto &v : value_seq) {
      UpdateSensParameter(v);
    }
  }
}

AnfNodePtr AutoGradCellImpl::MapParameter(const ValuePtr &value, const abstract::AbstractBasePtr &abs) {
  if (value->isa<tensor::Tensor>()) {
    const auto &tensor = value->cast<tensor::TensorPtr>();
    const auto &auto_grad_meta_data = tensor->auto_grad_meta_data();
    MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
    const auto &param = auto_grad_meta_data->parameter();
    if (param != nullptr) {
      // In dynamic shape scenario, abs my be need change
      param->set_abstract(abs);
      return param;
    }
    if (auto_grad_meta_data->grad_type() == TensorGradType::kParameter) {
      return AddParameterNode(tensor, abs);
    }
    return PyNativeAlgo::Common::CreateValueNodeByValue(value, abs);
  } else if (value->isa<ValueSequence>()) {
    const auto &val_seq = value->cast<ValueSequencePtr>()->value();
    const auto &abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(abs_seq);
    if (val_seq.size() != abs_seq->size()) {
      MS_LOG(EXCEPTION) << "Get value sequence size " << val_seq.size() << " not equal to abstract size "
                        << abs_seq->size();
    }
    AnfNodePtrList inputs;
    (void)inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    for (size_t i = 0; i < val_seq.size(); ++i) {
      (void)inputs.emplace_back(MapParameter(val_seq[i], abs_seq->elements()[i]));
    }
    auto cnode = ad_param()->tape_->FuncGraph::NewCNode(inputs);
    // For replacing fg parameter by user
    for (size_t i = 1; i < inputs.size(); ++i) {
      AddUser(inputs[i], cnode, i);
    }
    cnode->set_abstract(abs);
    return cnode;
  } else if (value->isa<tensor::COOTensor>()) {
    const auto &coo_tensor = value->cast<tensor::COOTensorPtr>();
    return MapParameter(coo_tensor->GetIndices(), abs);
  } else if (value->isa<tensor::CSRTensor>()) {
    const auto &csr_tensor = value->cast<tensor::CSRTensorPtr>();
    return MapParameter(csr_tensor->GetIndices(), abs);
  } else {
    return PyNativeAlgo::Common::CreateValueNodeByValue(value, abs);
  }
}

ParameterPtr AutoGradCellImpl::ExtractParameter(const tensor::TensorPtr &tensor) const {
  MS_EXCEPTION_IF_NULL(tensor);
  const auto &auto_grad_meta_data = tensor->auto_grad_meta_data();
  if (auto_grad_meta_data != nullptr && PyNativeAlgo::Common::IsParam(auto_grad_meta_data->grad_type())) {
    return auto_grad_meta_data->parameter();
  }
  return nullptr;
}

AnfNodePtr AutoGradCellImpl::TraceShape(const FunctionNodePtr &fn, const ValuePtr &out_value,
                                        const abstract::AbstractBasePtr &out_abs, const TensorPtr &input_tensor,
                                        const AnfNodePtr &din) {
  MS_EXCEPTION_IF_NULL(out_value);
  MS_EXCEPTION_IF_NULL(input_tensor);
  MS_EXCEPTION_IF_NULL(din);

  // The node corresponding output tensor is the same as the currently used tensor
  if (out_value->isa<tensor::Tensor>()) {
    // out_value is be used, may be it is one of multiple output
    auto out_tensor = out_value->cast<tensor::TensorPtr>();
    if (input_tensor->id() == out_tensor->id()) {
      return din;
    }
    return BuildSpecialNode(ad_param()->tape_, out_value, out_abs, SpecialType::kZerosLikeType);
  } else if (out_value->isa<ValueSequence>()) {
    // The corresponding output of node is ValueSequence, but used one of it
    AnfNodePtrList inputs;
    (void)inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    auto value_seq = out_value->cast<ValueSequencePtr>();
    auto abs_seq = out_abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(abs_seq);
    int index = -1;
    for (size_t i = 0; i < value_seq->size(); ++i) {
      // Find the value's din, if value equal to sub_value, means value be used, is it will get din; Otherwise value's
      // din is zero , which set by second branch condition above
      auto new_din = TraceShape(fn, value_seq->value()[i], abs_seq->elements()[i], input_tensor, din);
      (void)inputs.emplace_back(new_din);

      // if exist din == fake_dout, we record it in user vector
      if (din == fn->fake_dout() && new_din == din) {
        index = static_cast<int>(inputs.size()) - 1;
      }
    }
    auto new_din = ad_param()->tape_->FuncGraph::NewCNode(inputs);
    new_din->set_abstract(out_abs);
    if (index != -1) {
      LazyAddUser(fn->fake_dout(), new_din, index);
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
  AnfNodePtrList inputs{NewValueNode(bprop_cut)};
  // Get input, get output, get dout
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    (void)inputs.emplace_back(cnode->input(i));
  }
  auto bprop_cut_cnode = ad_param()->tape_->FuncGraph::NewCNode(inputs);

  size_t input_num = cnode->size() - 2;
  AbstractBasePtrList abs_list;
  for (size_t i = 1; i < cnode->size(); ++i) {
    // bprop_cut_cnode ith input used cnode->input(i)
    AddUser(cnode->input(i), bprop_cut_cnode, i);
    if (i < input_num) {
      auto din = ad_param()->tape_->FuncGraph::NewCNode(
        {NewValueNode(prim::kPrimTupleGetItem), bprop_cut_cnode, NewValueNode(static_cast<int64_t>(i - 1))});
      MS_EXCEPTION_IF_NULL(cnode->input(i)->abstract());
      din->set_abstract(cnode->input(i)->abstract());
      (void)abs_list.emplace_back(cnode->input(i)->abstract());
      (void)outputs->emplace_back(din);
    }
  }
  bprop_cut_cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abs_list));
  ad_param()->tape_->set_flag(kFlagEnableRunGraphBySingleOp, true);
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
  const auto &sens_abstract = BuildForwardLastNode();
  ParameterPtr sens_param = nullptr;
  if (has_sens_arg) {
    sens_param = ad_param()->tape_->add_parameter();
    sens_param->set_name(sens_param->UniqueName());
    sens_param->debug_info()->set_name("sens");
    sens_param->set_abstract(sens_abstract);
  }
  // Update dout for dout
  MS_EXCEPTION_IF_NULL(ad_param()->last_variable_);
  if (ad_param()->last_variable_->is_need_grad()) {
    if (has_sens_arg) {
      ad_param()->last_variable_->fn()->UpdateAccumulativeDout(sens_param);
    } else {
      ad_param()->last_variable_->fn()->UpdateAccumulativeDout(
        BuildSpecialNode(ad_param()->tape_, ad_param()->sens_value_, sens_abstract, SpecialType::kOnesLikeType));
    }
  }
  // Add weights parameter
  need_grad_weights_.reserve(weights.size());
  for (const auto &weight_tensor : weights) {
    (void)need_grad_weights_.emplace(weight_tensor->id());
    UpdateTapeParameter(weight_tensor);
  }
  for (auto &weight : weights_used_in_graph_) {
    auto tensor = PyNativeAlgo::Common::GetTensorFromParam(weight);
    MS_EXCEPTION_IF_NULL(tensor);
    if (need_grad_weights_.find(tensor->id()) == need_grad_weights_.end()) {
      UpdateTapeParameter(tensor);
    }
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
  UpdateLazyUser();
  const auto &last_node_reverse_iter = GetLastNodeReverseIter();
  SeenNum seen = NewSeenGeneration();
  for (auto iter = last_node_reverse_iter; iter != ad_param()->variable_adjoint_set_.rend(); ++iter) {
    const auto &variable = *iter;
    if (!variable->is_need_propagate() || !variable->is_need_grad()) {
      MS_LOG(DEBUG) << "No need grad, variable is: " << variable->ToString();
      continue;
    }
    if (static_cast<bool>(MS_UNLIKELY(variable->is_fake_bprop()))) {
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
    Replace(fn->fake_dout(), fn->accumulate_dout(), &ad_param()->users_.dout_user_, true);
    // replace edges which exist fake dout
    fn->ReplaceEdges();
    MS_LOG(DEBUG) << "Begin backpropagate: " << variable->ToString();
    const auto &next_edges = fn->next_edges();
    for (const auto &next_edge : next_edges) {
      const auto &last_variable = next_edge.first;
      const auto &din = next_edge.second;
      bprop_pass::ConvertMakeTupleInputToDynamicInput(din, seen, this);
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
      if (!IsValidTensorInput(cell_inputs_[index].first->abstract())) {
        MS_LOG(DEBUG) << "Get input node is not tensor "
                      << ", abs " << cell_inputs_[index].first->abstract()->ToString();
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
  auto input_grad_ret = ad_param()->tape_->FuncGraph::NewCNode(inputs_grad_list);
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
    for (const auto &weight : weights) {
      auto grad_node = GetGradNodeByIndex(weight);
      MS_EXCEPTION_IF_NULL(grad_node);
      (void)weights_grad_list.emplace_back(grad_node);
      (void)weights_grad_spec.emplace_back(grad_node->abstract());
    }
    auto weight_grad_ret = ad_param()->tape_->FuncGraph::NewCNode(weights_grad_list);
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
        ad_param()->tape_->FuncGraph::NewCNode({NewValueNode(prim::kPrimMakeTuple), inputs_grad_ret, weights_grad_ret});
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
    tape_output = ad_param()->tape_->FuncGraph::NewCNode({NewValueNode(prim::kPrimMakeTuple)});
    abstract::AbstractBasePtrList abs{};
    tape_output->set_abstract(std::make_shared<abstract::AbstractTuple>(abs));
  } else {
    // If there are input nodes, return gradient of first input node.
    // Tuple, List, scalar will be ignore
    if (IsValidTensorInput(cell_inputs_[0].first->abstract())) {
      tape_output = cell_inputs_[0].second->RealDout();
    } else {
      MS_LOG(DEBUG) << "Get first input node is not tensor " << cell_inputs_[0].second->out_value()->ToString();
      tape_output = NewValueNode(kNull);
      tape_output->set_abstract(nullptr);
    }
  }
  ad_param()->tape_->set_output(tape_output);
}

void AutoGradCellImpl::LazyAddUser(const AnfNodePtr &node, const CNodePtr &user, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(user);
  (void)ad_param()->lazy_user_data_.emplace_back(std::make_tuple(node, user, index));
}

void AutoGradCellImpl::UpdateLazyUser() {
  // For lazy add user data, we need emplace to user.
  for (const auto &user_data : ad_param()->lazy_user_data_) {
    AddUser(std::get<kInputNum1>(user_data), std::get<kInputNum2>(user_data), std::get<kInputNum3>(user_data));
  }
}

void AutoGradCellImpl::AddUser(const AnfNodePtr &node, const CNodePtr &user, size_t index) {
  MS_EXCEPTION_IF_NULL(ad_param_);
  (void)ad_param()->users_.dout_user_[node].emplace_back(user, index);
}

void AutoGradCellImpl::AddTupleGetItemUser(const AnfNodePtr &node, const CNodePtr &user, size_t index) {
  MS_EXCEPTION_IF_NULL(ad_param_);
  (void)ad_param()->users_.tuple_getitem_user_[node].emplace_back(user, index);
}

void AutoGradCellImpl::Replace(const AnfNodePtr &old_node, const AnfNodePtr &new_node, UserType *user,
                               bool need_update) {
  MS_EXCEPTION_IF_NULL(ad_param_);
  if ((*user).find(old_node) == (*user).end()) {
    return;
  }
  const auto &old_node_users = (*user)[old_node];
  for (const auto &pair_node : old_node_users) {
    auto cnode = pair_node.first.lock();
    if (cnode == nullptr) {
      continue;
    }
    size_t index = pair_node.second;
    if (index >= cnode->size()) {
      // After convert attr cnode input will less
      if (auto v = cnode->GetAttr(kAttrConvertAttrNode); v != nullptr) {
        index -= GetValue<size_t>(v);
      } else {
        MS_LOG(EXCEPTION) << "exception for index: " << index << "greater than cnode size: " << cnode->size();
      }
    }
    cnode->set_input(index, new_node);
    if (need_update && IsPrimitiveCNode(new_node, prim::kPrimTupleGetItem)) {
      AddTupleGetItemUser(new_node, cnode, index);
    }
  }
}

void AutoGradCellImpl::ElimateTupleGetItem() {
  for (auto &user : ad_param()->users_.tuple_getitem_user_) {
    auto old_node = user.first;
    auto old_cnode = old_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(old_cnode);
    auto tuple_node = old_cnode->input(kIndex1);
    if (!IsPrimitiveCNode(tuple_node, prim::kPrimMakeTuple)) {
      continue;
    }
    auto index_value = GetValueNode<Int64ImmPtr>(old_cnode->input(kIndex2));
    size_t index = LongToSize(index_value->value());
    auto tuple_cnode = tuple_node->cast<CNodePtr>();
    Replace(old_node, tuple_cnode->input(index + 1), &ad_param()->users_.tuple_getitem_user_);
  }
}

void AutoGradCellImpl::DoParameterReplaceByManager(bool has_sens_arg) {
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
  for (size_t i = weight_offset; i < parameters.size(); ++i) {
    auto tensor = PyNativeAlgo::Common::GetTensorFromParam(parameters[i]);
    MS_EXCEPTION_IF_NULL(tensor);
    auto parameter = ExtractParameter(tensor);
    MS_EXCEPTION_IF_NULL(parameter);
    (void)tr.Replace(parameter, parameters[i]);
  }
  tr.Commit();
}

void AutoGradCellImpl::DoParameterReplaceByUser(bool has_sens_arg) {
  const auto &parameters = ad_param()->tape_->parameters();
  auto cell_inputs_size = cell_inputs_.size();
  for (size_t i = 0; i < cell_inputs_size; ++i) {
    Replace(cell_inputs_[i].first, parameters[i], &ad_param()->users_.dout_user_);
  }
  size_t weight_offset = cell_inputs_size;
  if (has_sens_arg) {
    weight_offset = weight_offset + 1;
  }
  for (size_t i = weight_offset; i < parameters.size(); ++i) {
    auto tensor = PyNativeAlgo::Common::GetTensorFromParam(parameters[i]);
    MS_EXCEPTION_IF_NULL(tensor);
    auto parameter = ExtractParameter(tensor);
    MS_EXCEPTION_IF_NULL(parameter);
    Replace(parameter, parameters[i], &ad_param()->users_.dout_user_);
  }
}

void AutoGradCellImpl::ReplacePrimalParameter(bool has_sens_arg) {
  PyNativeAlgo::Common::DumpGraphIR("replace_param.ir", ad_param()->tape_);
  if (need_do_manager_replace_) {
    MS_LOG(DEBUG) << "Do parameter replace by manager.";
    DoParameterReplaceByManager(has_sens_arg);
    need_do_manager_replace_ = false;
  } else {
    MS_LOG(DEBUG) << "Do parameter replace by user.";
    DoParameterReplaceByUser(has_sens_arg);
  }
  ElimateTupleGetItem();
}

void AutoGradCellImpl::UpdateTapeParameter(const tensor::TensorPtr &tensor) {
  auto p = ad_param()->tape_->add_parameter();
  auto param = ExtractParameter(tensor);
  if (param == nullptr) {
    param = CreateTapeParameter(tensor, PyNativeAlgo::Common::SetAbstractValueToAnyValue(tensor->ToAbstract()));
  }
  MS_EXCEPTION_IF_NULL(param);
  const auto &param_info = tensor->param_info();
  if (param_info != nullptr) {
    const auto &param_name = param_info->name();
    p->set_name(param_name);
    p->debug_info()->set_name(param_name);
  }
  TraceGuard trace_guard(std::make_shared<TraceCopy>(p->debug_info()));
  p->set_default_param(tensor);
  p->set_abstract(PyNativeAlgo::Common::SetAbstractValueToAnyValue(tensor->ToAbstract()));
}

void ClearPyNativeAutoGradStaticRes() {
  pass_grad_graph_.clear();
  jit_call_graph_compile_cache_.clear();
  bprop_pass::ClearCache();
}
}  // namespace autograd
}  // namespace pynative
}  // namespace mindspore
