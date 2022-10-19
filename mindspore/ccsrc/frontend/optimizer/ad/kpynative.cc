/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include "mindspore/core/ops/core_ops.h"
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "ir/anf.h"
#include "frontend/optimizer/ad/prim_bprop_optimizer.h"
#include "frontend/optimizer/ad/adjoint.h"
#include "frontend/optimizer/ad/dfunctor.h"
#include "frontend/optimizer/ad/kpynative.h"
#include "frontend/operator/ops.h"
#include "utils/info.h"
#include "include/common/debug/anf_ir_dump.h"
#include "pipeline/jit/debug/trace.h"

namespace mindspore {
namespace ad {
using CacheKey = std::pair<std::string, size_t>;

static KPrim g_k_prims_pynative;
static ValuePtr add_ops;
static ValuePtr ones_like_ops;
static ValuePtr zeros_like_ops;
static std::shared_ptr<const opt::irpass::OptimizeIRPassLib> irpass;
static std::map<CacheKey, FuncGraphPtr> bprop_func_graph_cache;
static std::unordered_map<abstract::AbstractBasePtrList, FuncGraphPtr, abstract::AbstractBasePtrListHasher,
                          abstract::AbstractBasePtrListEqual>
  zeros_like_funcgraph_cache;
static std::unordered_map<abstract::AbstractBasePtrList, FuncGraphPtr, abstract::AbstractBasePtrListHasher,
                          abstract::AbstractBasePtrListEqual>
  ones_like_funcgraph_cache;

namespace {
FuncGraphPtr ZerosLikePrimOptPass(const pipeline::ResourcePtr &res) {
  if (irpass == nullptr) {
    irpass = std::make_shared<opt::irpass::OptimizeIRPassLib>();
  }
  opt::OptPassConfig eliminate_zeros_like_prim_pass = opt::OptPassConfig({
    irpass->zero_like_fill_zero_,
  });

  opt::OptPassGroupMap map({{"eliminate_zeros_like_prim_", eliminate_zeros_like_prim_pass}});

  auto eliminate_zeros_like_prim = opt::Optimizer::MakeOptimizer("eliminate_zeros_like_prim", res, map);
  FuncGraphPtr func_graph = res->func_graph();
  WITH(MsProfile::GetProfile()->Step("eliminate_zeros_like_prim"))[&eliminate_zeros_like_prim, &func_graph]() {
    func_graph = eliminate_zeros_like_prim->step(func_graph, true);
  };
  return func_graph;
}

FuncGraphPtr GetZerosLike(const abstract::AbstractBasePtrList &args_abs) {
  if (zeros_like_ops == nullptr) {
    zeros_like_ops = prim::GetPythonOps("zeros_like");
  }
  auto iter = zeros_like_funcgraph_cache.find(args_abs);
  if (iter != zeros_like_funcgraph_cache.end()) {
    MS_LOG(DEBUG) << "Cache hit for zeros_like: " << mindspore::ToString(args_abs);
    return BasicClone(iter->second);
  }
  if (!zeros_like_ops->isa<MetaFuncGraph>()) {
    MS_LOG(EXCEPTION) << "zeros_like is not a MetaFuncGraph";
  }
  auto zeros_like = zeros_like_ops->cast<MetaFuncGraphPtr>();
  auto zeros_like_fg = zeros_like->GenerateFuncGraph(args_abs);
  MS_EXCEPTION_IF_NULL(zeros_like_fg);
  pipeline::ResourcePtr resource = std::make_shared<pipeline::Resource>();
  auto specialized_zeros_like_fg = pipeline::Renormalize(resource, zeros_like_fg, args_abs);
  MS_EXCEPTION_IF_NULL(specialized_zeros_like_fg);
  auto opted_zeros_like_fg = ZerosLikePrimOptPass(resource);
  MS_EXCEPTION_IF_NULL(opted_zeros_like_fg);
  auto enable_grad_cache = MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_OP_GRAPH_CACHE);
  if (enable_grad_cache) {
    zeros_like_funcgraph_cache[args_abs] = BasicClone(opted_zeros_like_fg);
  }
  return opted_zeros_like_fg;
}

FuncGraphPtr GetHyperAdd(const abstract::AbstractBasePtrList &args_abs) {
  if (add_ops == nullptr) {
    add_ops = prim::GetPythonOps("hyper_add");
  }
  if (!add_ops->isa<MetaFuncGraph>()) {
    MS_LOG(EXCEPTION) << "add is not a MetaFuncGraph";
  }
  auto add = add_ops->cast<MetaFuncGraphPtr>();
  auto add_fg = add->GenerateFuncGraph(args_abs);
  MS_EXCEPTION_IF_NULL(add_fg);
  pipeline::ResourcePtr resource = std::make_shared<pipeline::Resource>();
  auto specialized_add_fg = pipeline::Renormalize(resource, add_fg, args_abs);
  MS_EXCEPTION_IF_NULL(specialized_add_fg);
  return specialized_add_fg;
}

AnfNodePtr BuildZerosLikeNode(const FuncGraphPtr &tape, const AnfNodePtr &node) {
  // Build zeros_like(node) as dout
  abstract::AbstractBasePtrList args_abs{node->abstract()->Broaden()};
  auto zeros_like_fg = GetZerosLike(args_abs);
  auto zeros_like_node = tape->NewCNode({NewValueNode(zeros_like_fg), node});
  zeros_like_node->set_abstract(zeros_like_fg->output()->abstract());
  return zeros_like_node;
}

AnfNodePtr BuildZerosLikeValue(const FuncGraphPtr &tape, const ValuePtr &out) {
  // Build zeros_like(out) as dout
  abstract::AbstractBasePtrList args_abs{out->ToAbstract()->Broaden()};
  auto zeros_like_fg = GetZerosLike(args_abs);
  auto zeros_like_value = tape->NewCNode({NewValueNode(zeros_like_fg), NewValueNode(out)});
  zeros_like_value->set_abstract(zeros_like_fg->output()->abstract());
  return zeros_like_value;
}

FuncGraphPtr GetOnesLike(const abstract::AbstractBasePtrList &args_abs) {
  if (ones_like_ops == nullptr) {
    ones_like_ops = prim::GetPythonOps("ones_like");
  }
  auto iter = ones_like_funcgraph_cache.find(args_abs);
  if (iter != ones_like_funcgraph_cache.end()) {
    MS_LOG(DEBUG) << "Cache hit for ones_like: " << mindspore::ToString(args_abs);
    return BasicClone(iter->second);
  }
  if (!ones_like_ops->isa<MetaFuncGraph>()) {
    MS_LOG(EXCEPTION) << "ones_like is not a MetaFuncGraph";
  }
  auto ones_like = ones_like_ops->cast<MetaFuncGraphPtr>();
  auto ones_like_fg = ones_like->GenerateFuncGraph(args_abs);
  MS_EXCEPTION_IF_NULL(ones_like_fg);
  pipeline::ResourcePtr resource = std::make_shared<pipeline::Resource>();
  auto specialized_ones_like_fg = pipeline::Renormalize(resource, ones_like_fg, args_abs);
  MS_EXCEPTION_IF_NULL(specialized_ones_like_fg);
  auto enable_grad_cache = MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_OP_GRAPH_CACHE);
  if (enable_grad_cache) {
    ones_like_funcgraph_cache[args_abs] = BasicClone(specialized_ones_like_fg);
  }
  return specialized_ones_like_fg;
}

bool ValueHasDynamicShape(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>()) {
    return value->cast<tensor::TensorPtr>()->base_shape_ptr() != nullptr;
  } else if (value->isa<ValueSequence>()) {
    auto value_seq = value->cast<ValueSequencePtr>();
    return std::any_of(value_seq->value().begin(), value_seq->value().end(),
                       [](const ValuePtr &elem) { return ValueHasDynamicShape(elem); });
  } else {
    return false;
  }
}

AnfNodePtr MakeDynShapeSensNode(const FuncGraphPtr &tape, const ValuePtr &sens_value) {
  MS_EXCEPTION_IF_NULL(tape);
  MS_EXCEPTION_IF_NULL(sens_value);
  if (sens_value->isa<tensor::Tensor>()) {
    auto value_node = NewValueNode(sens_value);
    auto value_node_abs = sens_value->ToAbstract()->Broaden();
    MS_LOG(DEBUG) << "Sens value abstract " << value_node_abs->ToString();
    value_node->set_abstract(value_node_abs);
    auto ones_like_value = tape->NewCNode({NewValueNode(prim::kPrimOnesLike), value_node});
    ones_like_value->set_abstract(value_node_abs);
    return ones_like_value;
  } else if (sens_value->isa<ValueTuple>()) {
    std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimMakeTuple)};
    auto value_tuple = sens_value->cast<ValueTuplePtr>();
    (void)std::transform(value_tuple->value().begin(), value_tuple->value().end(), std::back_inserter(inputs),
                         [&tape](const ValuePtr &elem) { return MakeDynShapeSensNode(tape, elem); });
    auto ones_like_value = tape->NewCNode(inputs);
    auto value_node_abs = sens_value->ToAbstract()->Broaden();
    MS_LOG(DEBUG) << "Tuple sens value abstract " << value_node_abs->ToString();
    ones_like_value->set_abstract(value_node_abs);
    return ones_like_value;
  } else if (sens_value->isa<tensor::COOTensor>()) {
    auto cootensor = sens_value->cast<tensor::COOTensorPtr>();
    return MakeDynShapeSensNode(tape, cootensor->GetValues());
  } else if (sens_value->isa<tensor::CSRTensor>()) {
    auto csrtensor = sens_value->cast<tensor::CSRTensorPtr>();
    return MakeDynShapeSensNode(tape, csrtensor->GetValues());
  } else {
    MS_LOG(EXCEPTION) << "Sens value must be a tensor or value tuple";
  }
}

AnfNodePtr BuildOnesLikeValue(const FuncGraphPtr &tape, const ValuePtr &out, const ValuePtr &sens_value) {
  // Build ones_like(out) as dout, shape is same with out.sens_value its id hold by pynative execute, which can be
  // replace forward, but out is not.
  if (ValueHasDynamicShape(out)) {
    return MakeDynShapeSensNode(tape, sens_value);
  }
  abstract::AbstractBasePtrList args_abs{out->ToAbstract()->Broaden()};
  auto ones_like_fg = GetOnesLike(args_abs);
  auto ones_like_value = tape->NewCNode({NewValueNode(ones_like_fg), NewValueNode(out)});
  ones_like_value->set_abstract(ones_like_fg->output()->abstract());
  return ones_like_value;
}

// This Faked BProp func_graph should not be present in the final top bprop func_graph.
FuncGraphPtr BuildFakeBProp(const PrimitivePtr &prim, size_t inputs_num) {
  auto func_graph = std::make_shared<FuncGraph>();
  std::vector<AnfNodePtr> outputs;
  outputs.push_back(NewValueNode(prim::kPrimMakeTuple));

  auto fake_bprop = std::make_shared<Primitive>("fake_bprop");
  (void)fake_bprop->AddAttr("info", MakeValue("Primitive " + prim->name() + "'s bprop not defined."));

  for (size_t i = 0; i < inputs_num; ++i) {
    // Mock params for inputs
    auto param = func_graph->add_parameter();
    MS_EXCEPTION_IF_NULL(param);
    // Mock derivatives for each inputs
    outputs.push_back(func_graph->NewCNode({NewValueNode(fake_bprop), param}));
  }
  // mock params for out and dout
  (void)func_graph->add_parameter();
  (void)func_graph->add_parameter();
  func_graph->set_output(func_graph->NewCNode(outputs));
  return func_graph;
}
}  // namespace

class PynativeAdjoint {
 public:
  enum class FuncGraphType { kForwardPropagate, kBackwardPropagate };
  PynativeAdjoint(const FuncGraphPtr &tape, const ValuePtrList &op_args, const ValuePtr &out, const FuncGraphPtr &fg,
                  FuncGraphType fg_type = FuncGraphType::kBackwardPropagate)
      : tape_(tape), op_args_(op_args), out_(out), fg_(fg), fg_type_(fg_type) {}

  ~PynativeAdjoint() = default;
  AnfNodePtrList &users() { return users_; }
  const ValuePtrList &op_args() const { return op_args_; }
  const ValuePtr &out() const { return out_; }
  const FuncGraphPtr &fg() const { return fg_; }
  const FuncGraphType &fg_type() const { return fg_type_; }
  AnfNodePtr RealDout() {
    if (dout_ != nullptr) {
      return dout_;
    }
    return BuildZerosLikeValue(tape_, out_);
  }

  void AccumulateDout(const AnfNodePtr &dout_factor) {
    if (dout_factor->abstract() == nullptr) {
      MS_LOG(EXCEPTION) << "Abstract of dout_factor should not be null: " << dout_factor->ToString();
    }
    if (dout_ != nullptr) {
      MS_LOG(DEBUG) << "Update dout " << dout_->ToString() << " with dout_factor " << dout_factor->ToString();
      auto arg = out_->ToAbstract()->Broaden();
      abstract::AbstractBasePtrList args_abs{arg, arg};
      auto add_fg = GetHyperAdd(args_abs);
      MS_EXCEPTION_IF_NULL(add_fg);
      dout_ = tape_->NewCNode({NewValueNode(add_fg), dout_, dout_factor});
      dout_->set_abstract(add_fg->output()->abstract());
      MS_LOG(DEBUG) << "New dout_ " << dout_->DebugString();
      return;
    }
    dout_ = dout_factor;
  }

  AnfNodePtr k_node() const { return k_node_; }
  void set_k_node(const AnfNodePtr &k_node) { k_node_ = k_node; }

  AnfNodePtr dout() const { return dout_; }

 private:
  const FuncGraphPtr tape_;
  AnfNodePtr dout_{nullptr};
  // Used by whose
  AnfNodePtrList users_;
  // cache these arguments from ad caller.
  const ValuePtrList op_args_;
  // For CNode , it's output of cnode. For Parameter or ValueNode, it's its value.
  const ValuePtr out_;
  // fg_ is a bprop_fg generated from Primitive.
  // or a fprop_fg passed from caller.
  // FuncGraph to tape_;
  const FuncGraphPtr fg_;
  const FuncGraphType fg_type_;
  // k mapped cnode for primal CNode; primal CNode is owned by primal funcgraph, this is owned by tape_;
  AnfNodePtr k_node_;
};
using PynativeAdjointPtr = std::shared_ptr<PynativeAdjoint>;

class KPynativeCellImpl : public KPynativeCell {
 public:
  KPynativeCellImpl(const AnfNodePtrList &cell_inputs, const std::vector<ValuePtr> &input_param_values)
      : tape_(std::make_shared<FuncGraph>()), cell_inputs_(cell_inputs) {
    tape_->debug_info()->set_name("grad_top");
    for (size_t i = 0; i < cell_inputs.size(); ++i) {
      TraceGuard trace_guard(std::make_shared<TraceCopy>(cell_inputs[i]->debug_info()));
      (void)tape_->add_parameter();
      // Build adjoint for every input parameter
      auto input_adjoint =
        std::make_shared<PynativeAdjoint>(tape_, ValuePtrList{}, input_param_values[i], FuncGraphPtr(nullptr));
      (void)anfnode_to_adjoin_.insert(std::make_pair(cell_inputs[i], input_adjoint));
    }
  }
  ~KPynativeCellImpl() override = default;
  bool KPynativeOp(const CNodePtr &cnode, const ValuePtrList &op_args, const ValuePtr &out);
  bool KPynativeWithBProp(const CNodePtr &cnode, const ValuePtrList &op_args, const ValuePtr &out,
                          const FuncGraphPtr &bprop_fg);
  bool KPynativeWithFProp(const CNodePtr &cnode, const ValuePtrList &op_args, const ValuePtr &out,
                          const FuncGraphPtr &fprop_fg) override;
  void UpdateOutputNodeOfTopCell(const AnfNodePtr &output_node, const ValuePtr &sens_out) override;
  // Build a back propagate funcgraph, each cnode in primal funcgraph is replaced by value node or formal cnode, so it
  // can be grad again.
  FuncGraphPtr Finish(const AnfNodePtrList &weights, const std::vector<size_t> &grad_position,
                      const GradAttr &grad_attr, bool build_formal_param);

 private:
  bool need_propagate_stop_gradient_{false};
  // Last cnode of this Cell, may be a primitive op or cell with user defined bprop.
  AnfNodePtr last_node_{nullptr};
  ValuePtr oneslike_sens_value_{nullptr};
  FuncGraphPtr tape_;
  AnfNodePtrList cell_inputs_;
  // These weights need to calculate gradient.
  mindspore::HashSet<AnfNodePtr> need_grad_weights_;
  OrderedMap<AnfNodePtr, PynativeAdjointPtr> anfnode_to_adjoin_;

  // For CNode like TupleGetItem, ListGetItem, MakeTuple, MakeList, it's bypassed by caller so
  // no KPynativeOp is called for these CNode. Here we forge Adjoint for these CNode.
  PynativeAdjointPtr ForgeCNodeAdjoint(const CNodePtr &cnode);
  PynativeAdjointPtr ForgeGetItemAdjoint(const CNodePtr &cnode);
  PynativeAdjointPtr ForgeMakeSequenceAdjoint(const CNodePtr &cnode);
  bool BuildAdjoint(const CNodePtr &cnode, const ValuePtrList &op_args, const ValuePtr &out, const FuncGraphPtr &fg,
                    const PynativeAdjoint::FuncGraphType fg_type = PynativeAdjoint::FuncGraphType::kBackwardPropagate);
  void BuildAdjointForInput(const CNodePtr &cnode, const ValuePtrList &op_args);
  bool IsCNodeNeedGrad(const AnfNodePtr &node_ptr) const;
  std::vector<bool> GetNeedGradFlags(const CNodePtr &cnode);
  void PropagateStopGradient();
  bool AllReferencesStopped(const CNodePtr &curr_cnode);
  OrderedMap<AnfNodePtr, PynativeAdjointPtr>::reverse_iterator GetLastNodeReverseIter();
  // Back propagate for all node;
  // if by_value is true, in bprop_app cnode, every input is value node;
  // if by_value is false, in bprop_app cnode, input is the k mapped node, so it can be grad again.
  bool BackPropagate(bool by_value);
  bool BackPropagateOneCNodeWithBPropFuncGraph(const CNodePtr &cnode, const PynativeAdjointPtr &adjoint,
                                               const FuncGraphPtr &bprop_fg, bool by_value);
  bool BackPropagateOneCNodeWithFPropFuncGraph(const CNodePtr &cnode, const PynativeAdjointPtr &adjoint,
                                               const FuncGraphPtr &fprop_fg, bool by_value);
  bool BackPropagate(const CNodePtr &cnode_primal, const CNodePtr &bprop_app);
  AnfNodePtr BuildKNodeForCNodeInput(const PynativeAdjointPtr &cnode_adjoint, const AnfNodePtr &input_node,
                                     size_t input_index);
  const AnfNodePtrList BuildKNodeListFromPrimalCNode(const CNodePtr &cnode, const PynativeAdjointPtr &adjoint);
  FuncGraphPtr BuildBPropCutFuncGraph(const PrimitivePtr &prim, const CNodePtr &cnode) const;
  // Back propagate for MakeList or MakeTuple is generated from MetaFuncGraph.
  FuncGraphPtr BuildMakeSequenceBprop(const PrimitivePtr &prim, const CNodePtr &cnode) const;
  // Replace input or weights parameter from primal funcgraph to parameters of tape_;
  void ReplacePrimalParameter(const AnfNodePtrList &weights, bool has_sens_arg);
  // Set sens and weights parameter nodes by user input info
  void SetSensAndWeights(const AnfNodePtrList &weights, bool has_sens_arg);
  // Set return node according to grad flag
  void SetOutput(const AnfNodePtrList &weights, const std::vector<size_t> &grad_position, const GradAttr &grad_attr);
  AnfNodePtr GetGradNodeByIndex(const AnfNodePtrList &node_list, size_t index) const;
  AnfNodePtr GetInputGrad(bool grad_all_inputs, bool get_by_position, const std::vector<size_t> &grad_position) const;
  AnfNodePtr GetWeightGrad(bool grad_weights, bool weight_param_is_tuple, const AnfNodePtrList &weights) const;

  // for higher order gradient;
  // Build k mapped node owned by tape_ for each cnode in primal funcgraph, so these node can be
  // used in tape_ to keep tracking the cnode dependency.
  bool BuildKNode();
  CNodePtr GetBPropFromFProp(const FuncGraphPtr &fprop_fg, const AnfNodePtrList &args);
};
using KPynativeCellImplPtr = std::shared_ptr<KPynativeCellImpl>;

KPynativeCellPtr GradPynativeCellBegin(const AnfNodePtrList &cell_inputs,
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
  return std::make_shared<KPynativeCellImpl>(cell_inputs, input_param_values);
}

FuncGraphPtr GradPynativeCellEnd(const KPynativeCellPtr &k_cell, const AnfNodePtrList &weights,
                                 const std::vector<size_t> &grad_position, const GradAttr &grad_attr,
                                 bool build_formal_param) {
  auto k_cell_impl = std::dynamic_pointer_cast<KPynativeCellImpl>(k_cell);
  return k_cell_impl->Finish(weights, grad_position, grad_attr, build_formal_param);
}

FuncGraphPtr KPynativeCellImpl::Finish(const AnfNodePtrList &weights, const std::vector<size_t> &grad_position,
                                       const GradAttr &grad_attr, bool build_formal_param) {
  // propagate stop_gradient flag to cnode before back propagate;
  PropagateStopGradient();
  // Set sens node and weights node
  SetSensAndWeights(weights, grad_attr.has_sens);
  // Build forward CNode;
  if (build_formal_param) {
    (void)BuildKNode();
  }
  // BackPropagate sensitivity, except when the last node is a valuenode which may be obtained by constant folding;
  if (!last_node_->isa<ValueNode>()) {
    (void)BackPropagate(!build_formal_param);
  }
  // Return the gradient;
  if (grad_attr.get_by_position && grad_position.empty()) {
    MS_LOG(EXCEPTION) << "grad_position should not be empty when grad by position!";
  }
  SetOutput(weights, grad_position, grad_attr);
  // Replace Parameter of primal funcgraph  with parameter of tape_;
  ReplacePrimalParameter(weights, grad_attr.has_sens);
#ifdef ENABLE_DUMP_IR
  auto save_graphs_flg = MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs_flg) {
    DumpIR("before_final_opt.ir", tape_);
  }
#endif
  return tape_;
}

bool GradPynativeOp(const KPynativeCellPtr &k_cell, const CNodePtr &cnode, const ValuePtrList &op_args,
                    const ValuePtr &out) {
  auto k_cell_impl = std::dynamic_pointer_cast<KPynativeCellImpl>(k_cell);
  return k_cell_impl->KPynativeOp(cnode, op_args, out);
}

bool KPynativeCellImpl::KPynativeOp(const CNodePtr &cnode, const ValuePtrList &op_args, const ValuePtr &out) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto prim = GetCNodePrimitive(cnode);
  if (prim == nullptr) {
    MS_LOG(EXCEPTION) << "Should be primitive, but: " << cnode->DebugString();
  }
  if (IsPrimitiveEquals(prim, prim::kPrimStopGradient) || IsPrimitiveEquals(prim, prim::kPrimUpdateState)) {
    need_propagate_stop_gradient_ = true;
  }

  FuncGraphPtr bprop_fg = nullptr;
  if (IsPrimitiveEquals(prim, prim::kPrimHookBackward) || IsPrimitiveEquals(prim, prim::kPrimCellBackwardHook)) {
    bprop_fg = BuildBPropCutFuncGraph(prim, cnode);
  } else if (IsPrimitiveEquals(prim, prim::kPrimMakeTuple) || IsPrimitiveEquals(prim, prim::kPrimMakeList)) {
    bprop_fg = BuildMakeSequenceBprop(prim, cnode);
  } else {
    bprop_fg = g_k_prims_pynative.GetPossibleBprop(prim);
    if (bprop_fg == nullptr) {
      MS_LOG(DEBUG) << "Cannot find defined bprop for cnode prim: " << cnode->DebugString();
      bprop_fg = BuildFakeBProp(prim, cnode->size() - 1);
    }
  }
  MS_EXCEPTION_IF_NULL(bprop_fg);
  (void)BuildAdjoint(cnode, op_args, out, bprop_fg);
  return true;
}

bool GradPynativeWithBProp(const KPynativeCellPtr &k_cell, const CNodePtr &cnode, const ValuePtrList &op_args,
                           const ValuePtr &out, const FuncGraphPtr &bprop_fg) {
  auto k_cell_impl = std::dynamic_pointer_cast<KPynativeCellImpl>(k_cell);
  return k_cell_impl->KPynativeWithBProp(cnode, op_args, out, bprop_fg);
}

bool KPynativeCellImpl::KPynativeWithBProp(const CNodePtr &cnode, const ValuePtrList &op_args, const ValuePtr &out,
                                           const FuncGraphPtr &bprop_fg) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto primal_fg = GetCNodeFuncGraph(cnode);
  if (primal_fg == nullptr) {
    MS_LOG(EXCEPTION) << "Should be func graph, but: " << cnode->DebugString();
  }
  MS_EXCEPTION_IF_NULL(bprop_fg);
  (void)BuildAdjoint(cnode, op_args, out, bprop_fg);

  return true;
}

bool KPynativeCellImpl::KPynativeWithFProp(const CNodePtr &cnode, const ValuePtrList &op_args, const ValuePtr &out,
                                           const FuncGraphPtr &fprop_fg) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(fprop_fg);

  (void)BuildAdjoint(cnode, op_args, out, fprop_fg, PynativeAdjoint::FuncGraphType::kForwardPropagate);

  return true;
}

void KPynativeCellImpl::UpdateOutputNodeOfTopCell(const AnfNodePtr &output_node, const ValuePtr &sens_out) {
  MS_EXCEPTION_IF_NULL(output_node);
  MS_EXCEPTION_IF_NULL(sens_out);
  MS_LOG(DEBUG) << "Real output node of top cell is " << output_node->DebugString();
  last_node_ = output_node;
  oneslike_sens_value_ = sens_out;

  const auto &last_node_adjoint_iter = anfnode_to_adjoin_.find(last_node_);
  if (last_node_adjoint_iter == anfnode_to_adjoin_.end()) {
    if (IsPrimitiveCNode(output_node, prim::kPrimTupleGetItem) ||
        IsPrimitiveCNode(output_node, prim::kPrimListGetItem)) {
      MS_LOG(DEBUG) << "Build cnode adjoint for anfnode: " << output_node->DebugString();
      auto cnode = output_node->cast<CNodePtr>();
      (void)ForgeGetItemAdjoint(cnode);
      return;
    } else if (output_node->isa<ValueNode>()) {
      auto v_node = output_node->cast<ValueNodePtr>();
      MS_LOG(DEBUG) << "Build adjoint for valuenode: " << v_node->ToString();
      auto v_node_pynative_adjoint =
        std::make_shared<PynativeAdjoint>(tape_, ValuePtrList{}, v_node->value(), FuncGraphPtr(nullptr));
      (void)anfnode_to_adjoin_.insert(std::make_pair(output_node, v_node_pynative_adjoint));
      return;
    }
    MS_LOG(EXCEPTION) << "BackPropagate adjoint does not exist for input: " << last_node_->DebugString();
  }
}

PynativeAdjointPtr KPynativeCellImpl::ForgeGetItemAdjoint(const CNodePtr &cnode) {
  if (cnode->size() != 3) {
    MS_LOG(EXCEPTION) << "TupleGetItem/ListGetItem CNode should have 3 inputs, but CNode: " << cnode->DebugString();
  }
  // Input 1 of CNode;
  PynativeAdjointPtr input_1_adjoint = nullptr;
  auto input_1 = cnode->input(1);
  auto input_1_adjoint_iter = anfnode_to_adjoin_.find(input_1);
  if (input_1_adjoint_iter == anfnode_to_adjoin_.end()) {
    if (!input_1->isa<CNode>()) {
      MS_LOG(EXCEPTION) << "Input 1 of CNode should be a CNode, CNode: " << cnode->DebugString();
    }
    input_1_adjoint = ForgeCNodeAdjoint(input_1->cast<CNodePtr>());
    if (input_1_adjoint == nullptr) {
      MS_LOG(EXCEPTION) << "Build adjoint for input 1 of CNode failed, CNode: " << cnode->DebugString();
    }
    input_1_adjoint->users().push_back(cnode);
  } else {
    input_1_adjoint = input_1_adjoint_iter->second;
  }
  if (!input_1_adjoint->out()->isa<ValueSequence>()) {
    MS_LOG(EXCEPTION) << "Input of CNode should be evaluated to a ValueSequence. CNode: " << cnode->DebugString()
                      << ", out of input1: " << input_1_adjoint->out()->ToString();
  }
  auto input_1_out = input_1_adjoint->out()->cast<ValueSequencePtr>();

  // Input 2 of CNode;
  auto index_value = GetValueNode<Int64ImmPtr>(cnode->input(2));
  if (index_value == nullptr) {
    MS_LOG(EXCEPTION) << "CNode input 2 should be a Int64Imm, CNode: " << cnode->DebugString();
  }
  if (index_value->value() < 0) {
    MS_LOG(EXCEPTION) << "CNode input 2 should not be less than 0, CNode: " << cnode->DebugString();
  }
  size_t index_value_imm = LongToSize(index_value->value());
  if (index_value_imm >= input_1_out->size()) {
    MS_LOG(EXCEPTION) << "CNode input 2 should be index between [0, " << input_1_out->size()
                      << ", but: " << index_value->ToString();
  }
  auto cnode_out = (*input_1_out)[index_value_imm];
  ValuePtrList op_args{input_1_out, index_value};
  auto built = KPynativeOp(cnode, op_args, cnode_out);
  if (!built) {
    MS_LOG(EXCEPTION) << "Build Adjoint for GetItem node failed, CNode: " << cnode->DebugString();
  }
  const auto &cnode_adjoint_iter = anfnode_to_adjoin_.find(cnode);
  if (cnode_adjoint_iter == anfnode_to_adjoin_.end()) {
    MS_LOG(EXCEPTION) << "Build Adjoint for GetItem node failed, CNode: " << cnode->DebugString();
  }
  return cnode_adjoint_iter->second;
}

PynativeAdjointPtr KPynativeCellImpl::ForgeMakeSequenceAdjoint(const CNodePtr &cnode) {
  // () or [] is not supported yet.
  if (cnode->size() <= 1) {
    MS_LOG(DEBUG) << "MakeTuple/MakeList CNode is empty Tuple/List, CNode: " << cnode->DebugString();
    auto empty_tuple = MakeValue(std::vector<ValuePtr>{});
    auto dummy_adjoint =
      std::make_shared<PynativeAdjoint>(FuncGraphPtr(nullptr), ValuePtrList{}, empty_tuple, FuncGraphPtr(nullptr));
    anfnode_to_adjoin_[cnode] = dummy_adjoint;
    cnode->set_stop_gradient(true);
    return dummy_adjoint;
  }
  ValuePtrList op_args;
  for (size_t i = 1; i < cnode->size(); ++i) {
    const auto &input = cnode->input(i);
    const auto &input_adjoint_iter = anfnode_to_adjoin_.find(input);
    if (input_adjoint_iter == anfnode_to_adjoin_.end()) {
      MS_LOG(DEBUG) << "Item in CNode cannot found in cache. Input is: " << input->DebugString();
      if (input->isa<CNode>()) {
        const auto input_cnode = input->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(input_cnode);
        auto forged_input_adjoint = ForgeCNodeAdjoint(input->cast<CNodePtr>());
        op_args.push_back(forged_input_adjoint->out());
      } else if (input->isa<ValueNode>()) {
        const auto &input_value = GetValueNode(input);
        op_args.push_back(input_value);
      } else if (input->isa<Parameter>()) {
        const auto input_parameter = dyn_cast<Parameter>(input);
        MS_EXCEPTION_IF_NULL(input_parameter);
        const auto &input_value = input_parameter->default_param();
        op_args.push_back(input_value);
      } else {
        MS_LOG(EXCEPTION) << "The input of MakeTuple/MakeList is not a CNode, ValueNode or Parameter, but "
                          << input->DebugString();
      }
    } else {
      op_args.push_back(input_adjoint_iter->second->out());
    }
  }
  ValuePtr cnode_out = nullptr;
  if (IsPrimitiveCNode(cnode, prim::kPrimMakeTuple)) {
    cnode_out = MakeValue(op_args);
  }
  if (IsPrimitiveCNode(cnode, prim::kPrimMakeList)) {
    cnode_out = std::make_shared<ValueList>(op_args);
  }
  // op_args is real inputs find by prev cnode outputs
  auto built = KPynativeOp(cnode, op_args, cnode_out);
  if (!built) {
    MS_LOG(EXCEPTION) << "Build Adjoint for MakeTuple/MakeList node failed, CNode: " << cnode->DebugString();
  }
  const auto &cnode_adjoint_iter = anfnode_to_adjoin_.find(cnode);
  if (cnode_adjoint_iter == anfnode_to_adjoin_.end()) {
    MS_LOG(EXCEPTION) << "Build Adjoint for MakeTuple/MakeList node failed, CNode: " << cnode->DebugString();
  }
  return cnode_adjoint_iter->second;
}

PynativeAdjointPtr KPynativeCellImpl::ForgeCNodeAdjoint(const CNodePtr &cnode) {
  if (IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem) || IsPrimitiveCNode(cnode, prim::kPrimListGetItem)) {
    MS_LOG(DEBUG) << "Build cnode adjoint for anfnode: " << cnode->DebugString();
    return ForgeGetItemAdjoint(cnode);
  }

  if (IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) || IsPrimitiveCNode(cnode, prim::kPrimMakeList)) {
    MS_LOG(DEBUG) << "Build cnode adjoint for anfnode: " << cnode->DebugString();
    return ForgeMakeSequenceAdjoint(cnode);
  }
  MS_LOG(EXCEPTION) << "Unknown cnode: " << cnode->DebugString();
}

void KPynativeCellImpl::BuildAdjointForInput(const CNodePtr &cnode, const ValuePtrList &op_args) {
  const auto &anfnode_adjoint_iter = anfnode_to_adjoin_.find(cnode);
  if (anfnode_adjoint_iter != anfnode_to_adjoin_.end()) {
    MS_LOG(EXCEPTION) << "CNode should be unique, but: " << cnode->DebugString();
  }
  // Book-keeping last cnode, as dout of this node will be given from outside;
  last_node_ = cnode;

  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    auto input = cnode->input(i);
    const auto &input_adjoint_iter = anfnode_to_adjoin_.find(input);
    if (input_adjoint_iter == anfnode_to_adjoin_.end()) {
      if (input->isa<CNode>()) {
        auto cnode_input = input->cast<CNodePtr>();
        auto forged_adjoint = ForgeCNodeAdjoint(cnode_input);
        if (forged_adjoint == nullptr) {
          MS_LOG(EXCEPTION) << "Cannot forge adjoint for anfnode: " << input->DebugString();
        }
        forged_adjoint->users().push_back(cnode);
      } else {
        MS_EXCEPTION_IF_NULL(op_args[i - 1]);
        auto input_adjoint =
          std::make_shared<PynativeAdjoint>(tape_, ValuePtrList{}, op_args[i - 1], FuncGraphPtr(nullptr));
        (void)anfnode_to_adjoin_.insert(std::make_pair(input, input_adjoint));
        input_adjoint->users().push_back(cnode);
      }
    } else {
      input_adjoint_iter->second->users().push_back(cnode);
    }
  }
}

bool KPynativeCellImpl::IsCNodeNeedGrad(const AnfNodePtr &node_ptr) const {
  MS_EXCEPTION_IF_NULL(node_ptr);
  if (node_ptr->isa<CNode>()) {
    const auto &cnode = node_ptr->cast<CNodePtr>();
    if (cnode == nullptr || !cnode->HasAttr(kAttrIsCNodeNeedGrad)) {
      return true;
    }

    return GetValue<bool>(cnode->GetAttr(kAttrIsCNodeNeedGrad));
  }

  auto param_ptr = node_ptr->cast<ParameterPtr>();
  if (param_ptr == nullptr) {
    // Value node will return here.
    return false;
  }
  auto param_value = param_ptr->param_info();
  if (param_value == nullptr) {
    // If node is a parameter, but param_info is null, node need to grad.
    return true;
  }
  return param_value->requires_grad();
}

std::vector<bool> KPynativeCellImpl::GetNeedGradFlags(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<bool> need_grad_flag_of_inputs;
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    (void)need_grad_flag_of_inputs.emplace_back(IsCNodeNeedGrad(cnode->input(i)));
  }
  return need_grad_flag_of_inputs;
}

bool KPynativeCellImpl::BuildAdjoint(const CNodePtr &cnode, const ValuePtrList &op_args, const ValuePtr &out,
                                     const FuncGraphPtr &fg, const PynativeAdjoint::FuncGraphType fg_type) {
  auto need_grad_flag_of_inputs = GetNeedGradFlags(cnode);
  size_t need_grad_input_num = std::count(need_grad_flag_of_inputs.begin(), need_grad_flag_of_inputs.end(), true);
  cnode->AddAttr(kAttrIsCNodeNeedGrad, MakeValue(need_grad_input_num != 0));
  if (need_grad_input_num != need_grad_flag_of_inputs.size()) {
    cnode->AddAttr(kAttrNeedGradFlagOfInputs, MakeValue(need_grad_flag_of_inputs));
  } else if (cnode->HasAttr(kAttrNeedGradFlagOfInputs)) {
    cnode->EraseAttr(kAttrNeedGradFlagOfInputs);
  }

  // Optimize the bprop_fg based on value.
  // Clone op_args and out, so the address of tensor data can be reset to nullptr if the value of tensor
  // is not used in bprop_fg;
  ValuePtrList cloned_op_args;
  (void)std::transform(op_args.begin(), op_args.end(), std::back_inserter(cloned_op_args),
                       [](const ValuePtr &value) { return ShallowCopyTensorValue(value); });
  ValuePtr cloned_out = ShallowCopyTensorValue(out);
  PynativeAdjointPtr cnode_adjoint;
  if (fg_type == PynativeAdjoint::FuncGraphType::kBackwardPropagate) {
    auto optimized_bprop_fg = OptimizeBPropFuncGraph(fg, cnode, cloned_op_args, cloned_out);
    cnode_adjoint = std::make_shared<PynativeAdjoint>(tape_, cloned_op_args, cloned_out, optimized_bprop_fg);
  } else {
    cnode_adjoint = std::make_shared<PynativeAdjoint>(tape_, cloned_op_args, cloned_out, fg, fg_type);
  }

  BuildAdjointForInput(cnode, op_args);
  MS_LOG(DEBUG) << "Build Adjoint for CNode: " << cnode->DebugString();
  (void)anfnode_to_adjoin_.insert(std::make_pair(cnode, cnode_adjoint));
  return true;
}

FuncGraphPtr OptimizeBPropFuncGraph(const FuncGraphPtr &bprop_fg, const CNodePtr &cnode, const ValuePtrList &op_args,
                                    const ValuePtr &out) {
  auto optimized_bprop_fg =
    PrimBpropOptimizer::GetPrimBpropOptimizerInst().OptimizeBPropFuncGraph(bprop_fg, cnode, op_args, out);
  return optimized_bprop_fg;
}

bool KPynativeCellImpl::BackPropagate(const CNodePtr &cnode_primal, const CNodePtr &bprop_app) {
  abstract::AbstractTuplePtr abstract_tuple = nullptr;
  auto bprop_app_abstract = bprop_app->abstract();
  // if input 0 of bprop_app is a CNode other than FuncGraph ValueNode, bprop_app_abstract is nullptr;
  // After tape_ returned, caller should renormalize tape_ to set abstract of each AnfNode.
  if (bprop_app_abstract != nullptr) {
    abstract_tuple = bprop_app_abstract->cast<abstract::AbstractTuplePtr>();
    if (abstract_tuple->size() != (cnode_primal->size() - 1)) {
      MS_LOG(EXCEPTION) << "AbstractTuple size: " << abstract_tuple->ToString()
                        << " not match primal cnode input size: " << cnode_primal->DebugString();
    }
  }
  for (size_t i = 1; i < cnode_primal->size(); i++) {
    auto input = cnode_primal->input(i);
    // Useless to accumulate sens for ValueNode, the sens for ValueNode should be zeros_like;
    if (input->isa<ValueNode>()) {
      continue;
    }
    auto cnode_input = input->cast<CNodePtr>();
    if (cnode_input != nullptr && cnode_input->stop_gradient()) {
      MS_LOG(DEBUG) << "Bypass accumulate dout to cnode with stop_gradient flag, cnode: " << input->DebugString();
      continue;
    }
    // Backprop sens wrt inputs.
    const auto &input_adjoint_iter = anfnode_to_adjoin_.find(input);
    if (input_adjoint_iter == anfnode_to_adjoin_.end()) {
      MS_LOG(EXCEPTION) << "BackPropagate adjoint does not exist input[" << i << "] " << input->DebugString();
    }
    AnfNodePtr din;
    if (abstract_tuple != nullptr) {
      din = tape_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), bprop_app, NewValueNode(SizeToLong(i - 1))});
      din->set_abstract((*abstract_tuple)[i - 1]);
    } else {
      // bprop_app[0] is env;
      din = tape_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), bprop_app, NewValueNode(SizeToLong(i))});
      din->set_abstract(input_adjoint_iter->second->out()->ToAbstract()->Broaden());
    }
    din = HandleRealToComplex(input, din->cast<CNodePtr>(), tape_);
    input_adjoint_iter->second->AccumulateDout(din);
  }
  return true;
}

AnfNodePtr KPynativeCellImpl::BuildKNodeForCNodeInput(const PynativeAdjointPtr &cnode_adjoint,
                                                      const AnfNodePtr &input_node, size_t input_index) {
  MS_EXCEPTION_IF_NULL(cnode_adjoint);
  MS_EXCEPTION_IF_NULL(input_node);
  if (input_node->isa<CNode>()) {
    const auto &input_adjoint_iter = anfnode_to_adjoin_.find(input_node);
    if (input_adjoint_iter == anfnode_to_adjoin_.end()) {
      MS_LOG(EXCEPTION) << "cannot find input in adjoint map, inp: " << input_node->DebugString();
    }
    return input_adjoint_iter->second->k_node();
  } else {
    if (input_node->isa<Parameter>()) {
      bool is_weight = input_node->cast<ParameterPtr>()->has_default();
      // If weight does not need to calculate gradient, it will be converted to value node.
      if (is_weight && need_grad_weights_.find(input_node) == need_grad_weights_.end()) {
        if (input_index < 1) {
          MS_EXCEPTION(ValueError) << "The input_index is smaller than 1.";
        }
        if (input_index > cnode_adjoint->op_args().size()) {
          MS_EXCEPTION(ValueError) << "The input_index: " << input_index
                                   << " out of range:" << cnode_adjoint->op_args().size();
        }
        return NewValueNode(cnode_adjoint->op_args()[input_index - 1]);
      }
    }
    return input_node;
  }
}

const AnfNodePtrList KPynativeCellImpl::BuildKNodeListFromPrimalCNode(const CNodePtr &cnode,
                                                                      const PynativeAdjointPtr &adjoint) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(adjoint);
  AnfNodePtrList node_list;
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    (void)node_list.emplace_back(BuildKNodeForCNodeInput(adjoint, cnode->input(i), i));
  }
  return node_list;
}

bool KPynativeCellImpl::BackPropagateOneCNodeWithBPropFuncGraph(const CNodePtr &cnode,
                                                                const PynativeAdjointPtr &adjoint,
                                                                const FuncGraphPtr &bprop_fg, bool by_value) {
  if (adjoint->dout() == nullptr) {
    // If dout is null, the node does not need to grad.
    MS_LOG(DEBUG) << "node dout is null, node:" << cnode->DebugString();
    return true;
  }

  AnfNodePtrList node_list;
  abstract::AbstractBasePtr bprop_output_abs;

  bprop_output_abs = bprop_fg->output()->abstract();
  if (bprop_output_abs == nullptr) {
    MS_LOG(EXCEPTION) << "Abstract of bprop_output_abs is not AbstractTuple, but nullptr";
  }
  if (!bprop_output_abs->isa<abstract::AbstractTuple>()) {
    MS_LOG(EXCEPTION) << "Abstract of bprop_output_abs is not AbstractTuple, but: " << bprop_output_abs->ToString();
  }
  node_list.push_back(NewValueNode(bprop_fg));

  if (by_value) {
    for (size_t i = 0; i < adjoint->op_args().size(); ++i) {
      auto input_node = cnode->input(i + 1);
      if (input_node->isa<Parameter>()) {
        bool is_weight = input_node->cast<ParameterPtr>()->has_default();
        if (!is_weight || need_grad_weights_.find(input_node) != need_grad_weights_.end()) {
          node_list.push_back(input_node);
          continue;
        }
      }
      auto v_node = NewValueNode(adjoint->op_args()[i]);
      v_node->set_abstract(adjoint->op_args()[i]->ToAbstract()->Broaden());
      node_list.push_back(v_node);
    }
    auto out_node = NewValueNode(adjoint->out());
    out_node->set_abstract(adjoint->out()->ToAbstract()->Broaden());
    node_list.push_back(out_node);
    node_list.push_back(adjoint->RealDout());
  } else {
    const auto &k_node_list = BuildKNodeListFromPrimalCNode(cnode, adjoint);
    (void)node_list.insert(node_list.cend(), k_node_list.cbegin(), k_node_list.cend());
    // out;
    node_list.push_back(adjoint->k_node());
    // dout
    node_list.push_back(adjoint->RealDout());
  }
  // Back propagate process
  auto bprop_app = tape_->NewCNode(node_list);
  bprop_app->set_abstract(bprop_output_abs);
  (void)BackPropagate(cnode, bprop_app);
  return true;
}

bool KPynativeCellImpl::BackPropagateOneCNodeWithFPropFuncGraph(const CNodePtr &cnode,
                                                                const PynativeAdjointPtr &adjoint,
                                                                const FuncGraphPtr &fprop_fg, bool by_value) {
  MS_LOG(DEBUG) << "BackPropagate for CNode: " << cnode->DebugString();

  if (adjoint->dout() == nullptr) {
    // If dout is null, the node does not need to grad.
    MS_LOG(DEBUG) << "node dout is null, node:" << cnode->DebugString();
    return true;
  }

  AnfNodePtrList node_list;
  CNodePtr bprop_cnode;
  if (by_value) {
    AnfNodePtrList args_node_list;
    for (size_t i = 0; i < adjoint->op_args().size(); ++i) {
      auto input_node = cnode->input(i + 1);
      if (input_node->isa<Parameter>()) {
        bool is_weight = input_node->cast<ParameterPtr>()->has_default();
        if (!is_weight || need_grad_weights_.find(input_node) != need_grad_weights_.end()) {
          args_node_list.push_back(input_node);
          continue;
        }
      }
      auto v_node = NewValueNode(adjoint->op_args()[i]);
      v_node->set_abstract(adjoint->op_args()[i]->ToAbstract()->Broaden());
      args_node_list.push_back(v_node);
    }
    bprop_cnode = GetBPropFromFProp(fprop_fg, args_node_list);
  } else {
    const auto &k_node_list = BuildKNodeListFromPrimalCNode(cnode, adjoint);
    bprop_cnode = GetBPropFromFProp(fprop_fg, k_node_list);
  }
  node_list.push_back(bprop_cnode);
  // dout;
  node_list.push_back(adjoint->RealDout());
  // Back propagate process
  auto bprop_app = tape_->NewCNode(node_list);
  (void)BackPropagate(cnode, bprop_app);
  return true;
}

OrderedMap<AnfNodePtr, PynativeAdjointPtr>::reverse_iterator KPynativeCellImpl::GetLastNodeReverseIter() {
  for (auto iter = anfnode_to_adjoin_.rbegin(); iter != anfnode_to_adjoin_.rend(); ++iter) {
    if (!iter->first->isa<CNode>()) {
      continue;
    }
    if (iter->first->cast<CNodePtr>() == last_node_) {
      return iter;
    }
  }
  return anfnode_to_adjoin_.rend();
}

bool KPynativeCellImpl::BackPropagate(bool by_value) {
  const auto &last_node_reverse_iter = GetLastNodeReverseIter();
  for (auto iter = last_node_reverse_iter; iter != anfnode_to_adjoin_.rend(); ++iter) {
    if (!iter->first->isa<CNode>()) {
      continue;
    }
    auto cnode = iter->first->cast<CNodePtr>();
    if (cnode->stop_gradient()) {
      MS_LOG(DEBUG) << "Bypass backpropagate for cnode with stop_gradient flag: " << cnode->DebugString();
      continue;
    }
    MS_LOG(DEBUG) << "BackPropagate for CNode: " << cnode->DebugString();
    auto fg = iter->second->fg();
    auto fg_type = iter->second->fg_type();
    if (fg_type == PynativeAdjoint::FuncGraphType::kBackwardPropagate) {
      (void)BackPropagateOneCNodeWithBPropFuncGraph(cnode, iter->second, fg, by_value);
    } else {
      (void)BackPropagateOneCNodeWithFPropFuncGraph(cnode, iter->second, fg, by_value);
    }
  }
  return true;
}

bool KPynativeCellImpl::AllReferencesStopped(const CNodePtr &curr_cnode) {
  // If all CNode use curr_cnode has stop_gradient_ flag, then curr_cnode also can set that flag.
  MS_EXCEPTION_IF_NULL(curr_cnode);
  const auto &iter = anfnode_to_adjoin_.find(curr_cnode);
  if (iter == anfnode_to_adjoin_.end()) {
    MS_LOG(EXCEPTION) << "Cannot find adjoint for cnode: " << curr_cnode->DebugString();
  }
  auto users = iter->second->users();
  if (users.empty()) {
    return false;
  }
  auto all_users_have_stopped = std::all_of(users.cbegin(), users.cend(), [](const AnfNodePtr &user) {
    if (!user->isa<CNode>() || !user->cast<CNodePtr>()->stop_gradient()) {
      return false;
    }
    return true;
  });
  if (curr_cnode == last_node_) {
    all_users_have_stopped = false;
  }
  return all_users_have_stopped;
}

void KPynativeCellImpl::PropagateStopGradient() {
  // propagate need_stop_gradient_ to cnode before back propagate;
  if (need_propagate_stop_gradient_) {
    for (auto iter = anfnode_to_adjoin_.rbegin(); iter != anfnode_to_adjoin_.rend(); ++iter) {
      const auto &node = iter->first;
      if (node->isa<CNode>()) {
        auto cnode = node->cast<CNodePtr>();
        if (!cnode->stop_gradient()) {
          // Cut off the cnode only when it's not referred any more
          if (IsPrimitiveCNode(cnode, prim::kPrimStopGradient) || IsPrimitiveCNode(cnode, prim::kPrimUpdateState) ||
              AllReferencesStopped(cnode)) {
            MS_LOG(DEBUG) << "Set stop_gradient flag for " << cnode->DebugString();
            cnode->set_stop_gradient(true);
          }
        }
      }
    }
  }
}

FuncGraphPtr KPynativeCellImpl::BuildBPropCutFuncGraph(const PrimitivePtr &prim, const CNodePtr &cnode) const {
  auto inputs_num = cnode->size() - 1;

  auto func_graph = std::make_shared<FuncGraph>();
  std::vector<AnfNodePtr> outputs;

  auto prim_py = prim->cast<PrimitivePyPtr>();
  MS_EXCEPTION_IF_NULL(prim_py);
  auto bprop_cut = std::make_shared<PrimitivePy>("bprop_cut");
  bprop_cut->CopyHookFunction(prim_py);
  prim_py->AddBpropCutPrim(bprop_cut);

  auto cell_id = GetValue<std::string>(prim->GetAttr("cell_id"));
  if (cell_id != "") {
    (void)bprop_cut->AddAttr("cell_hook", MakeValue(true));
    (void)bprop_cut->AddAttr("cell_id", MakeValue(cell_id));
  }

  outputs.push_back(NewValueNode(bprop_cut));
  for (size_t i = 0; i < inputs_num; ++i) {
    auto param = func_graph->add_parameter();
    outputs.push_back(param);
  }
  // out, dout
  auto p1 = func_graph->add_parameter();
  auto p2 = func_graph->add_parameter();
  outputs.push_back(p1);
  outputs.push_back(p2);

  func_graph->set_output(func_graph->NewCNode(outputs));
  return func_graph;
}

FuncGraphPtr KPynativeCellImpl::BuildMakeSequenceBprop(const PrimitivePtr &prim, const CNodePtr &cnode) const {
  auto inputs_num = cnode->size() - 1;
  CacheKey key{prim->name(), inputs_num};
  const auto &bprop_func_graph_iter = bprop_func_graph_cache.find(key);
  if (bprop_func_graph_iter != bprop_func_graph_cache.end()) {
    return bprop_func_graph_iter->second;
  }

  FuncGraphPtr b = std::make_shared<FuncGraph>();

  std::ostringstream ss;
  ss << "◀" << prim->ToString() << inputs_num;
  b->debug_info()->set_name(ss.str());
  for (size_t i = 0; i < inputs_num; ++i) {
    auto param = b->add_parameter();
    MS_EXCEPTION_IF_NULL(param);
  }
  // out, dout
  auto p1 = b->add_parameter();
  MS_EXCEPTION_IF_NULL(p1);
  AnfNodePtr dout = b->add_parameter();

  std::vector<AnfNodePtr> grads;
  PrimitivePtr getitem_prim;

  if (IsPrimitiveEquals(prim, prim::kPrimMakeTuple)) {
    getitem_prim = prim::kPrimTupleGetItem;
  } else if (IsPrimitiveEquals(prim, prim::kPrimMakeList)) {
    getitem_prim = prim::kPrimListGetItem;
  } else {
    MS_LOG(EXCEPTION) << "Prim should be MakeTuple or MakeList, Invalid prim: " << prim->ToString();
  }

  grads.push_back(NewValueNode(prim));
  for (size_t i = 0; i < inputs_num; ++i) {
    grads.push_back(b->NewCNode({NewValueNode(getitem_prim), dout, NewValueNode(SizeToLong(i))}));
  }

  b->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  b->set_output(b->NewCNode(grads));

  bprop_func_graph_cache[key] = b;
  return b;
}

void KPynativeCellImpl::SetSensAndWeights(const AnfNodePtrList &weights, bool has_sens_arg) {
  MS_EXCEPTION_IF_NULL(last_node_);
  MS_LOG(DEBUG) << "Last node info " << last_node_->DebugString();
  const auto &last_node_adjoint_iter = anfnode_to_adjoin_.find(last_node_);
  if (last_node_adjoint_iter == anfnode_to_adjoin_.end()) {
    MS_LOG(EXCEPTION) << "BackPropagate adjoint does not exist for input: " << last_node_->DebugString();
  }
  // Add sens parameter
  if (has_sens_arg) {
    auto sens_param = tape_->add_parameter();
    sens_param->debug_info()->set_name("sens");
    sens_param->set_abstract(last_node_->abstract()->Broaden());
    // Set dout of last node to sens;
    last_node_adjoint_iter->second->AccumulateDout(sens_param);
  } else {
    auto sens_node = BuildOnesLikeValue(tape_, last_node_adjoint_iter->second->out(), oneslike_sens_value_);
    last_node_adjoint_iter->second->AccumulateDout(sens_node);
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
    p->set_name(input_w->name());
    p->set_default_param(input_w->default_param());
  }
}

AnfNodePtr KPynativeCellImpl::GetGradNodeByIndex(const AnfNodePtrList &node_list, size_t index) const {
  if (index >= node_list.size()) {
    MS_LOG(EXCEPTION) << "Position index " << index << " is exceed input size " << node_list.size();
  }
  auto grad_node = node_list[index];
  MS_EXCEPTION_IF_NULL(grad_node);

  const auto &input_adjoint_iter = anfnode_to_adjoin_.find(grad_node);
  if (input_adjoint_iter == anfnode_to_adjoin_.end()) {
    if (grad_node->isa<Parameter>()) {
      // If weight is not used in the forward network, just return zeros_like() as dout.
      auto w = grad_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(w);
      const auto &default_param = w->default_param();
      MS_EXCEPTION_IF_NULL(default_param);
      if (last_node_->isa<ValueNode>()) {
        auto last_node_value = last_node_->cast<ValueNodePtr>();
        if (last_node_value->value() == default_param) {
          return BuildOnesLikeValue(tape_, default_param, oneslike_sens_value_);
        }
      }
      MS_LOG(WARNING) << "Weight does not participate in forward calculation, weight: " << grad_node->DebugString();
      return BuildZerosLikeValue(tape_, default_param);
    }
    // If input is not used in the forward network, just return zeros_like() as dout.
    MS_LOG(WARNING) << "Input does not participate in forward calculation, input: " << grad_node->DebugString();
    return BuildZerosLikeNode(tape_, grad_node);
  }
  return input_adjoint_iter->second->RealDout();
}

AnfNodePtr KPynativeCellImpl::GetInputGrad(bool grad_all_inputs, bool get_by_position,
                                           const std::vector<size_t> &grad_position) const {
  std::vector<size_t> grad_pos_list;
  if (get_by_position) {
    // If grad call from ops.grad, get_by_position is true by default. So if cell_inputs_ is empty indicate input are
    // all not tensor
    if (cell_inputs_.empty()) {
      return nullptr;
    }
    grad_pos_list = grad_position;
  } else if (grad_all_inputs) {
    grad_pos_list.resize(cell_inputs_.size());
    iota(grad_pos_list.begin(), grad_pos_list.end(), 0);
  } else {
    return nullptr;
  }

  AnfNodePtrList inputs_grad_list{NewValueNode(prim::kPrimMakeTuple)};
  AbstractBasePtrList inputs_grad_spec;
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
  auto input_grad_ret = tape_->NewCNode(inputs_grad_list);
  input_grad_ret->set_abstract(std::make_shared<abstract::AbstractTuple>(inputs_grad_spec));
  return input_grad_ret;
}

AnfNodePtr KPynativeCellImpl::GetWeightGrad(bool grad_weights, bool weight_param_is_tuple,
                                            const AnfNodePtrList &weights) const {
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

void KPynativeCellImpl::SetOutput(const AnfNodePtrList &weights, const std::vector<size_t> &grad_position,
                                  const GradAttr &grad_attr) {
  auto inputs_grad_ret = GetInputGrad(grad_attr.grad_all_inputs, grad_attr.get_by_position, grad_position);
  auto weights_grad_ret = GetWeightGrad(grad_attr.grad_weights, grad_attr.weight_param_is_tuple, weights);
  // Gradients wrt inputs and weights.
  if (inputs_grad_ret != nullptr && weights_grad_ret != nullptr) {
    auto tape_output = tape_->NewCNode({NewValueNode(prim::kPrimMakeTuple), inputs_grad_ret, weights_grad_ret});
    tape_output->set_abstract(std::make_shared<abstract::AbstractTuple>(
      abstract::AbstractBasePtrList{inputs_grad_ret->abstract(), weights_grad_ret->abstract()}));
    tape_->set_output(tape_output);
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

bool KPynativeCellImpl::BuildKNode() {
  for (auto iter = anfnode_to_adjoin_.cbegin(); iter != anfnode_to_adjoin_.cend(); ++iter) {
    if (!iter->first->isa<CNode>()) {
      continue;
    }

    AnfNodePtrList node_list;
    auto cnode = iter->first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    for (size_t i = 0; i < cnode->inputs().size(); ++i) {
      (void)node_list.emplace_back(BuildKNodeForCNodeInput(iter->second, cnode->input(i), i));
    }
    auto k_node = tape_->NewCNode(node_list);
    k_node->set_abstract(iter->second->out()->ToAbstract()->Broaden());
    iter->second->set_k_node(k_node);
  }
  return true;
}

CNodePtr KPynativeCellImpl::GetBPropFromFProp(const FuncGraphPtr &fprop_fg, const AnfNodePtrList &args) {
  // Wrap tuple_getitem(fprop_app, 1) in a FuncGraph and optimize it;
  auto bprop_builder = std::make_shared<FuncGraph>();
  bprop_builder->debug_info()->set_name("bprop_builder");

  AnfNodePtrList fprop_app_inputs{NewValueNode(fprop_fg)};
  AnfNodePtrList bprop_builder_inputs;
  for (const auto &arg : args) {
    auto param = bprop_builder->add_parameter();
    fprop_app_inputs.push_back(param);
    bprop_builder_inputs.push_back(arg);
  }
  auto fprop_app = bprop_builder->NewCNode(fprop_app_inputs);
  auto get_bprop =
    bprop_builder->NewCNode({NewValueNode(prim::kPrimTupleGetItem), fprop_app, NewValueNode(static_cast<int64_t>(1))});
  bprop_builder->set_output(get_bprop);
  (void)bprop_builder_inputs.insert(bprop_builder_inputs.cbegin(), NewValueNode(bprop_builder));
  get_bprop = tape_->NewCNode(bprop_builder_inputs);

  return get_bprop;
}

void KPynativeCellImpl::ReplacePrimalParameter(const AnfNodePtrList &weights, bool has_sens_arg) {
  auto mng = MakeManager({tape_}, false);
  auto tr = mng->Transact();
  const auto &parameters = tape_->parameters();
  auto cell_inputs_size = cell_inputs_.size();
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
}

void ClearKPynativeCellStaticRes() {
  irpass = nullptr;
  add_ops = nullptr;
  ones_like_ops = nullptr;
  zeros_like_ops = nullptr;
  g_k_prims_pynative.clear();
  bprop_func_graph_cache.clear();
  zeros_like_funcgraph_cache.clear();
  ones_like_funcgraph_cache.clear();
}
}  // namespace ad
}  // namespace mindspore
