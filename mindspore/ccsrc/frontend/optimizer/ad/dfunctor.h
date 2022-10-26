/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_D_FUNCTOR_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_D_FUNCTOR_H_

#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <utility>
#include <unordered_map>

#include "utils/hash_map.h"
#include "ir/anf.h"
#include "ir/meta_func_graph.h"
#include "ir/func_graph_cloner.h"
#include "pipeline/jit/resource.h"
#include "frontend/optimizer/ad/adjoint.h"
#include "frontend/optimizer/ad/pynative_dfunctor.h"
#include "frontend/operator/ops.h"
#include "pipeline/jit/debug/trace.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace ad {
using Registry = std::unordered_map<PrimitivePtr, FuncGraphPtr, PrimitiveHasher, PrimitiveTotalEqual>;
class KPrim;
extern KPrim g_k_prims;
class DFunctor;
using DFunctorPtr = std::shared_ptr<DFunctor>;

// Flag to control if fv should be lifted before grad. If this lift_fv feature is mature, then this flag can be removed.
extern bool lift_fv_before_grad;

// D Functor's rules to map closure object and morphisms.
class DFunctor : public std::enable_shared_from_this<DFunctor> {
 public:
  DFunctor(const FuncGraphPtr &primal_graph, const pipeline::ResourceBasePtr &resources);
  ~DFunctor() = default;
  // Map object in D category to K category.
  void MapObject();
  // Map morphism in D category to K category.
  void MapMorphism();
  FuncGraphPtr k_graph();
  FuncGraphPtr tape();
  // Construct user defined k object.
  FuncGraphPtr KUserDefined(const FuncGraphPtr &primal);
  // Register functor objects to form a global view.
  void Init(bool is_top = false);
  void Finish();

  // Clear resources.
  static void Clear();

  friend class PynativeDFunctor;

 private:
  // Map one morphism.
  AdjointPtr MapMorphism(const AnfNodePtr &morph);
  bool IsFreeMorphism(const AnfNodePtr &node);
  // Map morphism that's not attached to output.
  void MapFreeMorphism();
  void BackPropagateFv(const AnfNodePtr &fv, const AnfNodePtr &din);
  void BackPropagateSwitchLayer(const CNodePtr &cnode_morph, const CNodePtr &env);
  void BackPropagate(const CNodePtr &cnode_morph, const CNodePtr &k_app, const AdjointPtr &node_adjoint);
  AnfNodePtr AttachFvDoutToTape(const AnfNodePtr &grad_fv);
  AnfNodePtr AttachIndirectFvDoutToTape(const AnfNodePtr &grad_fv);
  // Map CNode/Index of Primitive to K.
  AnfNodePtr MapPrimitiveToK(const CNodePtr &primitive_user, size_t index);
  // Map ValueNode of FuncGraph to K.
  AnfNodePtr MapFuncGraphToK(const AnfNodePtr &primal);
  // Map ValueNode of Parameter to K.
  AnfNodePtr MapParameterToK(const AnfNodePtr &primal);
  // MapObject impls.
  void MapFvObject();
  void MapValueObject();
  void MapParamObject();
  // Find adjoint with its primary k.
  AdjointPtr FindAdjoint(const AnfNodePtr &primal) const;
  // Broadcast stop flags.
  void BroadCastStopFlag();
  bool AllReferencesStopped(const CNodePtr &node);
  // Update k hole with adjoint_definition, only applied in recursive case.
  void UpdateAdjoint(const AdjointPtr &adjoint_definition);
  void CallDoutHoleOnTape() const;
  // Replace the primal graph with k graph
  void EliminatePrimalGraph();
  // Pynative specialize
  ValueNodePtr GenNewTensor(const CNodePtr &forward_node);
  tensor::TensorPtr GenNewTensorInner(const TypePtr &type_elem, const BaseShapePtr &shape_elem);
  void GetForwardOutNodeAndBpropGraph(const CNodePtr &k_app, CNodePtr *forward_node, FuncGraphPtr *bprop_graph,
                                      FuncGraphPtr *fprop_graph);
  std::vector<AnfNodePtr> RunOutputReplace(const CNodePtr &forward_node, const FuncGraphPtr &bprop_graph,
                                           const FuncGraphPtr &fprop_graph, const CNodePtr &cnode_morph);
  std::vector<AnfNodePtr> RunInputReplace(const FuncGraphPtr &bprop_graph, const FuncGraphPtr &fprop_graph,
                                          const CNodePtr &cnode_morph);
  void ReplaceEquivdout(const CNodePtr &k_app, const CNodePtr &cnode_morph);

  mindspore::HashMap<AnfNodePtr, AdjointPtr> anfnode_to_adjoin_;
  // Cache for indirect fv backpropagation, K o K can only do backprop layer by layer.
  mindspore::HashMap<AnfNodePtr, AdjointPtr> anfnode_to_adjoin_indirect_fv_;
  // Cache for fv node -> pair<embed<fv_node>, zeros_like<fv_node>>, so EnvironGetTransform in optimizer
  // can hit its cache if fv_node is same.
  mindspore::HashMap<AnfNodePtr, std::pair<CNodePtr, CNodePtr>> anfnode_to_envitem_;
  FuncGraphPtr primal_graph_;
  // K object for primal_graph_;
  FuncGraphPtr k_graph_;
  // The Backprop part of k_graph_.
  FuncGraphPtr tape_;
  // Dout parameter for primal_graph_.
  AnfNodePtr dout_;
  pipeline::ResourceBasePtr resources_;
  // Cut off stopped objects in category D.
  bool need_cut_;
  bool is_top_;
  static mindspore::HashMap<FuncGraphPtr, std::shared_ptr<DFunctor>> func_graph_to_functor_;
  static mindspore::HashMap<AnfNodePtr, AdjointPtr> anfnode_to_adjoin_definition_;
};

// D Functor's rules to map primitive object.
class KPrim {
 public:
  KPrim() = default;
  ~KPrim() = default;

  FuncGraphPtr KPrimitive(const CNodePtr &cnode, const ValueNodePtr &value_node,
                          const pipeline::ResourceBasePtr &resources);
  MetaFuncGraphPtr KMetaFuncGraph(const PrimitivePtr &prim);
  // bprop_fg and primal_fg in bprop_fg's transforms are FuncGraph just after convert.
  // current_primal_fg is the specialized and AutoMonaded primal_fg.
  FuncGraphPtr KUserDefinedCellBprop(const FuncGraphPtr &bprop_fg, const FuncGraphPtr &current_primal_fg);

  bool CheckCustomVjp(const FuncGraphPtr &bprop_fg) const;
  FuncGraphPtr GetCustomVjpBprop(const FuncGraphPtr &bprop_fg);
  void clear() {
    bprop_registry_meta_.clear();
    bprop_registry_.clear();
  }
  FuncGraphPtr GetPossibleBprop(const PrimitivePtr &prim);

 private:
  FuncGraphPtr GetFprop(const PrimitivePtr &prim) const;
  FuncGraphPtr GetPrimBprop(const PrimitivePtr &prim, const ValueNodePtr &value_node,
                            const pipeline::ResourceBasePtr &resources);
  FuncGraphPtr FakeBprop(const ValueNodePtr &value_node, const pipeline::ResourceBasePtr &resources) const;
  FuncGraphPtr BpropCut(const ValueNodePtr &value_node, const pipeline::ResourceBasePtr &resources) const;
  // Given a bprop rule, do the K mapping.
  // current_primal_fg is only valid for user defined bprop for Cell, not for Primitive.
  // Refer the comment in KUserDefinedCellBprop.
  template <typename T>
  FuncGraphPtr BpropToK(const T &primal, const FuncGraphPtr &bprop_fg, const FuncGraphPtr &current_primal_fg,
                        const CNodePtr &cnode, const mindspore::HashMap<std::string, ValuePtr> &primal_attrs,
                        const std::vector<NodeDebugInfoPtr> &primal_debug_infos);
  AnfNodePtr BuildOutput(const FuncGraphPtr &bprop_fg, const FuncGraphPtr &current_primal_fg) const;
  void TransformArgsForPrimitive(const FuncGraphManagerPtr &mng, const FuncGraphPtr &bprop_fg,
                                 const PrimitivePtr &primitive, const FuncGraphPtr &outer,
                                 std::vector<AnfNodePtr> *const transf_args) const;
  template <typename T>
  void TransformArgsForFuncGraph(const FuncGraphManagerPtr &mng, const FuncGraphPtr &bprop_fg,
                                 const T &current_primal_fg, const FuncGraphPtr &outer,
                                 std::vector<AnfNodePtr> *const transf_args) const;
  void CheckBprop(const FuncGraphPtr &bprop_fg, const string &prim_to_check) const;

  Registry bprop_registry_;
  mindspore::HashMap<PrimitivePtr, MetaFuncGraphPtr> bprop_registry_meta_;
};

template <typename T>
FuncGraphPtr KPrim::BpropToK(const T &primal, const FuncGraphPtr &bprop_fg, const FuncGraphPtr &current_primal_fg,
                             const CNodePtr &cnode, const mindspore::HashMap<std::string, ValuePtr> &primal_attrs,
                             const std::vector<NodeDebugInfoPtr> &primal_debug_infos) {
  MS_EXCEPTION_IF_NULL(primal);
  MS_EXCEPTION_IF_NULL(bprop_fg);
  CheckBprop(bprop_fg, primal->ToString());

  FuncGraphPtr cloned_bprop_fg;
  {
    PrimalAttrGuard primal_attr_guard(primal_attrs);
    PrimalDebugInfoGuard primal_debug_info_guard(primal_debug_infos);
    cloned_bprop_fg = BasicClone(bprop_fg);
  }
  MS_EXCEPTION_IF_NULL(cloned_bprop_fg);

  GraphDebugInfoPtr debug_info = nullptr;
  {
    TraceGuard guard(std::make_shared<TraceCopy>(bprop_fg->debug_info()));
    debug_info = std::make_shared<GraphDebugInfo>();
  }
  if (debug_info->trace_info() != nullptr && debug_info->trace_info()->debug_info() != nullptr) {
    debug_info->trace_info()->debug_info()->set_name(primal->ToString());
  }
  cloned_bprop_fg->debug_info()->set_name("");
  cloned_bprop_fg->debug_info()->set_trace_info(std::make_shared<TraceGradBprop>(debug_info));

  // Make sure (out, dout) provided.
  if (cloned_bprop_fg->parameters().size() < 2) {
    MS_LOG(EXCEPTION)
      << "The function 'bprop' of Primitive or Cell requires at least 2 params 'out' and 'dout', but got only "
      << cloned_bprop_fg->parameters().size() << ".\n"
      << trace::GetDebugInfo(cloned_bprop_fg->debug_info());
  }
  AnfNodePtr bout = BuildOutput(cloned_bprop_fg, current_primal_fg);
  cloned_bprop_fg->set_output(bout);

  FuncGraphPtr outer = nullptr;
  {
    auto outer_debug_info = std::make_shared<GraphDebugInfo>();
    outer_debug_info->set_name(primal->ToString());
    TraceGuard guard(std::make_shared<TraceGradFprop>(outer_debug_info));
    outer = std::make_shared<FuncGraph>();
    (void)outer->transforms().emplace("primal", FuncGraphTransform(primal));
    outer->set_output(NewValueNode(kNone));
  }

  auto mng = Manage({cloned_bprop_fg, outer}, false);

  // In a bprop definition, the last two param should be out and dout.
  auto param_size = cloned_bprop_fg->parameters().size();
  auto param_num = param_size - 1;
  auto dout = cloned_bprop_fg->parameters()[param_num];
  param_num--;
  auto out_param = cloned_bprop_fg->parameters()[param_num];

  std::vector<AnfNodePtr> transf_args;

  if constexpr (std::is_same<T, PrimitivePtr>::value) {
    PrimitivePtr primitive = primal;
    TransformArgsForPrimitive(mng, cloned_bprop_fg, primal, outer, &transf_args);
    (void)transf_args.insert(transf_args.cbegin(), NewValueNode(primal));
  } else {
    TransformArgsForFuncGraph(mng, cloned_bprop_fg, current_primal_fg, outer, &transf_args);
    (void)transf_args.insert(transf_args.cbegin(), NewValueNode(current_primal_fg));
  }
  CNodePtr out_value = nullptr;
  if (cnode != nullptr) {  // Set equiv debug info. for Primitive CNode out.
    TraceGuard trace_guard(std::make_shared<TraceEquiv>(cnode->debug_info()));
    out_value = outer->NewCNode(transf_args);
    if constexpr (std::is_same<T, PrimitivePtr>::value) {
      out_value->CloneCNodeInfo(cnode);
    }
  } else {
    out_value = outer->NewCNode(transf_args);
  }
  (void)mng->Replace(out_param, out_value);

  TraceGuard guard(std::make_shared<TraceGradSens>(out_param->debug_info()));
  auto new_dout = cloned_bprop_fg->add_parameter();
  (void)mng->Replace(dout, new_dout);
  // We remove all parameters except new_dout.
  std::vector<AnfNodePtr> newBpropParams = {new_dout};
  cloned_bprop_fg->set_parameters(newBpropParams);
  outer->set_output(outer->NewCNode({NewValueNode(prim::kPrimMakeTuple), out_value, NewValueNode(cloned_bprop_fg)}));
  return BasicClone(outer);
}

// Handle bprob of op which input dtype is real number and output dtype is complex number.
// If the dtype of a gradient(din) is complex number and the input of that is real number,
// only the real part of the gradient make sense in back propagate. So we handle it by
// insert a Real() ops after the gradient.
// input: AnfNode with input of op which input dtype is real number and output dtype is complex number.
// din: CNodePtr with gradient of input.
// fg: Funcgraph witch input and din belong to.
// return: New din with inserted real op if necessarily.
AnfNodePtr HandleRealToComplex(const AnfNodePtr &input, const CNodePtr &din, const FuncGraphPtr &fg);
}  // namespace ad
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_D_FUNCTOR_H_
