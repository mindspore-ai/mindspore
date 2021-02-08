/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <unordered_map>
#include <vector>
#include <iostream>
#include <utility>

#include "ir/anf.h"
#include "ir/meta_func_graph.h"
#include "ir/func_graph_cloner.h"
#include "pipeline/jit/resource.h"
#include "frontend/optimizer/ad/adjoint.h"
#include "frontend/operator/ops.h"
#include "debug/trace.h"

namespace mindspore {
namespace ad {
struct PrimitiveTotalEqual {
  bool operator()(PrimitivePtr const &t1, PrimitivePtr const &t2) const {
    MS_EXCEPTION_IF_NULL(t1);
    MS_EXCEPTION_IF_NULL(t2);
    return *t1 == *t2;
  }
};

using Registry = std::unordered_map<PrimitivePtr, FuncGraphPtr, PrimitiveHasher, PrimitiveTotalEqual>;
class KPrim;
extern KPrim g_k_prims;
class DFunctor;
using DFunctorPtr = std::shared_ptr<DFunctor>;

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
  bool IsInScope(const AnfNodePtr &node);

  // Clear resources.
  static void Clear();

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
  AdjointPtr FindAdjoint(const AnfNodePtr &primal);
  // Broadcast stop flags.
  void BroadCastStopFlag();
  bool AllReferencesStopped(const CNodePtr &node);
  // Update k hole with adjoint_definition, only applied in recursive case.
  void UpdateAdjoint(const AdjointPtr &adjoint_definition);
  void CallDoutHoleOnTape();
  // Replace the primal graph with k graph
  void EliminatePrimalGraph();
  // Pynative specialize
  void ReplaceEquivdout(const CNodePtr &cnode, const CNodePtr &cnode_morph);
  ValuePtr GenNewTensorInner(const ValuePtr &value);
  ValuePtr GenNewTensor(const FuncGraphManagerPtr &mng, const AnfNodePtr &node, const ValuePtr &value,
                        bool need_replace_forward);

  std::unordered_map<AnfNodePtr, AdjointPtr> anfnode_to_adjoin_;
  // Cache for indirect fv backpropagation, K o K can only do backprop layer by layer.
  std::unordered_map<AnfNodePtr, AdjointPtr> anfnode_to_adjoin_indirect_fv_;
  // Cache for fv node -> pair<embed<fv_node>, zeros_like<fv_node>>, so EnvGetItemTransform in optimizer
  // can hit its cache if fv_node is same.
  std::unordered_map<AnfNodePtr, std::pair<CNodePtr, CNodePtr>> anfnode_to_envitem_;
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
  static std::unordered_map<FuncGraphPtr, std::shared_ptr<DFunctor>> func_graph_to_functor_;
  static std::unordered_map<AnfNodePtr, AdjointPtr> anfnode_to_adjoin_definition_;
  static FuncGraphSet scope_;
};

// D Functor's rules to map primitive object.
class KPrim {
 public:
  KPrim() = default;
  ~KPrim() = default;

  FuncGraphPtr KPrimitive(const CNodePtr &primal_user, const ValueNodePtr &value_node,
                          const pipeline::ResourceBasePtr &resources);
  MetaFuncGraphPtr KMetaFuncGraph(const PrimitivePtr &prim);
  // bprop_fg and primal_fg in bprop_fg's transforms are FuncGraph just after convert.
  // current_primal_fg is the specialized and AutoMonaded primal_fg.
  FuncGraphPtr KUserDefinedCellBprop(const FuncGraphPtr &bprop_fg, const FuncGraphPtr &current_primal_fg);

  void clear() {
    bprop_registry_meta_.clear();
    bprop_registry_.clear();
  }

 private:
  FuncGraphPtr GetBprop(const PrimitivePtr &prim);
  FuncGraphPtr GetFprop(const PrimitivePtr &prim);
  FuncGraphPtr FakeBprop(const ValueNodePtr &value_node, const pipeline::ResourceBasePtr &resources);
  FuncGraphPtr BpropCut(const ValueNodePtr &value_node, const pipeline::ResourceBasePtr &resources);
  // Given a bprop rule, do the K mapping.
  // current_primal_fg is only valid for user defined bprop for Cell, not for Primitive.
  // Refer the comment in KUserDefinedCellBprop.
  template <typename T>
  FuncGraphPtr BpropToK(const T &primal, const FuncGraphPtr &bprop_g, const FuncGraphPtr &current_primal_fg,
                        const CNodePtr &cnode);
  AnfNodePtr BuildOutput(const FuncGraphPtr &bprop_fg, const FuncGraphPtr &current_primal_fg);
  void TransformArgsForPrimitive(const FuncGraphManagerPtr &mng, const FuncGraphPtr &bprop_fg,
                                 const PrimitivePtr &primitive, const FuncGraphPtr &outer,
                                 std::vector<AnfNodePtr> *const transf_args);
  template <typename T>
  void TransformArgsForFuncGraph(const FuncGraphManagerPtr &mng, const FuncGraphPtr &bprop_fg,
                                 const T &current_primal_fg, const FuncGraphPtr &outer,
                                 std::vector<AnfNodePtr> *const transf_args);
  void CheckBprop(const FuncGraphPtr &bprop_fg, const string &prim_to_check);

  Registry bprop_registry_;
  std::unordered_map<PrimitivePtr, MetaFuncGraphPtr> bprop_registry_meta_;
};

template <typename T>
FuncGraphPtr KPrim::BpropToK(const T &primal, const FuncGraphPtr &bprop_fg, const FuncGraphPtr &current_primal_fg,
                             const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(primal);
  MS_EXCEPTION_IF_NULL(bprop_fg);
  CheckBprop(bprop_fg, primal->ToString());

  auto debug_info = std::make_shared<GraphDebugInfo>();
  debug_info->set_name(primal->ToString());

  auto cloned_bprop_fg = BasicClone(bprop_fg);
  MS_EXCEPTION_IF_NULL(cloned_bprop_fg);

  cloned_bprop_fg->debug_info()->set_name("");
  cloned_bprop_fg->debug_info()->set_trace_info(std::make_shared<TraceGradBprop>(debug_info));

  // Make sure (out, dout) provided.
  if (cloned_bprop_fg->parameters().size() < 2) {
    MS_LOG(EXCEPTION) << "Primitive or Cell " << primal->ToString()
                      << " bprop requires out and dout at least, but only got " << cloned_bprop_fg->parameters().size()
                      << " params. NodeInfo: " << trace::GetDebugInfo(cloned_bprop_fg->debug_info());
  }
  AnfNodePtr bout = BuildOutput(cloned_bprop_fg, current_primal_fg);
  cloned_bprop_fg->set_output(bout);

  FuncGraphPtr outer = nullptr;
  {
    TraceGuard guard(std::make_shared<TraceGradFprop>(debug_info));
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
    (void)transf_args.insert(transf_args.begin(), NewValueNode(primal));
  } else {
    TransformArgsForFuncGraph(mng, cloned_bprop_fg, current_primal_fg, outer, &transf_args);
    (void)transf_args.insert(transf_args.begin(), NewValueNode(current_primal_fg));
  }
  CNodePtr out_value = nullptr;
  if (cnode != nullptr) {  // Set equiv debug info. for Primitive CNode out.
    TraceGuard trace_guard(std::make_shared<TraceEquiv>(cnode->debug_info()));
    out_value = outer->NewCNode(transf_args);
    if constexpr (std::is_same<T, PrimitivePtr>::value) {
      out_value->CloneUserData(cnode);
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
}  // namespace ad
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_D_FUNCTOR_H_
