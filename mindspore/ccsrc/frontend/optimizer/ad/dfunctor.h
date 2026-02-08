/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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
#include "utils/trace_info.h"
#include "primitive/sequence_ops.h"
#include "ir/anf.h"
#include "ir/meta_func_graph.h"
#include "ir/func_graph_cloner.h"
#include "ir/primal_attr.h"
#include "ir/primal_debug_info.h"
#include "frontend/jit/ps/resource.h"
#include "frontend/optimizer/ad/adjoint.h"
#include "frontend/optimizer/ad/saved_tensors_hooks.h"
#include "frontend/operator/ops.h"
#include "frontend/jit/ps/debug/trace.h"
#include "include/utils/utils.h"
#include "ir/func_graph_flag.h"

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
  DFunctor(const FuncGraphPtr &primal_graph, const pipeline::ResourceBasePtr &resources, bool is_top,
           bool is_grad_by_j = false);
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
  void BackPropagate(const CNodePtr &cnode_morph, const AdjointPtr &node_adjoint);
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

  void AccumulateInputGradients(const CNodePtr &cnode_morph, const AdjointPtr &node_adjoint, const CNodePtr bprop_app);
  AnfNodePtr ApplyBackwardPreHook(const AnfNodePtr &dout);
  AnfNodePtr ApplyBackwardHook(const AdjointPtr &node_adjoint);

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
  bool is_grad_by_j_;
};

// D Functor's rules to map primitive object.
class KPrim {
 public:
  KPrim() = default;
  ~KPrim() = default;

  FuncGraphPtr KPrimitive(const CNodePtr &cnode, const ValueNodePtr &value_node,
                          const pipeline::ResourceBasePtr &resources);
  MetaFuncGraphPtr KMetaFuncGraph(const PrimitivePtr &prim, const AnfNodePtr &node);
  // bprop_fg and primal_fg in bprop_fg's transforms are FuncGraph just after convert.
  // current_primal_fg is the specialized and AutoMonaded primal_fg.
  FuncGraphPtr KUserDefinedCellBprop(const FuncGraphPtr &bprop_fg, const FuncGraphPtr &current_primal_fg);

  bool CheckCustomVjp(const FuncGraphPtr &bprop_fg) const;
  FuncGraphPtr GetCustomVjpBprop(const FuncGraphPtr &bprop_fg) const;
  void clear() {
    bprop_registry_meta_.clear();
    bprop_registry_.clear();
  }

 private:
  FuncGraphPtr GetFprop(const PrimitivePtr &prim) const;
  FuncGraphPtr GetPrimBprop(const PrimitivePtr &prim, const ValueNodePtr &value_node,
                            const pipeline::ResourceBasePtr &resources, const CNodePtr &cnode = nullptr);
  FuncGraphPtr FakeBprop(const ValueNodePtr &value_node, const pipeline::ResourceBasePtr &resources) const;
  FuncGraphPtr BpropCut(const ValueNodePtr &value_node, const pipeline::ResourceBasePtr &resources) const;
  // Given a bprop rule, do the K mapping.
  // current_primal_fg is only valid for user defined bprop for Cell, not for Primitive.
  // Refer the comment in KUserDefinedCellBprop.
  template <typename T>
  FuncGraphPtr BpropToK(const T &primal, const FuncGraphPtr &bprop_fg, const FuncGraphPtr &current_primal_fg,
                        const CNodePtr &cnode, const mindspore::HashMap<std::string, ValuePtr> &primal_attrs,
                        const std::vector<NodeDebugInfoPtr> &primal_debug_infos);
  template <typename T>
  FuncGraphPtr CloneBprop(const T &primal, const FuncGraphPtr &bprop_fg, const CNodePtr &cnode,
                          const mindspore::HashMap<std::string, ValuePtr> &primal_attrs,
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
FuncGraphPtr KPrim::CloneBprop(const T &primal, const FuncGraphPtr &bprop_fg, const CNodePtr &cnode,
                               const mindspore::HashMap<std::string, ValuePtr> &primal_attrs,
                               const std::vector<NodeDebugInfoPtr> &primal_debug_infos) {
  FuncGraphPtr cloned_bprop_fg;
  {
    PrimalAttrGuard primal_attr_guard(primal_attrs);
    PrimalDebugInfoGuard primal_debug_info_guard(primal_debug_infos);
    if (bprop_fg->has_flag(mindspore::kFuncGraphFlagMetaFuncGraphBprop) &&
        (cnode == nullptr || !cnode->primal_attrs().empty())) {
      cloned_bprop_fg = BasicClone(bprop_fg, true);
    } else {
      cloned_bprop_fg = BasicClone(bprop_fg);
    }
  }
  MS_EXCEPTION_IF_NULL(cloned_bprop_fg);

  GraphDebugInfoPtr debug_info = nullptr;
  {
    TraceGuard guard(MakeTraceInfo<TraceCopy>(bprop_fg->debug_info()));
    debug_info = std::make_shared<GraphDebugInfo>();
  }
  if (debug_info->trace_info() != nullptr && debug_info->trace_info()->debug_info() != nullptr) {
    debug_info->trace_info()->debug_info()->set_name(primal->ToString());
  }
  cloned_bprop_fg->debug_info()->set_name("");
  cloned_bprop_fg->debug_info()->set_trace_info(MakeTraceInfo<TraceGradBprop>(debug_info));

  // Make sure (out, dout) provided.
  constexpr auto number_two = 2;
  if (cloned_bprop_fg->parameters().size() < number_two) {
    MS_LOG_WITH_NODE(EXCEPTION, cloned_bprop_fg->return_node())
      << "The function 'bprop' of Primitive or Cell requires at least 2 params 'out' and 'dout', but got only "
      << cloned_bprop_fg->parameters().size() << ".\n"
      << trace::GetDebugInfoStr(cloned_bprop_fg->debug_info());
  }
  return cloned_bprop_fg;
}

template <typename T>
FuncGraphPtr KPrim::BpropToK(const T &primal, const FuncGraphPtr &bprop_fg, const FuncGraphPtr &current_primal_fg,
                             const CNodePtr &cnode, const mindspore::HashMap<std::string, ValuePtr> &primal_attrs,
                             const std::vector<NodeDebugInfoPtr> &primal_debug_infos) {
  MS_EXCEPTION_IF_NULL(primal);
  MS_EXCEPTION_IF_NULL(bprop_fg);
  CheckBprop(bprop_fg, primal->ToString());

  auto cloned_bprop_fg = CloneBprop(primal, bprop_fg, cnode, primal_attrs, primal_debug_infos);

  AnfNodePtr bout = BuildOutput(cloned_bprop_fg, current_primal_fg);
  cloned_bprop_fg->set_output(bout);

  FuncGraphPtr outer = nullptr;
  {
    auto outer_debug_info = std::make_shared<GraphDebugInfo>();
    outer_debug_info->set_name(primal->ToString());
    TraceGuard guard(MakeTraceInfo<TraceGradFprop>(outer_debug_info));
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
    auto prim_recompute_attr = primitive->GetAttr(kAttrRecompute);
    if (prim_recompute_attr != nullptr && prim_recompute_attr->isa<BoolImm>() && GetValue<bool>(prim_recompute_attr)) {
      cloned_bprop_fg->set_flag(FUNC_GRAPH_RECOMPUTE_GRAD_GRAPH, true);
    }
    TransformArgsForPrimitive(mng, cloned_bprop_fg, primal, outer, &transf_args);
    (void)transf_args.insert(transf_args.cbegin(), NewValueNode(primal));
  } else {
    TransformArgsForFuncGraph(mng, cloned_bprop_fg, current_primal_fg, outer, &transf_args);
    (void)transf_args.insert(transf_args.cbegin(), NewValueNode(current_primal_fg));
  }
  CNodePtr out_value = nullptr;
  if (cnode != nullptr) {  // Set equiv debug info. for Primitive CNode out.
    TraceGuard trace_guard(MakeTraceInfo<TraceEquiv>(cnode->debug_info()));
    out_value = outer->NewCNode(transf_args);
    if constexpr (std::is_same<T, PrimitivePtr>::value) {
      out_value->CloneCNodeInfo(cnode);
    }
    const DebugInfoPtr &old_debug_info = cnode->debug_info();
    if (old_debug_info != nullptr && out_value->debug_info() != nullptr) {
      const auto &old_real_loc = old_debug_info->real_loc();
      if (!old_real_loc.empty()) {
        out_value->debug_info()->set_real_loc(old_real_loc);
      }
    }
  } else {
    out_value = outer->NewCNode(transf_args);
  }
  (void)mng->Replace(out_param, out_value);

  TraceGuard guard(MakeTraceInfo<TraceGradSens>(out_param->debug_info()));
  auto new_dout = cloned_bprop_fg->add_parameter();
  (void)mng->Replace(dout, new_dout);
  // We remove all parameters except new_dout.
  cloned_bprop_fg->set_parameters({new_dout});
  CNodePtr fprop_cnode = outer->NewCNode(prim::kPrimMakeTuple, {out_value, NewValueNode(cloned_bprop_fg)});
  outer->set_output(fprop_cnode);
  ApplySavedTensorsHooksOnK(outer, current_primal_fg, cnode, mng, transf_args);
  return BasicClone(outer);
}
}  // namespace ad
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_D_FUNCTOR_H_
