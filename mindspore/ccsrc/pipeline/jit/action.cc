/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/action.h"

#include <memory>
#include <utility>
#include <vector>
#include <set>
#include <string>
#include <algorithm>
#include <functional>

#include "ir/func_graph_cloner.h"
#include "ir/param_info.h"
#include "ir/cell.h"
#include "include/common/utils/python_adapter.h"
#include "include/common/utils/anfalgo.h"
#include "abstract/abstract_value.h"
#include "frontend/parallel/costmodel_context.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/graph_util/graph_splitter.h"
#include "pipeline/jit/pipeline.h"
#include "pipeline/jit/pass.h"
#include "pipeline/jit/parse/parse_base.h"
#include "pipeline/jit/parse/data_converter.h"
#include "pipeline/jit/static_analysis/auto_monad.h"
#include "pipeline/jit/static_analysis/order_enforce.h"
#include "pipeline/jit/static_analysis/static_analysis.h"
#include "pipeline/jit/static_analysis/async_eval_result.h"
#include "pipeline/jit/static_analysis/program_specialize.h"
#include "pipeline/jit/resource.h"
#include "pipeline/jit/remove_value_node_dup.h"
#include "pipeline/pynative/pynative_execute.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/ad/grad.h"
#include "frontend/optimizer/py_pass_manager.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "backend/graph_compiler/transform.h"
#include "load_mindir/infer_mindir.h"
#include "debug/data_dump/dump_json_parser.h"
#ifdef WITH_BACKEND
#include "ps/scheduler.h"
#include "distributed/cluster/cluster_context.h"
#endif

namespace mindspore {
namespace pipeline {
namespace {
bool ExistControlFlow(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  return !func_graph->func_graphs_used_total().empty();
}

bool EnableGradForScalar(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  return MsContext::GetInstance()->get_param<bool>(MS_CTX_GRAD_FOR_SCALAR) && abs->BuildType() != nullptr &&
         abs->BuildType()->isa<Number>();
}

bool EnableTupleBroaden(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  return abs->isa<abstract::AbstractTuple>() && abs->cast<abstract::AbstractTuplePtr>()->ContainsAllBroadenTensors();
}

void UpdateFuncGraphParameter(const FuncGraphPtr &func_graph, const std::vector<ValuePtr> &arguments) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> new_paras;
  for (size_t i = 0; i < func_graph->parameters().size(); ++i) {
    const auto &param = func_graph->parameters()[i];
    auto param_node = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_node);
    if (param_node->has_default()) {
      new_paras.push_back(param_node);
      continue;
    }

    // Handle the Parameter from input arguments.
    if (i < arguments.size()) {
      auto param_value = dyn_cast<tensor::MetaTensor>(arguments[i]);
      if (param_value != nullptr && param_value->is_parameter()) {
        param_node->set_default_param(param_value);
      }
    }

    AbstractBasePtr param_abs = param_node->abstract();
    MS_EXCEPTION_IF_NULL(param_abs);
    if (param_abs->BuildValue() == kAnyValue || EnableGradForScalar(param_abs) || EnableTupleBroaden(param_abs)) {
      new_paras.push_back(param_node);
    } else {
      MS_LOG(INFO) << "Remove the " << i << "th parameter, since it's passed a constant argument.";
    }
  }
  func_graph->set_parameters(new_paras);
}

bool IsDynamicShapeGraph(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return());
  return std::any_of(node_list.begin(), node_list.end(), [](const AnfNodePtr &node) {
    if (common::AnfAlgo::IsCallNode(node)) {
      return false;
    }
    return common::AnfAlgo::IsDynamicShape(node);
  });
}

// Disable mindRT in the heterogeneous scenario + dynamic_shape scenario.
void DisableMindRT(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<bool>(MS_CTX_ENABLE_MINDRT) == false) {
    return;
  }
#ifdef WITH_BACKEND
  if (ps::PSContext::instance()->cache_enable()) {
    return;
  }
#endif

  auto func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto parallel_mode = parallel_context->parallel_mode();
  bool is_parallel_mode = parallel_mode == parallel::kSemiAutoParallel || parallel_mode == parallel::kAutoParallel;
  bool use_old_vm_for_control_parallel =
    func_graph->exist_multi_target() && ExistControlFlow(func_graph) && is_parallel_mode;
  if (use_old_vm_for_control_parallel) {
    MS_LOG(INFO) << "Disable mindRT in the heterogeneous + control flow + parallel scenario.";
    context_ptr->set_param<bool>(MS_CTX_ENABLE_MINDRT, false);
    // Async update the backend.
    resource->SetBackendAsync([]() {
      auto new_backend = compile::CreateBackend();
      new_backend->SetDebugger();
      return new_backend;
    });
  }
}

void TaskEmitActionForMindRT(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  // Get the mindRT backend.
  auto bc_ptr = resource->GetBackend();
  auto mindrt_bc_ptr = std::dynamic_pointer_cast<compile::MindRTBackend>(bc_ptr);
  MS_EXCEPTION_IF_NULL(mindrt_bc_ptr);

  // The output of graph compiler is actor.
  auto actor_info = mindrt_bc_ptr->CompileGraphs(resource->func_graph());
  resource->SetResult(kOutput, actor_info);
}

void ExecuteActionForMindRT(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  const auto actor_info = resource->GetResult(kOutput).cast<compile::ActorInfo>();
  // Get the mindRT backend.
  auto bc_ptr = resource->GetBackend();
  auto mindrt_bc_ptr = (std::dynamic_pointer_cast<compile::MindRTBackend>(bc_ptr)).get();
  MS_EXCEPTION_IF_NULL(mindrt_bc_ptr);

  // Construct the graph run function ptr.
  compile::VmEvalFuncPtr run =
    std::make_shared<compile::VmEvalFunc>([mindrt_bc_ptr, actor_info](const VectorRef &args) -> BaseRef {
      MS_LOG(DEBUG) << "Execute args size " << args.size();
      VectorRef outputs;
      mindrt_bc_ptr->RunGraph(actor_info, args, &outputs);
      MS_LOG(DEBUG) << "out size " << outputs.size();
      if (outputs.empty()) {
        return VectorRef();
      } else {
        return outputs[0];
      }
    });
  resource->SetResult(kOutput, run);
}

// Modify the output node of func_graph to add forward nodes used in bprop graph.
void ModifyOutputNode(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &used_forward_nodes = func_graph->used_forward_nodes();

  // Get original output node and abstract
  auto original_output_node = func_graph->output();
  MS_EXCEPTION_IF_NULL(original_output_node);
  auto original_output_abs = original_output_node->abstract();
  MS_EXCEPTION_IF_NULL(original_output_abs);

  // Create a new make tuple node to hold all forward used nodes.
  abstract::AbstractBasePtrList added_abs_list;
  std::vector<AnfNodePtr> added_node_list{NewValueNode(prim::kPrimMakeTuple)};
  std::for_each(used_forward_nodes.begin(), used_forward_nodes.end(),
                [&added_abs_list, &added_node_list](const AnfNodePtr &node) {
                  MS_EXCEPTION_IF_NULL(node);
                  added_node_list.push_back(node);
                  added_abs_list.push_back(node->abstract());
                });
  AnfNodePtr added_output_node = nullptr;
  AbstractBasePtr added_output_abs = nullptr;
  if (added_abs_list.empty()) {
    added_output_node = NewValueNode(MakeValue<int32_t>(1));
    added_output_abs = std::make_shared<abstract::AbstractScalar>(std::make_shared<Int32Imm>(1));
  } else {
    added_output_node = func_graph->NewCNode(std::move(added_node_list));
    added_output_abs = std::make_shared<abstract::AbstractTuple>(added_abs_list);
  }
  added_output_node->set_abstract(added_output_abs);
  MS_LOG(DEBUG) << "Added output node info: " << added_output_node->DebugString();

  // Merge original output node and used forward nodes to return node.
  std::vector<AnfNodePtr> new_output_nodes{NewValueNode(prim::kPrimMakeTuple), original_output_node, added_output_node};
  auto merge_node = func_graph->NewCNode(std::move(new_output_nodes));
  abstract::AbstractBasePtrList new_output_abs{original_output_abs, added_output_abs};
  merge_node->set_abstract(std::make_shared<abstract::AbstractTuple>(new_output_abs));
  MS_LOG(DEBUG) << "Merge node info: " << merge_node->DebugString();
  func_graph->set_output(merge_node);

  // Clear
  func_graph->set_modify_output(true);
  func_graph->ClearUsedForwardNodes();
}
}  // namespace
using CompileGraphs = compile::CompileGraphs;
using abstract::AnalysisResult;
using mindspore::abstract::AnalysisContextPtr;

// Whether this process in a MindSpore cluster.
static bool is_cluster_initialized = false;

abstract::AnalysisResult AbstractAnalyze(const ResourcePtr &resource, const FuncGraphPtr &func_graph,
                                         const abstract::AbstractBasePtrList &args_abs, bool clear) {
  MS_LOG(DEBUG) << "AbstractAnalyze start";
  auto engine = resource->engine();
  MS_EXCEPTION_IF_NULL(engine);
  if (clear || resource->is_load()) {
    auto manager = resource->manager();
    MS_EXCEPTION_IF_NULL(manager);
    engine->Clear();
    static const auto enable_eliminate_unused_element = (common::GetEnv("MS_DEV_ENABLE_DDE") != "0");
    for (auto &node : manager->all_nodes()) {
      MS_EXCEPTION_IF_NULL(node);

      // Handle previous inferred value for CNode if is loaded from MindIR
      if (resource->is_load()) {
        // If the primitive is not defined in front end, keep the inferred value loaded from MindIR.
        auto primitive = GetCNodePrimitive(node);
        if (primitive != nullptr) {
          auto is_load = primitive->GetAttr("is_load");
          if (abstract::GetPrimEvaluator(primitive, engine) == nullptr && is_load != nullptr &&
              GetValue<bool>(is_load)) {
            MS_LOG(WARNING) << "The primitive is not defined in front end. Primitive: " << primitive->ToString();
            continue;
          }
        }
        if (!clear && node->isa<Parameter>()) {
          continue;
        }
      }

      const AbstractBasePtr &prev_inferred = node->abstract();
      // Keep previous inferred value for ValueNode if the inferred value is not AbstractFunction.
      if (!node->isa<ValueNode>() || (prev_inferred != nullptr && prev_inferred->isa<abstract::AbstractFunction>())) {
        // Reset tuple/list abstract use flags.
        if (enable_eliminate_unused_element && prev_inferred != nullptr &&
            prev_inferred->isa<abstract::AbstractSequence>()) {
          SetSequenceNodeElementsUseFlags(node, nullptr);
        }
        node->set_abstract(nullptr);
        MS_LOG(DEBUG) << "Abstract of node " << node->DebugString() << " is set to nullptr";
      }
    }
  }
  auto res = engine->Run(func_graph, args_abs);
  MS_LOG(INFO) << "function call max depth: " << abstract::FunctionCallMaxDepth()
               << ", simulate call max depth: " << abstract::StackFrameMaxDepth();
  MS_LOG(DEBUG) << "AbstractAnalyze end";
  return res;
}

FuncGraphPtr ProgramSpecialize(const ResourcePtr &resource, const FuncGraphPtr &func_graph,
                               const abstract::AnalysisContextPtr &context) {
  MS_EXCEPTION_IF_NULL(resource);
  MS_LOG(DEBUG) << "ProgramSpecialize start";
  abstract::ProgramSpecializer specializer(resource->engine());
  FuncGraphPtr result = specializer.Run(func_graph, context);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->KeepRoots({result});
  specializer.SpecializeCNodeInput0FuncGraph();
  MS_LOG(DEBUG) << "ProgramSpecialize end";
  return result;
}

FuncGraphPtr Renormalize(const ResourcePtr &resource, const FuncGraphPtr &func_graph,
                         const abstract::AbstractBasePtrList &args_abs) {
  MS_EXCEPTION_IF_NULL(resource);
  MS_LOG(DEBUG) << "Renormalize start";
#ifdef ENABLE_PROFILE
  double t1 = GetTime();
#endif
  abstract::AnalysisResult result = AbstractAnalyze(resource, func_graph, args_abs, true);
#ifdef ENABLE_PROFILE
  double t2 = GetTime();
#endif
  auto res = ProgramSpecialize(resource, func_graph, result.context);
  resource->set_func_graph(res);
#ifdef ENABLE_PROFILE
  double t3 = GetTime();
  MsProfile::StatTime("renormalize.infer", t2 - t1);
  MsProfile::StatTime("renormalize.specialize", t3 - t2);
#endif

  MS_LOG(DEBUG) << "Renormalize end";

  return res;
}

void SetLoadFlag(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  FuncGraphPtr loaded_graph = nullptr;
  size_t loaded_graph_num = 0;
  auto all_graphs = manager->func_graphs();
  for (auto &graph : all_graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    if (graph->has_attr("is_load")) {
      loaded_graph = graph;
      loaded_graph_num += 1;
      resource->set_is_load(true);
      return;
    }
  }
}

bool ParseAction(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  TraceManager::OpenRecordDebugInfoFlag();
  if (!resource->source_input()) {
    MS_LOG(EXCEPTION) << "Parse error";
  }

  py::object input = resource->source_input();
  parse::Parser::InitParserEnvironment(input);
  py::module path = py::module::import("os.path");
  std::string dir = path.attr("dirname")(py::globals()["__file__"]).cast<std::string>();

  python_adapter::set_python_env_flag(true);
  python_adapter::SetPythonPath(dir);

  ValuePtr converted_ret = nullptr;
  bool converted = parse::ConvertData(input, &converted_ret, true);
  if (!converted) {
    MS_LOG(EXCEPTION) << "Attribute convert error with type:" << std::string(py::str(input));
  }

  FuncGraphPtr top_graph = nullptr;
  if (py::hasattr(input, parse::PYTHON_PARSE_METHOD)) {
    top_graph = parse::MakeTopGraph(input, converted_ret);
  } else if (converted_ret->isa<FuncGraph>()) {
    top_graph = converted_ret->cast<FuncGraphPtr>();
  } else {
    MS_LOG(EXCEPTION) << "Object to parse " << std::string(py::str(input)) << " is not function or cell.";
  }
  parse::Parser::UpdateTopFuncGraph(top_graph);

  resource->set_func_graph(top_graph);

  FuncGraphManagerPtr manager = resource->manager();
  if (manager == nullptr) {
    MS_LOG(EXCEPTION) << "Manager is nullptr.";
  }
  manager->AddFuncGraph(top_graph);
  return true;
}

// obj_map's graphs have the same construct, these graphs can be optimized to one graph.
// This step do this optimize: graph1(x){xx(fv1),xxx(fv2)}, graph2(x){xxx(fv3),xxx(fv4)}->
// graph1(x){base_graph(x, fv1, fv2)}, graph1(x){base_graph(x, fv3, fv4)}, base_graph(x, fv...){xxx,xxx}
// all obj_map's graph shared base_graph
bool CombineLikeGraphs(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  auto &obj_map = parse::data_converter::GetObjGraphs();
  for (auto it = obj_map.rbegin(); it != obj_map.rend(); ++it) {
    auto &graphs = it->second;
    MS_LOG(DEBUG) << "Start combine like graph:" << it->first << ", size:" << graphs.size();
    auto fg = graphs[0];
    FuncGraphVector func_graphs = {fg};
    Cloner cloner(func_graphs, false, false, true, std::make_shared<TraceCopy>(),
                  std::make_shared<TraceCombileLikeGraphs>());
    cloner.Run();
    auto cloned_fg_iter = cloner.cloned_func_graphs().find(fg);
    if (cloned_fg_iter == cloner.cloned_func_graphs().end()) {
      MS_LOG(EXCEPTION) << "Clone func graph failed! " << fg->ToString();
    }
    auto base_graph = cloned_fg_iter->second;
    MS_LOG(DEBUG) << "Basegraph:" << base_graph->ToString();

    if (fg->paramter_obj_nodes().empty() || graphs.size() <= 1 || fg->has_flag(FUNC_GRAPH_OUTPUT_NO_RECOMPUTE)) {
      continue;
    }
    auto &cloned_nodes = cloner.cloned_nodes();
    for (auto &fv : fg->paramter_obj_nodes()) {
      TraceGuard guard(std::make_shared<TraceCombileLikeGraphs>(fv->debug_info()));
      auto param = base_graph->add_parameter();
      MS_EXCEPTION_IF_NULL(resource->manager());
      auto &node_users = resource->manager()->node_users()[fv];
      for (auto &n : node_users) {
        // If the user is not in this graph, no need to change.
        auto iter = cloned_nodes.find(n.first);
        if (iter == cloned_nodes.end()) {
          continue;
        }
        auto repl_n = iter->second->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(repl_n);
        repl_n->set_input(IntToSize(n.second), param);
      }
    }
    MS_LOG(DEBUG) << "Fg0 paramter_obj_nodes size :" << fg->paramter_obj_nodes().size();

    for (auto &g : graphs) {
      auto &fvs = g->paramter_obj_nodes();
      std::vector<AnfNodePtr> new_node_inputs;
      new_node_inputs.push_back(NewValueNode(base_graph));
      for (auto &p : g->parameters()) {
        AnfNodePtr para_after_cast = parse::GetMixedPrecisionCastHelp(g, p);
        new_node_inputs.push_back(para_after_cast);
      }
      (void)new_node_inputs.insert(new_node_inputs.end(), fvs.cbegin(), fvs.cend());
      AnfNodePtr out = g->NewCNodeBefore(g->get_return(), new_node_inputs);
      g->set_output(out);
      const int recursive_level = 4;
      MS_LOG(DEBUG) << "Combine graph newout:" << out->DebugString(recursive_level);
    }
    MS_LOG(DEBUG) << "End combine graph:" << it->first;
  }
  return true;
}

namespace {
bool IsSideEffectCNode(const AnfNodePtr &node) {
  const auto &primitive = GetCNodePrimitiveWithoutDoSignature(node);
  if (primitive != nullptr) {
    auto effect_info = GetPrimEffectInfo(primitive);
    if (effect_info.memory || effect_info.io) {
      MS_LOG(DEBUG) << "Side Effect Primitive CNode: " << node->DebugString();
      return true;
    }
  }
  return false;
}

bool HasIsolatedSideEffectNode(const FuncGraphPtr &func_graph) {
  const auto node = func_graph->output();
  if (!IsPrimitiveCNode(node, prim::kPrimDepend)) {
    return false;
  }
  auto cnode = dyn_cast<CNode>(node);
  MS_EXCEPTION_IF_NULL(cnode);
  auto attr_sort_rhs_first = cnode->GetAttr(kAttrTopoSortRhsFirst);
  auto sort_rhs_first =
    attr_sort_rhs_first != nullptr && attr_sort_rhs_first->isa<BoolImm>() && GetValue<bool>(attr_sort_rhs_first);
  if (!sort_rhs_first) {
    // Return false if it's definitely not isolated Depend CNode.
    return false;
  }

  // To check isolated nodes in {Depend -> StopGradient -> MakeTuple(...)}.
  constexpr size_t stop_gradient_pos = 2;
  auto stop_gradient_node = cnode->input(stop_gradient_pos);
  auto stop_gradient_cnode = dyn_cast<CNode>(stop_gradient_node);
  MS_EXCEPTION_IF_NULL(stop_gradient_cnode);
  constexpr size_t isolated_node_pos = 1;
  auto isolated_node = stop_gradient_cnode->input(isolated_node_pos);
  if (IsPrimitiveCNode(isolated_node, prim::kPrimMakeTuple)) {
    auto isolated_cnode = dyn_cast<CNode>(isolated_node);
    MS_EXCEPTION_IF_NULL(isolated_cnode);
    for (size_t i = 1; i < isolated_cnode->size(); ++i) {
      if (IsSideEffectCNode(isolated_cnode->input(i))) {
        MS_LOG(DEBUG) << "Multiple isolated side-effect node[" << i << "]: " << isolated_cnode->input(i)->DebugString();
        return true;
      }
    }
  } else {
    if (IsSideEffectCNode(isolated_node)) {
      MS_LOG(DEBUG) << "Single isolated side-effect node: " << isolated_node->DebugString();
      return true;
    }
  }
  return false;
}

void CheckIsolatedSideEffectNode(const FuncGraphPtr &func_graph) {
  if (!HasIsolatedSideEffectNode(func_graph)) {
    return;
  }

  auto new_return = func_graph->get_return();
  new_return->set_has_isolated_side_effect_node(true);
  func_graph->set_has_isolated_side_effect_node(true);
  auto output_cnode = dyn_cast<CNode>(func_graph->output());
  if (output_cnode != nullptr) {
    output_cnode->set_has_isolated_side_effect_node(true);
  }
  MS_LOG(INFO) << "Set isolated side-effect node flag for " << func_graph->ToString();
}

// Get all the trainable parameters of the reusable cell.
void GetTrainableParameters(const FuncGraphPtr &fg, std::vector<AnfNodePtr> *parameters) {
  MS_EXCEPTION_IF_NULL(parameters);
  if (fg->manager() == nullptr) {
    MS_LOG(INFO) << fg->ToString() << " manager is null. This Cell init should not be assigned cell_attr_register.";
    return;
  }
  auto used_fgs = fg->func_graphs_used_total();
  std::set<const AnfNode *> memo;
  for (auto &g : used_fgs) {
    for (auto &item : g->paramter_obj_nodes()) {
      MS_LOG(DEBUG) << fg->ToString() << " has_default: " << item->cast<ParameterPtr>()->has_default()
                    << " parameter: " << item->cast<ParameterPtr>()->ToString();
      if (item->cast<ParameterPtr>()->has_default() && memo.emplace(item.get()).second) {
        parameters->push_back(item);
      }
    }
  }
  MS_LOG(DEBUG) << fg->ToString() << ", parameters: " << parameters->size();
}

FuncGraphPtr GenerateReusingGraph(const FuncGraphPtr &fg) {
  std::vector<AnfNodePtr> parameters;
  MS_LOG(DEBUG) << fg->ToString();
  GetTrainableParameters(fg, &parameters);
  if (parameters.empty()) {
    MS_LOG(DEBUG) << "Finish handling the reusable graph: " << fg->ToString()
                  << ", parameter size: " << parameters.size();
    return nullptr;
  }
  FuncGraphVector func_graphs = {fg};
  Cloner cloner(func_graphs, false, false, true, std::make_shared<TraceCopy>(), std::make_shared<TraceGraphReusing>());
  cloner.Run();
  auto cloned_fg_iter = cloner.cloned_func_graphs().find(fg);
  if (cloned_fg_iter == cloner.cloned_func_graphs().end()) {
    MS_LOG(EXCEPTION) << "Clone func graph failed! " << fg->ToString();
  }
  auto reusing_graph = cloned_fg_iter->second;

  // Make the reusable graph to be the no_inline status.
  reusing_graph->set_flag(FUNC_GRAPH_FLAG_NO_INLINE, true);

  // Make the all trainable parameters of the reusable cell to be the
  // parameters of the reusable graph.
  auto &cloned_nodes = cloner.cloned_nodes();
  auto manager = fg->manager();
  for (auto &fv : parameters) {
    TraceGuard guard(std::make_shared<TraceGraphReusing>(fv->debug_info()));
    auto param = reusing_graph->add_parameter();
    auto &node_users = manager->node_users()[fv];
    for (auto &n : node_users) {
      auto iter = cloned_nodes.find(n.first);
      if (iter == cloned_nodes.end()) {
        continue;
      }
      auto repl_n = iter->second->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(repl_n);
      repl_n->set_input(IntToSize(n.second), param);
    }
  }
  MS_LOG(DEBUG) << "The reusable graph parameter size: " << reusing_graph->parameters().size();
  return reusing_graph;
}

void ReplaceWithReusingGraph(const FuncGraphPtr &reusing_graph, const FuncGraphPtr &origin_graph) {
  std::vector<AnfNodePtr> fvs;
  MS_LOG(DEBUG) << origin_graph->ToString();
  GetTrainableParameters(origin_graph, &fvs);
  std::vector<AnfNodePtr> new_node_inputs;
  new_node_inputs.push_back(NewValueNode(reusing_graph));
  for (auto &p : origin_graph->parameters()) {
    AnfNodePtr para_after_cast = parse::GetMixedPrecisionCastHelp(origin_graph, p);
    new_node_inputs.push_back(para_after_cast);
  }
  (void)new_node_inputs.insert(new_node_inputs.cend(), fvs.cbegin(), fvs.cend());
  AnfNodePtr out = origin_graph->NewCNodeBefore(origin_graph->get_return(), new_node_inputs);
  origin_graph->set_output(out);
  MS_LOG(DEBUG) << "The original graph's new out: " << out->DebugString();
  origin_graph->erase_flag(FUNC_GRAPH_FLAG_NO_INLINE);
}
}  // namespace

// Make the reusable cell to be the reusable function graph.
bool GraphReusingAction(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  constexpr size_t graph_reusing_count = 2;
  const auto &obj_map = parse::data_converter::GetObjGraphs();
  for (const auto &[cell_key, graphs] : obj_map) {
    MS_LOG(DEBUG) << "Start to handle the reusable graph: " << cell_key << ", size: " << graphs.size();
    // Only make the reusable cell that is used more than graph_reusing_count to be reusable.
    if (graphs.size() < graph_reusing_count) {
      continue;
    }
    const auto &fg = graphs[0];
    // fg->paramter_obj_nodes().empty() have been handled by combine like.
    if (!fg->paramter_obj_nodes().empty()) {
      MS_LOG(DEBUG) << "Finish handling the reusable graph: " << cell_key;
      continue;
    }
    auto reusing_graph = GenerateReusingGraph(fg);
    if (reusing_graph == nullptr) {
      MS_LOG(DEBUG) << "Finish handling the reusable graph: " << cell_key;
      continue;
    }
    // Let the original cell graph call the reusable graph.
    (void)std::for_each(graphs.begin(), graphs.end(), [&reusing_graph](const auto &origin_graph) {
      ReplaceWithReusingGraph(reusing_graph, origin_graph);
    });

    MS_LOG(DEBUG) << "Finish handling the reusable graph: " << cell_key;
  }
  return true;
}

bool SymbolResolveAction(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (resource->manager() == nullptr) {
    MS_LOG(EXCEPTION) << "SymbolResolve error, manager is null";
  }
  auto func_graph = resource->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "SymbolResolve error, graph is null";
  }
  bool ret = parse::ResolveFuncGraph(func_graph, resource);
  // Remove unused nodes in cnode order list,
  // and check isolated side-effect nodes.
  if (func_graph != nullptr) {
    func_graph->EraseUnusedNodeInOrder();
    CheckIsolatedSideEffectNode(func_graph);
    for (auto fg : func_graph->func_graphs_used_total()) {
      if (fg != nullptr) {
        fg->EraseUnusedNodeInOrder();
        CheckIsolatedSideEffectNode(fg);
      }
    }
  }
  return ret;
}

bool AutoMonadAction(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (resource->manager() == nullptr) {
    MS_LOG(EXCEPTION) << "Auto-Monad failed, manager is null";
  }
  auto func_graph = resource->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "Auto-Monad failed, graph is null";
  }
  (void)pipeline::AutoMonad(func_graph);
  return true;
}

bool OrderEnforceAction(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (resource->manager() == nullptr) {
    MS_LOG(EXCEPTION) << "Order-Enforce error, manager is null";
  }
  auto func_graph = resource->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "Order-Enforce error, graph is null";
  }
  pipeline::OrderEnforce(func_graph);
  return true;
}

bool MetaUnpackPrepareAction(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (resource->manager() == nullptr) {
    MS_LOG(EXCEPTION) << "MetaUnpackPrepareAction error, manager is null.";
  }
  if (resource->func_graph() == nullptr) {
    MS_LOG(EXCEPTION) << "MetaUnpackPrepareAction error, graph is null.";
  }
  return MetaUnpackPreparePass(resource);
}

namespace {
abstract::AbstractBasePtrList GetArgsAbs(const ResourcePtr &resource) {
  FuncGraphPtr func_graph = resource->func_graph();
  abstract::AbstractBasePtrList args_abs = resource->args_abs();

  // Parallel checking.
  auto context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel::ParallelContext::GetInstance());
  context->ParallelParameterContextInitShape(func_graph);

  // Handle the Parameter from FV inputs.
  for (const auto &param : func_graph->parameters()) {
    auto param_node = std::static_pointer_cast<Parameter>(param);
    MS_EXCEPTION_IF_NULL(param_node);
    if (param_node->has_default()) {
      auto value = param_node->default_param();
      MS_EXCEPTION_IF_NULL(value);
      auto abs_value = value->ToAbstract()->cast<abstract::AbstractTensorPtr>();
      auto ref_key = std::make_shared<RefKey>(param_node->name());
      auto abs_ref = std::make_shared<abstract::AbstractRefTensor>(abs_value, ref_key);
      context->ParallelParameterContextRestoreShape(func_graph, param_node, abs_ref);
      (void)args_abs.emplace_back(abs_ref);
      context->ParallelParameterContextCkptShape(func_graph, param_node, abs_ref);
    }
  }
  return args_abs;
}
}  // namespace

bool AbstractSpecializeAction(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (resource->func_graph() == nullptr) {
    MS_LOG(EXCEPTION) << "AbstractSpecialize error";
  }

  SetLoadFlag(resource);

  // Abstract analyze
  auto engine = resource->engine();
  MS_EXCEPTION_IF_NULL(engine);
  engine->set_check_isolated_side_effect(true);
  AnalysisResult result = AbstractAnalyze(resource, resource->func_graph(), GetArgsAbs(resource));

  // The top graph may be replaced by infer, update the top graph when the infer is done
  parse::Parser::UpdateTopFuncGraph(result.context->func_graph());

  // Specialize
  FuncGraphPtr new_fg = ProgramSpecialize(resource, result.context->func_graph(), result.context);
  resource->set_func_graph(new_fg);
  engine->set_check_isolated_side_effect(false);

  // Remove unused nodes in cnode order list, this is prepared for auto-monad.
  if (new_fg) {
    new_fg->EraseUnusedNodeInOrder();
    for (auto fg : new_fg->func_graphs_used_total()) {
      if (fg) {
        fg->EraseUnusedNodeInOrder();
      }
    }
  }

  UpdateFuncGraphParameter(new_fg, resource->arguments());
  MS_LOG(DEBUG) << "End graph: " << new_fg->ToString() << ", return: " << new_fg->get_return()->DebugString(true);
  return true;
}

bool OptimizeAction(const ResourcePtr &resource, const std::vector<PassItem> &passes) {
  MS_EXCEPTION_IF_NULL(resource);
  size_t counter = 0;
  for (auto &pass : passes) {
    WITH(MsProfile::GetProfile()->Step(pass.first))[&pass, &resource, &counter]() {
      MS_LOG(DEBUG) << "Pass " << pass.first << " start ...";
      auto result = pass.second(resource);
      if (!result) {
        MS_LOG(EXCEPTION) << "Pass running to end, failed in pass:" << pass.first;
      }
#ifdef ENABLE_DUMP_IR
      if (MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG) && resource->func_graph() != nullptr) {
        auto fg_name = "opt_pass_" + std::to_string(counter) + "_" + pass.first;
        auto func_graph = resource->func_graph();
        MS_EXCEPTION_IF_NULL(func_graph);
        DumpIR(fg_name + ".ir", func_graph);
        ExportIR(fg_name + ".dat", func_graph);
        draw::Draw(fg_name + ".dot", func_graph);
        MS_LOG(DEBUG) << "Dump " << fg_name << " func graph.";
      }
#endif
      counter++;
      MS_LOG(DEBUG) << "Pass " << pass.first << " end.";
    };
  }

  return true;
}

bool OptInlineAction(const ResourcePtr &resource) {
  if (parallel::ParallelContext::GetInstance()->parallel_mode() == "semi_auto_parallel" ||
      parallel::ParallelContext::GetInstance()->parallel_mode() == "auto_parallel") {
    return OptimizeAction(resource, kInlinePasses);
  }
  if (opt::python_pass::PyPassManager::GetInstance()->GetPassGroup(opt::python_pass::Phase::PREAD)->size() != 0) {
    return OptimizeAction(resource, kInlinePasses);
  }
  return true;
}

bool GeOptimizeAction(const ResourcePtr &resource) { return OptimizeAction(resource, kGePasses); }

bool VmOptimizeAction(const ResourcePtr &resource) {
#ifdef WITH_BACKEND
  if (ps::PSContext::instance()->is_ps_mode()) {
    (void)kVmPasses.emplace_back(PassItem("server_communication_op_fusion", ps::Util::FuseServerCommOps));
  }
#endif
  auto ret = OptimizeAction(resource, kVmPasses);
  TraceManager::ClearParseOrResolveDebugInfo();
  TraceManager::CloseRecordDebugInfoFlag();
  return ret;
}

static bool IsCtrlSink() {
  auto ms_ctx = MsContext::GetInstance();
  if (ms_ctx->get_param<int>(MS_CTX_EXECUTION_MODE) != kGraphMode) {
    return false;
  }

  std::string device_target = ms_ctx->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_target != kAscendDevice) {
    return false;
  }

  if (!ms_ctx->get_param<bool>(MS_CTX_ENABLE_TASK_SINK)) {
    return false;
  }

  return ms_ctx->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK);
}

bool CheckGraphOutputConstOrParameter(const FuncGraphPtr &func_graph) {
  if (func_graph != nullptr) {
    AnfNodePtr output = func_graph->output();
    if (output != nullptr && (output->isa<ValueNode>() || output->isa<Parameter>())) {
      return true;
    }
  }
  return false;
}

bool EliminateForwardCNode(const ResourcePtr &resource) {
  // This function only works in Pynative mode. The func_graph is decorated by ms_function.
  if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode) {
    return true;
  }

  auto graph_executor = pipeline::GraphExecutorPy::GetInstance();
  MS_EXCEPTION_IF_NULL(graph_executor);
  auto phase = graph_executor->phase();
  MS_LOG(DEBUG) << "The phase of current pipeline graph is: " << phase;
  // Exporting graph in PyNative mode or only running forward process no need to do this action.
  auto pynative_exec = pynative::PyNativeExecutor::GetInstance();
  if (phase.find("export") == 0 || !pynative_exec->grad_flag()) {
    MS_LOG(DEBUG) << "When exporting graph or only running forward process, no need to eliminate forward cnode.";
    auto grad_exec = pynative_exec->grad_executor();
    grad_exec->set_eliminate_forward(true);
    return true;
  }

  // Run grad process for func_graph and replace forward nodes with its output tensors.
  MS_LOG(INFO) << "Run eliminate forward nodes action.";
  MS_EXCEPTION_IF_NULL(resource);
  auto ms_func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(ms_func_graph);
  auto grad_exec = pynative_exec->grad_executor();
  bool eliminate_forward = grad_exec->eliminate_forward();
  grad_exec->set_eliminate_forward(eliminate_forward && ms_func_graph->func_graphs_used().empty());
  auto grad_graph = ad::Grad(ms_func_graph, opt::Optimizer::MakeEmptyOptimizer(resource));
  MS_EXCEPTION_IF_NULL(grad_graph);
  graph_executor->SetGradGraph(grad_graph, phase);
  ModifyOutputNode(ms_func_graph);

  // Keep roots for only keeping forward func graph in resource.
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->KeepRoots({ms_func_graph});

  grad_exec->set_eliminate_forward(true);
  return true;
}

bool EliminateAdRelatedSpecialOpNode(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (resource->manager() == nullptr) {
    MS_LOG(EXCEPTION) << "PynativeElimOpt error, manager is null.";
  }
  if (resource->func_graph() == nullptr) {
    MS_LOG(EXCEPTION) << "PynativeElimOpt error, graph is null.";
  }
  return EliminateAdRelatedSpecialOpOptPass(resource);
}

bool HasAbstractFunction(const AbstractBasePtr &abs) {
  if (abs->isa<abstract::AbstractSequence>() && !abs->isa<abstract::AbstractSparseTensor>()) {
    auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    return std::any_of(abs_seq->elements().cbegin(), abs_seq->elements().cend(), HasAbstractFunction);
  }
  // if abs it not AbstractSequence.
  return abs->isa<abstract::AbstractFunction>();
}

bool HasIncorporateCall(const std::vector<AnfNodePtr> &all_nodes) {
  for (const auto &node : all_nodes) {
    if (IsValueNode<FuncGraph>(node)) {
      auto func_graph = GetValueNode<FuncGraphPtr>(node);
      if (HasAbstractFunction(func_graph->output()->abstract())) {
        return true;
      }
    }
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (IsPrimitiveCNode(cnode, prim::kPrimPartial)) {
      auto partial_function = cnode->input(kPartialGraphIndex);
      if (!IsValueNode<FuncGraph>(partial_function)) {
        MS_LOG(INFO) << "Partial has indirect call: " << cnode->DebugString();
        return true;
      }
      continue;
    }
    if (IsPrimitiveCNode(cnode, prim::kPrimSwitch)) {
      const auto &switch_inputs = cnode->inputs();
      if (std::any_of(switch_inputs.begin() + kSwitchTrueBranchIndex, switch_inputs.end(), [](const AnfNodePtr &input) {
            return !IsPrimitiveCNode(input, prim::kPrimPartial) && !IsValueNode<FuncGraph>(input);
          })) {
        MS_LOG(INFO) << "Switch has indirect call: " << cnode->DebugString();
        return true;
      }
      continue;
    }
    if (IsPrimitiveCNode(cnode, prim::kPrimSwitchLayer)) {
      auto make_tuple = cnode->input(kSwitchLayerBranchesIndex);
      if (!IsPrimitiveCNode(make_tuple, prim::kPrimMakeTuple)) {
        MS_LOG(EXCEPTION) << "SwitchLayer input2 should be make_tuple, but got: " << make_tuple->DebugString();
      }
      const auto &make_tuple_inputs = make_tuple->cast<CNodePtr>()->inputs();
      if (std::any_of(make_tuple_inputs.begin() + 1, make_tuple_inputs.end(), [](const AnfNodePtr &input) {
            return !IsPrimitiveCNode(input, prim::kPrimPartial) && !IsValueNode<FuncGraph>(input);
          })) {
        MS_LOG(INFO) << "SwitchLayer has indirect call: " << cnode->DebugString();
        return true;
      }
      continue;
    }
    if (!IsValueNode<Primitive>(cnode->input(0))) {  // If cnode is a call node.
      auto input0 = cnode->input(0);
      if (IsPrimitiveCNode(input0, prim::kPrimSwitch) || IsPrimitiveCNode(input0, prim::kPrimSwitchLayer) ||
          IsValueNode<FuncGraph>(input0)) {
        continue;
      }
      MS_LOG(INFO) << "Call has indirect call: " << cnode->DebugString();
      return true;
    }
  }
  return false;
}

bool ExistTarget(const std::vector<AnfNodePtr> &all_nodes, const std::string &target) {
  for (const auto &node : all_nodes) {
    if (node == nullptr || !node->isa<CNode>()) {
      continue;
    }
    if (GetCNodeTarget(node) == target) {
      return true;
    }
  }
  return false;
}

// If the return value of subgraph is Ref in control flow scenarios, should run graph mode with kernelbykernel.
bool ExistSwitchRef(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &all_nodes) {
  // %1 = switch(cond, func1, func2)
  // %2 = %1()  if the abstract of the node is AbstractRefTensor or Tuple/List(AbstractRefTensor, ...), return true.
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &node_users = manager->node_users();
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::string device_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimSwitch)) {
      continue;
    }
    auto iter = node_users.find(node);
    if (iter != node_users.end()) {
      auto &users = iter->second;
      for (auto &user : users) {
        auto &user_node = user.first;
        if (common::AnfAlgo::HasAbstractRef(user_node) || common::AnfAlgo::SequenceHasAbstractRef(user_node)) {
          if (device_target == kAscendDevice) {
            MS_LOG(WARNING) << "On the Ascend platform, when the return value of the control flow subgraph is "
                            << "parameter, the performance may be degraded. The value of the parameter can be returned "
                            << "to improve the performance. "
                            << "For example, change 'return param' to 'return param.value()'\n"
                            << "Please check your code:" << trace::GetDebugInfo(user_node->debug_info());
          }
          return true;
        }
      }
    }
  }
  return false;
}

void SetRunMode(const FuncGraphPtr &func_graph, compile::Backend *backend_ptr) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(backend_ptr);
  auto set_ctx = [&context_ptr, &backend_ptr](bool task_sink, bool is_multi_graph_sink, bool enable_loop_sink) {
    context_ptr->set_param<bool>(MS_CTX_ENABLE_TASK_SINK, task_sink);
    context_ptr->set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, is_multi_graph_sink);
    context_ptr->set_param<bool>(MS_CTX_ENABLE_LOOP_SINK, enable_loop_sink);
    backend_ptr->set_is_multi_graph_sink(is_multi_graph_sink);
  };

  auto jit_level = pipeline::GetJitLevel();
  func_graph->set_attr(kAttrJitLevel, MakeValue<std::string>(jit_level));

  const auto &all_nodes = TopoSort(func_graph->return_node(), SuccDeeperSimple, AlwaysInclude);
  // GPU/CPU no need set any context.
  if (!ExistTarget(all_nodes, kAscendDevice)) {
    return;
  }

  // GRAPH | Single Op : KernelByKernel path in MindRT.
  if (common::GetEnv(kGraphOpRun) == "1") {
    MS_LOG(INFO) << "Run graph mode with kernelbykernel.";
    set_ctx(false, false, false);
    return;
  }

  // GRAPH | Dynamic Shape : KernelByKernel path in MindRT.
  if (IsDynamicShapeGraph(func_graph)) {
    MS_LOG(INFO) << "Run graph mode with kernelbykernel(Dynamic Shape).";
    set_ctx(false, false, false);
    return;
  }

  // GRAPH | Closure\ENV\While scenario : KernelByKernel path in MindRT.
  auto graphs = func_graph->func_graphs_used_total();
  (void)graphs.insert(func_graph);
  bool exist_control_flow = ExistControlFlow(func_graph);
  bool exist_func = exist_control_flow && HasIncorporateCall(all_nodes);
  MS_LOG(INFO) << func_graph->ToString() << " exist_func: " << exist_func;
  if (exist_func) {
    MS_LOG(INFO) << "Run graph mode with kernelbykernel.";
    set_ctx(false, false, false);
    return;
  }
  bool exist_while =
    std::any_of(graphs.cbegin(), graphs.cend(), [](const FuncGraphPtr &fg) { return fg->recursive(); });
  MS_LOG(INFO) << func_graph->ToString() << " exist_while: " << exist_while;
  if (exist_while || ExistSwitchRef(func_graph, all_nodes)) {
    MS_LOG(INFO) << "Run graph mode with kernelbykernel.";
    set_ctx(false, false, false);
    return;
  }

  // Multiple device targets scenario.
  if (func_graph->exist_multi_target()) {
    // Heterogeneous scenario + ControlFlow : KernelByKernel path in MindRT.
    if (exist_control_flow) {
      MS_LOG(INFO) << "Run graph mode with kernelbykernel.";
      set_ctx(false, false, false);
      return;
    }
    // GRAPH | Heterogeneous scenario : No control flow, subgraph sink path in MindRT.
    MS_LOG(INFO) << "Run graph mode with subgraph sink.";
    set_ctx(true, false, false);
    return;
  }

#ifdef WITH_BACKEND
  if (ps::PSContext::instance()->cache_enable()) {
    set_ctx(true, false, false);
    return;
  }
#endif

  // GRAPH | normal network and if/for/switch scenario etc : MultiGraph path in MindRT.
  MS_LOG(INFO) << "Run graph mode with multigraph sink.";
  set_ctx(true, true, true);
  return;
}

void OriginSetRunMode(const ResourcePtr &resource) {
  FuncGraphPtr func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto bc_ptr = resource->GetBackend();
  auto context_ptr = MsContext::GetInstance();
  std::string backend = MsContext::GetInstance()->backend_policy();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto task_sink = context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
  if (func_graph->exist_multi_target() || !task_sink) {
    bc_ptr->set_is_multi_graph_sink(false);
    context_ptr->set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, false);
    context_ptr->set_param<bool>(MS_CTX_ENABLE_LOOP_SINK, false);
  } else if (context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode) {
    std::string device_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    auto manager = func_graph->manager();
    auto graphs = manager->func_graphs();
    if (graphs.size() > 1 && device_target == kAscendDevice) {
      MS_LOG(INFO) << "This func_graph has control flow nodes, owns " << graphs.size() << " subgraphs.";
    }
    bool exist_while =
      std::any_of(graphs.cbegin(), graphs.cend(), [](const FuncGraphPtr &fg) { return fg->recursive(); });
    if (device_target == kAscendDevice && backend != kMsVm && !exist_while) {
      MS_LOG(INFO) << "Run graph mode with multigraph sink.";
      bc_ptr->set_is_multi_graph_sink(true);
      context_ptr->set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, true);
    } else {
      MS_LOG(INFO) << "Run graph mode with vm.";
      bc_ptr->set_is_multi_graph_sink(false);
      context_ptr->set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, false);
      context_ptr->set_param<bool>(MS_CTX_ENABLE_LOOP_SINK, false);
    }
  }
}

void SetRunMode(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<bool>(MS_CTX_ENABLE_MINDRT) && common::GetEnv("DISABLE_ASCEND_MINDRT") != "1") {
    SetRunMode(resource->func_graph(), resource->GetBackend().get());
  } else {
    OriginSetRunMode(resource);
  }
  auto mode = context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE);
  auto is_task_sink = context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
  auto enable_hccl = context_ptr->get_param<bool>(MS_CTX_ENABLE_HCCL);
  // After the distributed interface on D is unified, the following flag and judgment will be removed.
  bool enbale_distributed_mindrt = false;
#ifdef WITH_BACKEND
  enbale_distributed_mindrt = ps::PSContext::instance()->enable_distributed_mindrt();
#endif
  if (!is_task_sink && mode == kGraphMode && enable_hccl && !common::UseMPI() && !enbale_distributed_mindrt) {
    MS_LOG(EXCEPTION) << "Current execute mode must launch process with OpenMPI";
  }
}

bool TaskEmitAction(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr func_graph = resource->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "TaskEmit args error";
  }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto mode = context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE);
  if (mode == kGraphMode && CheckGraphOutputConstOrParameter(func_graph)) {
    return true;
  }

  func_graph->SetMultiTarget();
  if (DumpJsonParser::GetInstance().IsDumpEnabled() && func_graph->exist_multi_target()) {
    MS_LOG(WARNING) << "Multi device target is detected, CPU data is dumped in rank_0 directory";
  }
  DisableMindRT(resource);

  SetRunMode(resource);
  auto bc_ptr = resource->GetBackend();
  MS_EXCEPTION_IF_NULL(bc_ptr);
  std::string backend = context_ptr->backend_policy();
  // The graph compiling of mindRT.
  if ((backend == kMsConvert || backend == kGeVm) && context_ptr->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    TaskEmitActionForMindRT(resource);
    return true;
  }
  // The graph compiling of control sink.
  if (IsCtrlSink() && backend == kMsConvert) {
    auto graph_id = bc_ptr->CompileGraph(NOT_NULL(func_graph));
    resource->SetResult(kOutput, graph_id);
    return true;
  }
  std::vector<PrimitivePtr> cut_list = compile::GetNonlinearOps();
  if (bc_ptr->name() == kMsConvert) {
    cut_list = compile::GetMsNonlinearOps();
  }
  std::shared_ptr<CompileGraphs> compile = std::make_shared<CompileGraphs>(bc_ptr, cut_list);
  auto vm = compile->CompileAndLink(func_graph);
  resource->SetResult(kOutput, vm);
  return true;
}

bool ExecuteAction(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode &&
      CheckGraphOutputConstOrParameter(resource->func_graph())) {
    return true;
  }
  if (!resource->HasResult(kOutput)) {
    MS_LOG(EXCEPTION) << "Execute args error";
  }
  std::string backend = MsContext::GetInstance()->backend_policy();
  // The graph running of mindRT.
  if ((backend == kMsConvert || backend == kGeVm) && MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    ExecuteActionForMindRT(resource);
    return true;
  }

  // The graph running of control sink.
  if (IsCtrlSink() && backend == kMsConvert) {
    auto graph_id = resource->GetResult(kOutput).cast<GraphId>();
    auto bc_ptr = resource->GetBackend();
    compile::MsBackend *msbc_ptr = std::dynamic_pointer_cast<compile::MsBackend>(bc_ptr).get();
    MS_EXCEPTION_IF_NULL(msbc_ptr);
    compile::VmEvalFuncPtr run =
      std::make_shared<compile::VmEvalFunc>([msbc_ptr, graph_id](const VectorRef &args) -> BaseRef {
        MS_LOG(INFO) << "Execute args size " << args.size();
        auto outs = msbc_ptr->RunGraph(graph_id, args);
        MS_LOG(DEBUG) << "out size " << outs.size();
        return outs[0];
      });
    resource->SetResult(kOutput, run);
    return true;
  }

  compile::FinalVMPtr vm = resource->GetResult(kOutput).cast<compile::FinalVMPtr>();
  if (vm == nullptr) {
    MS_LOG(INFO) << "Call GE to Run the func_graph instead of VM";
    return true;
  }
  compile::VmEvalFuncPtr run =
    std::make_shared<compile::VmEvalFunc>(std::bind(&compile::FinalVM::Eval, vm, std::placeholders::_1));
  resource->SetResult(kOutput, run);
  return true;
}

#ifdef WITH_BACKEND
bool StartPSSchedulerAction(const ResourcePtr &) {
  if (distributed::cluster::ClusterContext::instance()->initialized()) {
    MS_LOG(INFO) << "This node is scheduler. Start wait for finalizing.";
    if (!distributed::cluster::ClusterContext::instance()->Finalize(UINT32_MAX)) {
      MS_LOG(ERROR) << "Failed to finalize server.";
      return false;
    }
    MS_LOG(INFO) << "Scheduler is successfully finalized.";
    return true;
  }
  ps::Scheduler::GetInstance().Run();
  return true;
}

bool DistributedSplitAction(const ResourcePtr &resource) {
  // Only run this action when the cluster is initialized.
  if (!distributed::cluster::ClusterContext::instance()->initialized()) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr func_graph = resource->func_graph();
  auto node = distributed::cluster::ClusterContext::instance()->node();
  MS_EXCEPTION_IF_NULL(node);
  auto node_role = distributed::cluster::ClusterContext::instance()->node_role();

  parallel::GraphSplitterPtr splitter =
    std::make_shared<parallel::GraphSplitter>(func_graph, node->rank_id(), node_role);
  MS_EXCEPTION_IF_NULL(splitter);
  splitter->Run();

  // Renomalize: Infer shape and Set abstract for all nodes in graph.
  if (func_graph->has_flag(kFlagNeedRenormalize)) {
    abstract::AbstractBasePtrList args_abs;
    auto parameters = func_graph->parameters();
    (void)std::transform(parameters.begin(), parameters.end(), std::back_inserter(args_abs),
                         [](const AnfNodePtr &p) -> AbstractBasePtr { return p->abstract(); });
    FuncGraphPtr new_fg = Renormalize(resource, func_graph, args_abs);
    resource->set_func_graph(new_fg);
    resource->set_args_abs(args_abs);
  }
  return true;
}
#endif

// The parallel primitive related valuenode might be partitioned so that its value changes by device,
// that will result in a synchronization error due to different executing order.
// Here we temporarily avoid the problem by skipping valuenode merging used by parallel related primitive,
// the final solution will be proposed later as a parallel feature.
bool KeepValueNodeDuplication(const AnfNodePtr &value_node, const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  MS_EXCEPTION_IF_NULL(resource->manager());
  auto &node_users = resource->manager()->node_users();
  auto &users = node_users[value_node];
  auto used_by_keep_value_prim =
    std::any_of(users.begin(), users.end(), [](const std::pair<AnfNodePtr, int64_t> &user) -> bool {
      MS_EXCEPTION_IF_NULL(user.first);
      auto cnode = user.first->cast<CNodePtr>();
      if (cnode == nullptr) {
        return false;
      }
      auto prim_node = cnode->input(0);
      if (IsValueNode<Primitive>(prim_node)) {
        auto prim = GetValue<PrimitivePtr>(prim_node->cast<ValueNodePtr>()->value());
        MS_EXCEPTION_IF_NULL(prim);
        // value_node is referenced by some parallel primitive
        return prim->HasAttr("keep_value_node_input");
      }
      return false;
    });
  return used_by_keep_value_prim;
}

bool RemoveValueNodeDuplicationsAction(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr func_graph = resource->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "Remove value node duplications error.";
  }
  auto manager = resource->manager();
  // Remove duplicated value nodes, due to replace operation, can't use reference.
  auto value_nodes = func_graph->value_nodes();
  HashCache hash_cache;
  HashValue hashes;
  for (const auto &value_pair : value_nodes) {
    if (KeepValueNodeDuplication(value_pair.first, resource)) {
      continue;
    }
    TryToDoReplace(manager.get(), value_pair.first, &hash_cache, &hashes);
  }
  return true;
}

bool PipelineSplitAction(const ResourcePtr &resource) { return PipelineSplitPass(resource); }

bool ValidateAction(const ResourcePtr &resource) { return ValidatePass(resource); }

bool GeSpecializedAction(const ResourcePtr &resource) { return GeSpecializedPass(resource); }

bool SetMindIRGraphAction(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  resource->set_is_load(true);
  auto cell = py::cast<CellPtr>(resource->source_input());
  if (cell == nullptr) {
    MS_LOG(EXCEPTION) << "The graph loaded from mindir is null.";
  }
  const std::string mindir_graph = "graph_load_from_mindir";
  auto obj = cell->GetAttr(mindir_graph);
  if (obj == nullptr) {
    MS_LOG(EXCEPTION) << "The graph loaded from mindir is null. The cell has not attribute: " << mindir_graph;
  }
  auto fg = GetValue<FuncGraphPtr>(obj);
  if (fg == nullptr) {
    MS_LOG(EXCEPTION) << "The graph loaded from mindir is null.";
  }
  resource->set_func_graph(fg);
  FuncGraphManagerPtr mng = fg->manager();
  if (mng == nullptr) {
    auto res_mng = resource->manager();
    MS_EXCEPTION_IF_NULL(res_mng);
    res_mng->Clear();
    res_mng->AddFuncGraph(fg);
  }
  abstract::AbstractBasePtrList broaded_args;
  const auto &args_spec_list = resource->args_abs();
  (void)std::transform(args_spec_list.begin(), args_spec_list.end(), std::back_inserter(broaded_args),
                       [](const AbstractBasePtr &arg) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(arg);
                         if (arg->GetValueTrack() != kAnyValue) {
                           return arg->Broaden();
                         }
                         return arg;
                       });

  abstract::AbstractBasePtrList func_args;
  const auto inputs = fg->get_inputs();
  (void)std::transform(inputs.begin(), inputs.end(), std::back_inserter(func_args),
                       [](const AnfNodePtr &arg) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(arg);
                         return arg->abstract()->Broaden();
                       });

  bool is_equal_input_args = true;
  if (!AbstractBasePtrListDeepEqual(func_args, broaded_args)) {
    MS_LOG(INFO) << "The input arguments is not compatible with the function graph which has been exported before."
                 << "Please check the args is same with export.\n"
                 << "The export input argument size: " << func_args.size() << "\n"
                 << "The load input argument size: " << broaded_args.size() << "\n"
                 << "Export input args info: " << abstract::ArgsToString(func_args) << "\n"
                 << "The input args info: " << abstract::ArgsToString(broaded_args);
    is_equal_input_args = false;
  }

  if (!is_equal_input_args) {
    // Use InferMindir which will find c++ infer in eval_map and backend_eval_map;
    (void)InferMindir(resource->func_graph(), args_spec_list, true);
  }
  return true;
}

bool ActionPyStub(const ResourcePtr &resource, opt::python_pass::Phase phase) {
  MS_EXCEPTION_IF_NULL(resource->manager());
  MS_EXCEPTION_IF_NULL(resource->func_graph());
  auto ppm = opt::python_pass::PyPassManager::GetInstance();
  ppm->SetResource(resource);
  return ppm->GetPassGroup(phase)->Run(resource->func_graph());
}

bool PreAdActionPyStub(const ResourcePtr &resource) {
  if (!ActionPyStub(resource, opt::python_pass::Phase::PREAD)) {
    MS_LOG(DEBUG) << "No Match.";
  }
  return true;
}

bool OptActionVmPyStub(const ResourcePtr &resource) {
  if (ActionPyStub(resource, opt::python_pass::Phase::OPT)) {
    if (opt::python_pass::PyPassManager::GetInstance()->ShouldRenorm()) {
      // Renomalize
      FuncGraphPtr func_graph = resource->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      abstract::AbstractBasePtrList args_abs;
      auto parameters = func_graph->parameters();
      (void)std::transform(parameters.begin(), parameters.end(), std::back_inserter(args_abs),
                           [](const AnfNodePtr &p) -> AbstractBasePtr { return p->abstract(); });
      FuncGraphPtr new_fg = Renormalize(resource, func_graph, args_abs);
      resource->set_func_graph(new_fg);
      resource->set_args_abs(args_abs);
    }
    if (opt::python_pass::PyPassManager::GetInstance()->ShouldReOpt()) {
      return VmOptimizeAction(resource);
    }
  }
  return true;
}

bool OptActionGePyStub(const ResourcePtr &resource) {
  if (ActionPyStub(resource, opt::python_pass::Phase::OPT)) {
    if (opt::python_pass::PyPassManager::GetInstance()->ShouldRenorm()) {
      // Renomalize
      FuncGraphPtr func_graph = resource->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      abstract::AbstractBasePtrList args_abs;
      auto parameters = func_graph->parameters();
      (void)std::transform(parameters.begin(), parameters.end(), std::back_inserter(args_abs),
                           [](const AnfNodePtr &p) -> AbstractBasePtr { return p->abstract(); });
      FuncGraphPtr new_fg = Renormalize(resource, func_graph, args_abs);
      resource->set_func_graph(new_fg);
      resource->set_args_abs(args_abs);
    }
    if (opt::python_pass::PyPassManager::GetInstance()->ShouldReOpt()) {
      return GeOptimizeAction(resource);
    }
  }
  return true;
}

static std::vector<ActionItem> CommonPipeline() {
  std::vector<ActionItem> actions;

  // Parse the python ast to ANF graph
  (void)actions.emplace_back(std::make_pair("parse", ParseAction));

  // Resolve the python func
  (void)actions.emplace_back(std::make_pair("symbol_resolve", SymbolResolveAction));

  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto parallel_mode = parallel_context->parallel_mode();
  const bool is_parallel_mode =
    parallel_mode == parallel::kSemiAutoParallel || parallel_mode == parallel::kAutoParallel;
  if (!is_cluster_initialized && !is_parallel_mode && pipeline::GetJitLevel() != "O0") {
    (void)actions.emplace_back(std::make_pair("combine_like_graphs", CombineLikeGraphs));
  }

  // Make the reusable cell to be the reusable function graph
  static bool enable_graph_reusing = (common::GetEnv("MS_DEV_GRAPH_REUSE") == "1");
  if (enable_graph_reusing) {
    (void)actions.emplace_back(std::make_pair("graph_reusing", GraphReusingAction));
  }

  (void)actions.emplace_back(std::make_pair("meta_unpack_prepare", MetaUnpackPrepareAction));
  // Evaluate type and shape, and specialize.
  (void)actions.emplace_back(std::make_pair("abstract_specialize", AbstractSpecializeAction));
  // Auto-monad for side-effects handling.
  (void)actions.emplace_back(std::make_pair("auto_monad", AutoMonadAction));
  // Do data structure simplifications and inline.
  (void)actions.emplace_back(std::make_pair("inline", OptInlineAction));
  // Add pre-ad, post-inline python pass stub.
  (void)actions.emplace_back(std::make_pair("py_pre_ad", PreAdActionPyStub));
  // Do PipelineSplit action.
  (void)actions.emplace_back(std::make_pair("pipeline_split", PipelineSplitAction));

  return actions;
}

std::vector<ActionItem> GePipeline() {
  auto actions = CommonPipeline();
  // Optimize
  (void)actions.emplace_back(std::make_pair("optimize", GeOptimizeAction));
  // Add opt-stage python pass stub
  (void)actions.emplace_back(std::make_pair("py_opt", OptActionGePyStub));
  (void)actions.emplace_back(std::make_pair("remove_value_node_duplications", RemoveValueNodeDuplicationsAction));
  (void)actions.emplace_back(std::make_pair("auto_monad_reorder", OrderEnforceAction));
  (void)actions.emplace_back(std::make_pair("ge_specialized_prepare", GeSpecializedAction));
  (void)actions.emplace_back(std::make_pair("eliminate_ad_related_special_op_node", EliminateAdRelatedSpecialOpNode));
  (void)actions.emplace_back(std::make_pair("validate", ValidateAction));
#ifdef WITH_BACKEND
  // Compile the ANF graph
  (void)actions.emplace_back(std::make_pair("task_emit", TaskEmitAction));

  // Execute the graph
  (void)actions.emplace_back(std::make_pair("execute", ExecuteAction));
#endif
  return actions;
}

std::vector<ActionItem> VmPipeline(const ResourcePtr &resource) {
  is_cluster_initialized = distributed::cluster::ClusterContext::instance()->initialized();
  std::vector<ActionItem> actions;
  // If enable compilation cache and the cache is read successfully, only do the backend actions.
  if (!resource->EnableCompileCache() || resource->func_graph() == nullptr) {
    actions = CommonPipeline();

    // Optimize
    (void)actions.emplace_back(std::make_pair("optimize", VmOptimizeAction));

    // Add opt-stage python pass stub
    (void)actions.emplace_back(std::make_pair("py_opt", OptActionVmPyStub));

    (void)actions.emplace_back(std::make_pair("auto_monad_reorder", OrderEnforceAction));

    // Eliminate forward cnode for grad graph
    (void)actions.emplace_back(std::make_pair("eliminate_forward_cnode", EliminateForwardCNode));

    // Eliminate the virtual mirror node
    (void)actions.emplace_back(std::make_pair("eliminate_ad_related_special_op_node", EliminateAdRelatedSpecialOpNode));
    (void)actions.emplace_back(std::make_pair("validate", ValidateAction));
  }

#ifdef WITH_BACKEND
  (void)actions.emplace_back(std::make_pair("distribtued_split", DistributedSplitAction));
  if (ps::PSContext::instance()->is_worker()) {
    if (distributed::cluster::ClusterContext::instance()->initialized()) {
      MS_LOG(INFO) << "This worker is initialized. No need to add worker action.";
    } else {
      std::string server_mode = ps::PSContext::instance()->server_mode();
    }
  }
#endif
  // Compile the ANF graph
  (void)actions.emplace_back(std::make_pair("task_emit", TaskEmitAction));

  // Execute the graph
  (void)actions.emplace_back(std::make_pair("execute", ExecuteAction));

  return actions;
}

std::vector<ActionItem> MindIRPipeline() {
  auto context_ptr = MsContext::GetInstance();
  if (context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    MS_LOG(EXCEPTION)
      << "The graph generated form MindIR is not support to execute in the PynativeMode, please convert "
         "to the GraphMode.";
  }
  std::vector<ActionItem> actions;
  // Set funcGraph loaded from MindIR to resource.
  (void)actions.emplace_back(std::make_pair("load_mindir", SetMindIRGraphAction));
  (void)actions.emplace_back(std::make_pair("validate", ValidateAction));
  // Compile the ANF graph
  (void)actions.emplace_back(std::make_pair("task_emit", TaskEmitAction));
  // Execute the graph
  (void)actions.emplace_back(std::make_pair("execute", ExecuteAction));
  return actions;
}

#ifdef WITH_BACKEND
std::vector<ActionItem> PSchedulerPipeline(const ResourcePtr &resource) {
  if (resource->EnableCompileCache() && resource->func_graph() != nullptr) {
    return {std::make_pair("scheduler", StartPSSchedulerAction)};
  }
  auto actions = CommonPipeline();
  (void)actions.emplace_back(std::make_pair("optimize", VmOptimizeAction));
  (void)actions.emplace_back(std::make_pair("auto_monad_reorder", OrderEnforceAction));
  (void)actions.emplace_back(std::make_pair("eliminate_forward_cnode", EliminateForwardCNode));
  (void)actions.emplace_back(std::make_pair("eliminate_ad_related_special_op_node", EliminateAdRelatedSpecialOpNode));
  (void)actions.emplace_back(std::make_pair("validate", ValidateAction));
  (void)actions.emplace_back(std::make_pair("scheduler", StartPSSchedulerAction));
  return actions;
}
#endif
}  // namespace pipeline
}  // namespace mindspore
