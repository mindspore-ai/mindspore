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
#include <string>
#include <algorithm>
#include <functional>

#include "ir/func_graph_cloner.h"
#include "ir/param_info.h"
#include "ir/cell.h"
#include "parse/python_adapter.h"
#include "abstract/abstract_value.h"
#include "frontend/parallel/costmodel_context.h"
#include "frontend/parallel/context.h"
#include "frontend/parallel/graph_util/graph_splitter.h"
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
#include "frontend/optimizer/irpass/parameter_eliminate.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "backend/graph_compiler/transform.h"
#include "load_mindir/infer_mindir.h"
#if ((defined ENABLE_CPU) && (!defined _WIN32))
#include "ps/parameter_server.h"
#include "ps/scheduler.h"
#include "ps/worker.h"
#include "fl/worker/fl_worker.h"
#include "fl/server/server.h"
#include "distributed/cluster/cluster_context.h"
#endif

namespace mindspore {
namespace pipeline {
namespace {
bool ExistControlFlow(const FuncGraphPtr &func_graph) { return !func_graph->func_graphs_used_total().empty(); }

void UpdateFuncGraphParameter(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> new_paras;
  for (const auto &param : func_graph->parameters()) {
    auto param_node = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_node);
    if (param_node->has_default()) {
      new_paras.push_back(param_node);
      continue;
    }
    AbstractBasePtr par_abs = param_node->abstract();
    MS_EXCEPTION_IF_NULL(par_abs);
    if (par_abs->isa<abstract::AbstractUndetermined>() ||
        (MsContext::GetInstance()->get_param<bool>(MS_CTX_GRAD_FOR_SCALAR) && par_abs->BuildType() != nullptr &&
         par_abs->BuildType()->isa<Number>()) ||
        (par_abs->isa<abstract::AbstractTuple>() &&
         par_abs->cast<abstract::AbstractTuplePtr>()->ContainsAllBroadenTensors())) {
      new_paras.push_back(param_node);
    }
  }
  func_graph->set_parameters(new_paras);
}

bool IsDynamicShapeGraph(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return());
  return std::any_of(node_list.begin(), node_list.end(),
                     [](const AnfNodePtr &node) { return AnfAlgo::IsDynamicShape(node); });
}

// Disable mindRT in the heterogeneous scenario + dynamic_shape scenario.
void DisableMindRT(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<bool>(MS_CTX_ENABLE_MINDRT) == false) {
    return;
  }
  auto func_graph = res->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto parallel_mode = parallel_context->parallel_mode();
  bool is_parallel_mode = parallel_mode == parallel::SEMI_AUTO_PARALLEL || parallel_mode == parallel::AUTO_PARALLEL;
  bool enable_old_runtime = (common::GetEnv("MS_DEV_ENABLE_CLOSURE") == "0");
  bool use_old_vm_for_dynamic_shape = func_graph->exist_multi_target() && IsDynamicShapeGraph(func_graph);
  bool use_old_vm_for_control_parallel =
    func_graph->exist_multi_target() && ExistControlFlow(func_graph) && is_parallel_mode;
  if (enable_old_runtime || use_old_vm_for_dynamic_shape || use_old_vm_for_control_parallel) {
    // Heterogeneous scenario + dynamic_shape runs in MsBackend.
    MS_LOG(INFO) << "Disable mindRT in the heterogeneous + dynamic shape scenario.";
    context_ptr->set_param<bool>(MS_CTX_ENABLE_MINDRT, false);
    // Update the backend.
    auto new_backend = compile::CreateBackend();
    new_backend->SetDebugger();
    res->SetResult(kBackend, new_backend);
  }
}

void TaskEmitActionForMindRT(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  // Get the mindRT backend.
  auto bc_ptr = res->GetResult(kBackend).cast<compile::BackendPtr>();
  auto mindrt_bc_ptr = std::dynamic_pointer_cast<compile::MindRTBackend>(bc_ptr);
  MS_EXCEPTION_IF_NULL(mindrt_bc_ptr);

  // The output of graph compiler is actor.
  auto actor_info = mindrt_bc_ptr->CompileGraphs(res->func_graph());
  res->SetResult(kOutput, actor_info);
}

void ExecuteActionForMindRT(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  const auto actor_info = res->GetResult(kOutput).cast<compile::ActorInfo>();
  // Get the mindRT backend.
  std::shared_ptr<compile::Backend> bc_ptr = res->GetResult(kBackend).cast<std::shared_ptr<compile::Backend>>();
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
  res->SetResult(kOutput, run);
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

abstract::AnalysisResult AbstractAnalyze(const ResourcePtr &resource, const FuncGraphPtr &func_graph,
                                         const abstract::AbstractBasePtrList &args_spec, bool clear) {
  MS_LOG(DEBUG) << "AbstractAnalyze start";
  auto engine = resource->engine();
  MS_EXCEPTION_IF_NULL(engine);
  if (clear) {
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
        if (primitive != nullptr && abstract::GetPrimEvaluator(primitive, engine) == nullptr) {
          MS_LOG(INFO) << "The primitive is not defined in front end. Primitive: " << primitive->ToString();
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
  auto res = engine->Run(func_graph, args_spec);
  MS_LOG(INFO) << "function call max depth: " << abstract::FunctionCallMaxDepth()
               << ", simulate call max depth: " << abstract::StackFrameMaxDepth();
  MS_LOG(DEBUG) << "AbstractAnalyze end";
  return res;
}

FuncGraphPtr ProgramSpecialize(const ResourcePtr &res, const FuncGraphPtr &func_graph,
                               const abstract::AnalysisContextPtr &context) {
  MS_EXCEPTION_IF_NULL(res);
  MS_LOG(DEBUG) << "ProgramSpecialize start";
  abstract::ProgramSpecializer specializer(res->engine());
  FuncGraphPtr result = specializer.Run(func_graph, context);
  auto manager = res->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->KeepRoots({result});
  specializer.SpecializeCNodeInput0FuncGraph();
  MS_LOG(DEBUG) << "ProgramSpecialize end";
  return result;
}

FuncGraphPtr Renormalize(const ResourcePtr &res, const FuncGraphPtr &func_graph,
                         const abstract::AbstractBasePtrList &args_spec) {
  MS_EXCEPTION_IF_NULL(res);
  MS_LOG(DEBUG) << "Renormalize start";
#ifdef ENABLE_PROFILE
  double t1 = GetTime();
#endif
  abstract::AnalysisResult result = AbstractAnalyze(res, func_graph, args_spec, true);
#ifdef ENABLE_PROFILE
  double t2 = GetTime();
#endif
  auto ret = ProgramSpecialize(res, func_graph, result.context);
  res->set_func_graph(ret);
#ifdef ENABLE_PROFILE
  double t3 = GetTime();
  MsProfile::StatTime("renormalize.infer", t2 - t1);
  MsProfile::StatTime("renormalize.specialize", t3 - t2);
#endif

  MS_LOG(DEBUG) << "Renormalize end";

  return ret;
}

const FuncGraphPtr GetLoadedGraph(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  auto manager = res->manager();
  MS_EXCEPTION_IF_NULL(manager);
  FuncGraphPtr loaded_graph = nullptr;
  size_t loaded_graph_num = 0;
  auto all_graphs = manager->func_graphs();
  for (auto &graph : all_graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    if (graph->has_attr("is_load")) {
      loaded_graph = graph;
      loaded_graph_num += 1;
      res->set_is_load(true);
    }
  }
  if (loaded_graph_num == 0) {
    return nullptr;
  }
  if (loaded_graph_num == 1) {
    return loaded_graph;
  }
  MS_LOG(EXCEPTION) << "The loaded sub graph currently should be less than 2, but got " << loaded_graph_num;
}

void CheckRootInputShapeAndType(const ResourcePtr &res, const FuncGraphPtr &loaded_graph) {
  MS_EXCEPTION_IF_NULL(res);
  auto manager = res->manager();
  MS_EXCEPTION_IF_NULL(manager);
  FuncGraphPtr root_graph = *(manager->roots().begin());
  auto root_inputs = root_graph->get_inputs();
  auto loaded_inputs = loaded_graph->get_inputs();
  MS_LOG(DEBUG) << "root_graph: " << root_graph->ToString();
  MS_LOG(DEBUG) << "loaded_graph: " << loaded_graph->ToString();
  size_t root_inputs_num = root_inputs.size();
  size_t loaded_inputs_num = loaded_inputs.size();
  if (root_inputs_num != loaded_inputs_num) {
    MS_LOG(EXCEPTION) << "The inputs number " << root_inputs_num << " not equal to the inputs number of loaded graph "
                      << loaded_inputs_num;
  }

  for (size_t index = 0; index < root_inputs_num; index++) {
    auto root_input = root_inputs[index];
    auto loaded_input = loaded_inputs[index];

    MS_LOG(DEBUG) << "root_input[" << index << "]: " << root_input->DebugString(1);
    MS_LOG(DEBUG) << "loaded_input[" << index << "]: " << loaded_input->DebugString(1);
    MS_LOG(DEBUG) << "root_input abstract[" << index
                  << "]: " << (root_input->abstract() ? root_input->abstract()->ToString() : "NULL");
    MS_LOG(DEBUG) << "loaded_input abstract [" << index
                  << "]: " << (loaded_input->abstract() ? loaded_input->abstract()->ToString() : "NULL");

    auto root_shape = root_input->Shape() == nullptr ? nullptr : dyn_cast<abstract::Shape>(root_input->Shape());
    auto loaded_shape = loaded_input->Shape() == nullptr ? nullptr : dyn_cast<abstract::Shape>(loaded_input->Shape());
    auto root_type = root_input->Type() == nullptr ? nullptr : dyn_cast<Type>(root_input->Type());
    auto loaded_type = loaded_input->Type() == nullptr ? nullptr : dyn_cast<Type>(loaded_input->Type());

    MS_EXCEPTION_IF_NULL(root_shape);
    MS_EXCEPTION_IF_NULL(loaded_shape);
    MS_EXCEPTION_IF_NULL(root_type);
    MS_EXCEPTION_IF_NULL(loaded_type);

    auto shapeEqu = (root_shape->shape() == loaded_shape->shape()) ||
                    (root_shape->shape().size() <= 1 && loaded_shape->shape().size() <= 1);
    if (!shapeEqu) {
      MS_EXCEPTION(ValueError) << "The " << index
                               << " th input shape differ from loaded graph. Input shape: " << root_shape->ToString()
                               << ", input shape of loaded graph: " << loaded_shape->ToString();
    }
    if (root_type->type_id() != loaded_type->type_id()) {
      MS_EXCEPTION(TypeError) << "The " << std::to_string(index)
                              << " th input type differ from loaded graph. Input type: " << root_type->ToString()
                              << ", input type of loaded graph: " << loaded_type->ToString();
    }
  }
}

bool ParseAction(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  TraceManager::OpenRecordDebugInfoFlag();
  if (!res->source_input()) {
    MS_LOG(EXCEPTION) << "Parse error";
  }

  py::object input = res->source_input();
  parse::Parser::InitParserEnvironment(input);
  py::module path = py::module::import("os.path");
  std::string dir = path.attr("dirname")(py::globals()["__file__"]).cast<std::string>();

  parse::python_adapter::set_python_env_flag(true);
  parse::python_adapter::SetPythonPath(dir);

  ValuePtr converted_ret = nullptr;
  bool converted = parse::ConvertData(input, &converted_ret, true);
  if (!converted) {
    MS_LOG(EXCEPTION) << "Attribute convert error with type:" << std::string(py::str(input));
  }

  FuncGraphPtr top_graph = nullptr;
  if (py::isinstance<Cell>(input)) {
    top_graph = parse::MakeTopGraph(input, converted_ret);
  } else if (converted_ret->isa<FuncGraph>()) {
    top_graph = converted_ret->cast<FuncGraphPtr>();
  } else {
    MS_LOG(EXCEPTION) << "Object to parse " << std::string(py::str(input)) << " is not function or cell.";
  }
  parse::Parser::UpdateTopFuncGraph(top_graph);

  res->set_func_graph(top_graph);

  FuncGraphManagerPtr manager = res->manager();
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
bool CombineLikeGraphs(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  auto &obj_map = parse::data_converter::GetObjGraphs();
  for (auto it : obj_map) {
    auto &graphs = it.second;
    MS_LOG(DEBUG) << "Start combine like graph:" << it.first << ", size:" << graphs.size();
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
      MS_EXCEPTION_IF_NULL(res->manager());
      auto &node_users = res->manager()->node_users()[fv];
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
      (void)new_node_inputs.insert(new_node_inputs.end(), fvs.begin(), fvs.end());
      AnfNodePtr out = g->NewCNodeBefore(g->get_return(), new_node_inputs);
      g->set_output(out);
      const int recursive_level = 4;
      MS_LOG(DEBUG) << "Combine graph newout:" << out->DebugString(recursive_level);
    }
    MS_LOG(DEBUG) << "End combine graph:" << it.first;
  }
  return true;
}

bool SymbolResolveAction(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  if (res->manager() == nullptr) {
    MS_LOG(EXCEPTION) << "SymbolResolve error, manager is null";
  }
  auto func_graph = res->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "SymbolResolve error, graph is null";
  }
  bool ret = parse::ResolveFuncGraph(func_graph, res);
  // Remove unused nodes in cnode order list.
  if (func_graph) {
    func_graph->EraseUnusedNodeInOrder();
    for (auto fg : func_graph->func_graphs_used_total()) {
      if (fg) {
        fg->EraseUnusedNodeInOrder();
      }
    }
  }
  return ret;
}

bool AutoMonadAction(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  if (res->manager() == nullptr) {
    MS_LOG(EXCEPTION) << "Auto-Monad failed, manager is null";
  }
  auto func_graph = res->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "Auto-Monad failed, graph is null";
  }
  (void)pipeline::AutoMonad(func_graph);
  return true;
}

bool OrderEnforceAction(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  if (res->manager() == nullptr) {
    MS_LOG(EXCEPTION) << "Order-Enforce error, manager is null";
  }
  auto func_graph = res->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "Order-Enforce error, graph is null";
  }
  pipeline::OrderEnforce(func_graph);
  return true;
}

bool InferenceOptPrepareAction(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  if (res->manager() == nullptr) {
    MS_LOG(EXCEPTION) << "InferenceOptPrepare error, manager is null.";
  }
  if (res->func_graph() == nullptr) {
    MS_LOG(EXCEPTION) << "InferenceOptPrepare error, graph is null.";
  }
  return InferenceOptPreparePass(res);
}

bool EliminateUnusedParameterAction(const ResourcePtr &res) {
  static const auto transform_tail_call_to_parallel_call = (common::GetEnv("MS_DEV_PARALLEL_CALL") == "1");
  if (!transform_tail_call_to_parallel_call) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(res);
  FuncGraphPtr func_graph = res->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  bool changed = opt::irpass::ParameterEliminator()(func_graph, nullptr);
  MS_LOG(DEBUG) << "Eliminate parameter, changed: " << changed;
  return true;
}

bool AbstractSpecializeAction(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  if (res->func_graph() == nullptr) {
    MS_LOG(EXCEPTION) << "AbstractSpecialize error";
  }
  FuncGraphPtr func_graph = res->func_graph();
  abstract::AbstractBasePtrList args_spec = res->args_spec();
  auto context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel::ParallelContext::GetInstance());
  context->ParallelParameterContextInitShape(func_graph);

  // Get original loaded graph to check inputs later
  auto loaded_graph_ptr = GetLoadedGraph(res);
  // suppose that there is not KeywordArgument for the top graph
  // get the hyper parameter
  for (const auto &param : func_graph->parameters()) {
    auto param_node = std::static_pointer_cast<Parameter>(param);
    MS_EXCEPTION_IF_NULL(param_node);
    if (param_node->has_default()) {
      auto value = param_node->default_param();
      MS_EXCEPTION_IF_NULL(value);
      auto abs_value = value->ToAbstract()->cast<abstract::AbstractTensorPtr>();
      auto ref_key = std::make_shared<RefKey>(param_node->name());
      auto abs_ref_key = ref_key->ToAbstract();
      auto abs_ref = std::make_shared<abstract::AbstractRef>(abs_ref_key, abs_value);
      context->ParallelParameterContextRestoreShape(func_graph, param_node, abs_ref);
      args_spec.push_back(abs_ref);
      context->ParallelParameterContextCkptShape(func_graph, param_node, abs_ref);
    }
  }
  // Analyze
  AnalysisResult result = AbstractAnalyze(res, func_graph, args_spec);

  // The top graph may be replaced by infer, update the top graph when the infer is done
  parse::Parser::UpdateTopFuncGraph(result.context->func_graph());

  // Specialize
  FuncGraphPtr new_fg = ProgramSpecialize(res, result.context->func_graph(), result.context);
  res->set_func_graph(new_fg);

  // Remove unused nodes in cnode order list, this is prepared for auto-monad.
  if (new_fg) {
    new_fg->EraseUnusedNodeInOrder();
    for (auto fg : new_fg->func_graphs_used_total()) {
      if (fg) {
        fg->EraseUnusedNodeInOrder();
      }
    }
  }
  // Check input after abstract when there is a loaded graph
  if (loaded_graph_ptr != nullptr) {
    CheckRootInputShapeAndType(res, loaded_graph_ptr);
  }

  UpdateFuncGraphParameter(new_fg);
  MS_LOG(DEBUG) << "End graph: " << new_fg->ToString() << ", return: " << new_fg->get_return()->DebugString(true);
  return true;
}

bool OptimizeAction(const ResourcePtr &res, const std::vector<PassItem> &passes) {
  MS_EXCEPTION_IF_NULL(res);
  size_t counter = 0;
  for (auto &pass : passes) {
    WITH(MsProfile::GetProfile()->Step(pass.first))[&pass, &res, &counter]() {
      MS_LOG(DEBUG) << "Pass " << pass.first << " start ...";
      auto result = pass.second(res);
      if (!result) {
        MS_LOG(EXCEPTION) << "Pass running to end, failed in pass:" << pass.first;
      }
#ifdef ENABLE_DUMP_IR
      if (MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG) && res->func_graph() != nullptr) {
        auto fg_name = "opt_pass_" + std::to_string(counter) + "_" + pass.first;
        auto func_graph = res->func_graph();
        MS_EXCEPTION_IF_NULL(func_graph);
        DumpIR(fg_name + ".ir", func_graph);
        ExportIR(fg_name + ".dat", func_graph);
        func_graph->DumpFuncGraph(fg_name);
        MS_LOG(DEBUG) << "Dump " << fg_name << " func graph.";
      }
#endif
      counter++;
      MS_LOG(DEBUG) << "Pass " << pass.first << " end.";
    };
  }

  return true;
}

bool OptInlineAction(const ResourcePtr &res) {
  if (parallel::ParallelContext::GetInstance()->parallel_mode() == "semi_auto_parallel" ||
      parallel::ParallelContext::GetInstance()->parallel_mode() == "auto_parallel") {
    return OptimizeAction(res, kInlinePasses);
  }
  if (opt::python_pass::PyPassManager::GetInstance()->GetPassGroup(opt::python_pass::Phase::PREAD)->size() != 0) {
    return OptimizeAction(res, kInlinePasses);
  }
  return true;
}

bool GeOptimizeAction(const ResourcePtr &res) { return OptimizeAction(res, kGePasses); }

bool VmOptimizeAction(const ResourcePtr &res) {
#if ((defined ENABLE_CPU) && (!defined _WIN32))
  if (ps::PSContext::instance()->is_ps_mode()) {
    (void)kVmPasses.emplace_back(PassItem("server_communication_op_fusion", ps::Util::FuseServerCommOps));
  }
#endif
  auto ret = OptimizeAction(res, kVmPasses);
  TraceManager::ClearParseOrResolveDebugInfo();
  TraceManager::CloseRecordDebugInfoFlag();
  return ret;
}

bool PynativeElimOpt(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  if (res->manager() == nullptr) {
    MS_LOG(EXCEPTION) << "PynativeElimOpt error, manager is null.";
  }
  if (res->func_graph() == nullptr) {
    MS_LOG(EXCEPTION) << "PynativeElimOpt error, graph is null.";
  }
  return PynativeOptPass(res);
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

  if (!ms_ctx->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK)) {
    return false;
  }
  return true;
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

bool EliminateForwardCNode(const ResourcePtr &res) {
  // This function only works in Pynative mode. The func_graph is decorated by ms_function.
  if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode) {
    return true;
  }

  auto graph_executor = pipeline::GraphExecutorPy::GetInstance();
  MS_EXCEPTION_IF_NULL(graph_executor);
  auto phase = graph_executor->phase();
  MS_LOG(DEBUG) << "The phase of current pipeline graph is: " << phase;
  // Exporting graph in PyNative mode or only running forward process no need to do this action.
  auto pynative_exec = pynative::PynativeExecutor::GetInstance();
  if (phase.find("export") == 0 || !pynative_exec->grad_flag()) {
    MS_LOG(DEBUG) << "When exporting graph or only running forward process, no need to eliminate forward cnode.";
    auto grad_exec = pynative_exec->grad_executor();
    grad_exec->set_eliminate_forward(true);
    return true;
  }

  // Run grad process for func_graph and replace forward nodes with its output tensors.
  MS_LOG(INFO) << "Run eliminate forward nodes action.";
  MS_EXCEPTION_IF_NULL(res);
  auto ms_func_graph = res->func_graph();
  MS_EXCEPTION_IF_NULL(ms_func_graph);
  auto grad_exec = pynative_exec->grad_executor();
  bool eliminate_forward = grad_exec->eliminate_forward();
  grad_exec->set_eliminate_forward(eliminate_forward && ms_func_graph->func_graphs_used().empty());
  auto grad_graph = ad::Grad(ms_func_graph, opt::Optimizer::MakeEmptyOptimizer(res));
  MS_EXCEPTION_IF_NULL(grad_graph);
  graph_executor->SetGradGraph(grad_graph, phase);
  ModifyOutputNode(ms_func_graph);

  // Keep roots for only keeping forward func graph in resource.
  auto manager = res->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->KeepRoots({ms_func_graph});

  grad_exec->set_eliminate_forward(true);
  return true;
}

bool HasIncorporateCall(const std::vector<AnfNodePtr> &all_nodes) {
  for (const auto &node : all_nodes) {
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
    if (!node->isa<CNode>()) {
      continue;
    }
    if (GetCNodeTarget(node) == target) {
      return true;
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

  // PYNATIVE: no need set any context.
  if (context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    MS_LOG(INFO) << "Run graph mode with pynative.";
    set_ctx(false, false, false);
    return;
  }

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
    MS_LOG(INFO) << "Run Graph mode with kernelbykernel(Dynamic Shape).";
    set_ctx(false, false, false);
    return;
  }

  // GRAPH | Closure\ENV\While scenario : KernelByKernel path in MindRT.
  auto graphs = func_graph->func_graphs_used_total();
  (void)graphs.insert(func_graph);
  bool exist_func = ExistControlFlow(func_graph) ? HasIncorporateCall(all_nodes) : false;
  bool exist_while =
    std::any_of(graphs.cbegin(), graphs.cend(), [](const FuncGraphPtr &fg) { return fg->recursive(); });
  MS_LOG(INFO) << func_graph->ToString() << " exist_func: " << exist_func << " exist_while: " << exist_while;
  if (exist_while || exist_func) {
    MS_LOG(INFO) << "Run graph mode with kernelbykernel.";
    set_ctx(false, false, false);
    return;
  }

  // Multiple device targets scenario.
  if (func_graph->exist_multi_target()) {
    // Heterogeneous scenario + ControlFlow : KernelByKernel path in MindRT.
    if (ExistControlFlow(func_graph)) {
      MS_LOG(INFO) << "Run graph mode with kernelbykernel.";
      set_ctx(false, false, false);
      return;
    }
    // GRAPH | Heterogeneous scenario : No control flow, subgraph sink path in MindRT.
    MS_LOG(INFO) << "Run graph mode with subgraph sink.";
    set_ctx(true, false, false);
    return;
  }

  // GRAPH | normal network and if/for/switch scenario etc : MultiGraph path in MindRT.
  MS_LOG(INFO) << "Run graph mode with multigraph sink.";
  set_ctx(true, true, true);
  return;
}

void OriginSetRunMode(const ResourcePtr &res) {
  FuncGraphPtr func_graph = res->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto bc_ptr = res->GetResult(kBackend).cast<compile::BackendPtr>();
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

bool TaskEmitAction(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  FuncGraphPtr func_graph = res->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "TaskEmit args error";
  }
  if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode &&
      CheckGraphOutputConstOrParameter(func_graph)) {
    return true;
  }

  func_graph->SetMultiTarget();
  DisableMindRT(res);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<bool>(MS_CTX_ENABLE_MINDRT) && common::GetEnv("DISABLE_ASCEND_MINDRT") != "1") {
    SetRunMode(res->func_graph(), res->GetResult(kBackend).cast<compile::BackendPtr>().get());
  } else {
    OriginSetRunMode(res);
  }

  auto bc_ptr = res->GetResult(kBackend).cast<compile::BackendPtr>();
  MS_EXCEPTION_IF_NULL(bc_ptr);
  std::string backend = context_ptr->backend_policy();
  // The graph compiling of mindRT.
  if ((backend == kMsConvert) && context_ptr->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    TaskEmitActionForMindRT(res);
    return true;
  }

  // The graph compiling of control sink.
  if (IsCtrlSink() && backend == kMsConvert) {
    auto graph_id = bc_ptr->CompileGraph(NOT_NULL(func_graph));
    res->SetResult(kOutput, graph_id);
    return true;
  }
  std::vector<PrimitivePtr> cut_list = compile::nonlinear_ops;
  if (bc_ptr->name() == kMsConvert) {
    cut_list = compile::GetMsNonlinearOps();
  }
  std::shared_ptr<CompileGraphs> compile = std::make_shared<CompileGraphs>(bc_ptr, cut_list);
  auto vm = compile->CompileAndLink(func_graph);
  res->SetResult(kOutput, vm);
  return true;
}

bool ExecuteAction(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode &&
      CheckGraphOutputConstOrParameter(res->func_graph())) {
    return true;
  }
  if (!res->HasResult(kOutput)) {
    MS_LOG(EXCEPTION) << "Execute args error";
  }
  std::string backend = MsContext::GetInstance()->backend_policy();
  // The graph running of mindRT.
  if ((backend == kMsConvert) && MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    ExecuteActionForMindRT(res);
    return true;
  }

  // The graph running of control sink.
  if (IsCtrlSink() && backend == kMsConvert) {
    auto graph_id = res->GetResult(kOutput).cast<GraphId>();
    std::shared_ptr<compile::Backend> bc_ptr = res->GetResult(kBackend).cast<std::shared_ptr<compile::Backend>>();
    compile::MsBackend *msbc_ptr = std::dynamic_pointer_cast<compile::MsBackend>(bc_ptr).get();
    MS_EXCEPTION_IF_NULL(msbc_ptr);
    compile::VmEvalFuncPtr run =
      std::make_shared<compile::VmEvalFunc>([msbc_ptr, graph_id](const VectorRef &args) -> BaseRef {
        MS_LOG(INFO) << "Execute args size " << args.size();
        auto outs = msbc_ptr->RunGraph(graph_id, args);
        MS_LOG(DEBUG) << "out size " << outs.size();
        return outs[0];
      });
    res->SetResult(kOutput, run);
    return true;
  }

  compile::FinalVMPtr vm = res->GetResult(kOutput).cast<compile::FinalVMPtr>();
  if (vm == nullptr) {
    MS_LOG(INFO) << "Call GE to Run the func_graph instead of VM";
    return true;
  }
  compile::VmEvalFuncPtr run =
    std::make_shared<compile::VmEvalFunc>(std::bind(&compile::FinalVM::Eval, vm, std::placeholders::_1));
  res->SetResult(kOutput, run);
  return true;
}

#if ((defined ENABLE_CPU) && (!defined _WIN32))
bool StartPSWorkerAction(const ResourcePtr &) {
  ps::Worker::GetInstance().Run();
  return true;
}
bool StartFLWorkerAction(const ResourcePtr &) {
  fl::worker::FLWorker::GetInstance().Run();
  return true;
}

bool StartPSServerAction(const ResourcePtr &res) {
  if (distributed::cluster::ClusterContext::instance()->initialized()) {
    MS_LOG(INFO) << "This node is server. Start wait for finalizing.";
    if (!distributed::cluster::ClusterContext::instance()->Finalize(UINT32_MAX)) {
      MS_LOG(ERROR) << "Failed to finalize server.";
      return false;
    }
    MS_LOG(INFO) << "Server is successfully finalized.";
    return true;
  }
  MS_EXCEPTION_IF_NULL(res);
  FuncGraphPtr func_graph = res->func_graph();
  auto &ps = ps::ParameterServer::GetInstance();
  ps.Run(func_graph);
  return true;
}

bool StartServerAction(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  FuncGraphPtr func_graph = res->func_graph();
  const std::string &server_mode_ = ps::PSContext::instance()->server_mode();
  uint32_t worker_num = ps::PSContext::instance()->initial_worker_num();
  uint32_t server_num = ps::PSContext::instance()->initial_server_num();
  uint16_t fl_server_port = ps::PSContext::instance()->fl_server_port();

  // Update model threshold is a certain ratio of start_fl_job threshold.
  // update_model_threshold = start_fl_job_threshold * update_model_ratio.
  size_t start_fl_job_threshold = ps::PSContext::instance()->start_fl_job_threshold();
  float update_model_ratio = ps::PSContext::instance()->update_model_ratio();
  size_t update_model_threshold = static_cast<size_t>(std::ceil(start_fl_job_threshold * update_model_ratio));
  uint64_t start_fl_job_time_window = ps::PSContext::instance()->start_fl_job_time_window();
  uint64_t update_model_time_window = ps::PSContext::instance()->update_model_time_window();

  std::vector<fl::server::RoundConfig> rounds_config = {
    {"startFLJob", true, start_fl_job_time_window, true, start_fl_job_threshold},
    {"updateModel", true, update_model_time_window, true, update_model_threshold},
    {"getModel"},
    {"pullWeight"},
    {"pushWeight", false, 3000, true, server_num, true},
    {"pushMetrics", false, 3000, true, 1}};

  float share_secrets_ratio = ps::PSContext::instance()->share_secrets_ratio();
  uint64_t cipher_time_window = ps::PSContext::instance()->cipher_time_window();
  size_t minimum_clients_for_reconstruct = ps::PSContext::instance()->reconstruct_secrets_threshold() + 1;

  size_t exchange_keys_threshold =
    std::max(static_cast<size_t>(std::ceil(start_fl_job_threshold * share_secrets_ratio)), update_model_threshold);
  size_t get_keys_threshold =
    std::max(static_cast<size_t>(std::ceil(exchange_keys_threshold * share_secrets_ratio)), update_model_threshold);
  size_t share_secrets_threshold =
    std::max(static_cast<size_t>(std::ceil(get_keys_threshold * share_secrets_ratio)), update_model_threshold);
  size_t get_secrets_threshold =
    std::max(static_cast<size_t>(std::ceil(share_secrets_threshold * share_secrets_ratio)), update_model_threshold);
  size_t client_list_threshold = std::max(static_cast<size_t>(std::ceil(update_model_threshold * share_secrets_ratio)),
                                          minimum_clients_for_reconstruct);
  size_t push_list_sign_threshold = std::max(
    static_cast<size_t>(std::ceil(client_list_threshold * share_secrets_ratio)), minimum_clients_for_reconstruct);
  size_t get_list_sign_threshold = std::max(
    static_cast<size_t>(std::ceil(push_list_sign_threshold * share_secrets_ratio)), minimum_clients_for_reconstruct);
#ifdef ENABLE_ARMOUR
  std::string encrypt_type = ps::PSContext::instance()->encrypt_type();
  if (encrypt_type == ps::kPWEncryptType) {
    MS_LOG(INFO) << "Add secure aggregation rounds.";
    rounds_config.push_back({"exchangeKeys", true, cipher_time_window, true, exchange_keys_threshold});
    rounds_config.push_back({"getKeys", true, cipher_time_window, true, get_keys_threshold});
    rounds_config.push_back({"shareSecrets", true, cipher_time_window, true, share_secrets_threshold});
    rounds_config.push_back({"getSecrets", true, cipher_time_window, true, get_secrets_threshold});
    rounds_config.push_back({"getClientList", true, cipher_time_window, true, client_list_threshold});
    rounds_config.push_back({"reconstructSecrets", true, cipher_time_window, true, minimum_clients_for_reconstruct});
    if (ps::PSContext::instance()->pki_verify()) {
      rounds_config.push_back({"pushListSign", true, cipher_time_window, true, push_list_sign_threshold});
      rounds_config.push_back({"getListSign", true, cipher_time_window, true, get_list_sign_threshold});
    }
  }
  if (encrypt_type == ps::kStablePWEncryptType) {
    MS_LOG(INFO) << "Add stable secure aggregation rounds.";
    rounds_config.push_back({"exchangeKeys", true, cipher_time_window, true, exchange_keys_threshold});
    rounds_config.push_back({"getKeys", true, cipher_time_window, true, get_keys_threshold});
  }
#endif
  fl::server::CipherConfig cipher_config = {share_secrets_ratio,     cipher_time_window,
                                            exchange_keys_threshold, get_keys_threshold,
                                            share_secrets_threshold, get_secrets_threshold,
                                            client_list_threshold,   push_list_sign_threshold,
                                            get_list_sign_threshold, minimum_clients_for_reconstruct};

  size_t executor_threshold = 0;
  if (server_mode_ == ps::kServerModeFL || server_mode_ == ps::kServerModeHybrid) {
    executor_threshold = update_model_threshold;
    fl::server::Server::GetInstance().Initialize(true, true, fl_server_port, rounds_config, cipher_config, func_graph,
                                                 executor_threshold);
  } else if (server_mode_ == ps::kServerModePS) {
    executor_threshold = worker_num;
    fl::server::Server::GetInstance().Initialize(true, false, 0, rounds_config, cipher_config, func_graph,
                                                 executor_threshold);
  } else {
    MS_LOG(EXCEPTION) << "Server mode " << server_mode_ << " is not supported.";
    return false;
  }
  fl::server::Server::GetInstance().Run();
  return true;
}

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

bool DistributedSplitAction(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  FuncGraphPtr func_graph = res->func_graph();
  auto node = distributed::cluster::ClusterContext::instance()->node();
  MS_EXCEPTION_IF_NULL(node);
  auto node_role = distributed::cluster::ClusterContext::instance()->node_role();

  parallel::GraphSplitterPtr splitter =
    std::make_shared<parallel::GraphSplitter>(func_graph, node->rank_id(), node_role);
  MS_EXCEPTION_IF_NULL(splitter);
  splitter->Run();
  return true;
}
#endif

// The parallel primitive related valuenode might be partitioned so that its value changes by device,
// that will result in a synchronization error due to different executing order.
// Here we temporarily avoid the problem by skipping valuenode merging used by parallel related primitive,
// the final solution will be proposed later as a parallel feature.
bool KeepValueNodeDuplication(const AnfNodePtr &value_node, const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  MS_EXCEPTION_IF_NULL(res->manager());
  auto &node_users = res->manager()->node_users();
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

bool RemoveValueNodeDuplicationsAction(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  FuncGraphPtr func_graph = res->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "Remove value node duplications error.";
  }
  auto manager = res->manager();
  // Remove duplicated value nodes, due to replace operation, can't use reference.
  auto value_nodes = func_graph->value_nodes();
  HashCache hash_cache;
  HashValue hashes;
  for (const auto &value_pair : value_nodes) {
    if (KeepValueNodeDuplication(value_pair.first, res)) {
      continue;
    }
    TryToDoReplace(manager.get(), value_pair.first, &hash_cache, &hashes);
  }
  return true;
}

bool PipelineSplitAction(const ResourcePtr &res) { return PipelineSplitPass(res); }

bool ValidateAction(const ResourcePtr &res) { return ValidatePass(res); }

bool GeSpecializedAction(const ResourcePtr &res) { return GeSpecializedPass(res); }

bool SetMindIRGraphAction(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  res->set_is_load(true);
  auto cell = py::cast<CellPtr>(res->source_input());
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
  res->set_func_graph(fg);
  FuncGraphManagerPtr mng = fg->manager();
  if (mng == nullptr) {
    auto res_mng = res->manager();
    MS_EXCEPTION_IF_NULL(res_mng);
    res_mng->AddFuncGraph(fg);
    fg->set_manager(res_mng);
  }
  abstract::AbstractBasePtrList broaded_args;
  const auto &args_spec_list = res->args_spec();
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
    MS_LOG(WARNING) << "The input arguments is not compatible with the function graph which has been exported before."
                    << "Please check the args is same with export.\n"
                    << "The export input argument size: " << func_args.size() << "\n"
                    << "The load input argument size: " << broaded_args.size() << "\n"
                    << "Export input args info: " << abstract::ArgsToString(func_args) << "\n"
                    << "The input args info: " << abstract::ArgsToString(broaded_args);
    is_equal_input_args = false;
  }

  // suppose that there is not KeywordArgument for the top graph
  // get the hyper parameter
  for (const auto &param : fg->parameters()) {
    auto param_node = std::static_pointer_cast<Parameter>(param);
    MS_EXCEPTION_IF_NULL(param_node);
    if (param_node->has_default()) {
      auto value = param_node->default_param();
      MS_EXCEPTION_IF_NULL(value);
      auto abs_value = value->ToAbstract()->cast<abstract::AbstractTensorPtr>();
      auto ref_key = std::make_shared<RefKey>(param_node->name());
      auto abs_ref_key = ref_key->ToAbstract();
      auto abs_ref = std::make_shared<abstract::AbstractRef>(abs_ref_key, abs_value);
      broaded_args.push_back(abs_ref);
    }
  }

  if (is_equal_input_args) {
    (void)AbstractAnalyze(res, res->func_graph(), broaded_args, true);
  } else {
    // Use InferMindir which will find c++ infer in eval_map and backend_eval_map;
    (void)InferMindir(res->func_graph(), args_spec_list, true);
  }
  auto it = abstract::AnalysisResultCacheMgr::GetInstance().begin();
  auto it_end = abstract::AnalysisResultCacheMgr::GetInstance().end();
  for (; it != it_end; ++it) {
    it->first->node()->set_abstract(it->second->abstract());
  }
  abstract::AnalysisResultCacheMgr::GetInstance().Clear();
  return true;
}

bool ActionPyStub(const ResourcePtr &res, opt::python_pass::Phase phase) {
  MS_EXCEPTION_IF_NULL(res->manager());
  MS_EXCEPTION_IF_NULL(res->func_graph());
  auto ppm = opt::python_pass::PyPassManager::GetInstance();
  ppm->SetResource(res);
  return ppm->GetPassGroup(phase)->Run(res->func_graph());
}

bool PreAdActionPyStub(const ResourcePtr &res) {
  if (!ActionPyStub(res, opt::python_pass::Phase::PREAD)) {
    MS_LOG(DEBUG) << "No Match.";
  }
  return true;
}

bool OptActionVmPyStub(const ResourcePtr &res) {
  if (ActionPyStub(res, opt::python_pass::Phase::OPT)) {
    if (opt::python_pass::PyPassManager::GetInstance()->ShouldRenorm()) {
      // Renomalize
      FuncGraphPtr func_graph = res->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      abstract::AbstractBasePtrList args_spec;
      auto parameters = func_graph->parameters();
      (void)std::transform(parameters.begin(), parameters.end(), std::back_inserter(args_spec),
                           [](const AnfNodePtr &p) -> AbstractBasePtr { return p->abstract(); });
      FuncGraphPtr new_fg = Renormalize(res, func_graph, args_spec);
      res->set_func_graph(new_fg);
      res->set_args_spec(args_spec);
    }
    if (opt::python_pass::PyPassManager::GetInstance()->ShouldReOpt()) {
      return VmOptimizeAction(res);
    }
  }
  return true;
}

bool OptActionGePyStub(const ResourcePtr &res) {
  if (ActionPyStub(res, opt::python_pass::Phase::OPT)) {
    if (opt::python_pass::PyPassManager::GetInstance()->ShouldRenorm()) {
      // Renomalize
      FuncGraphPtr func_graph = res->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      abstract::AbstractBasePtrList args_spec;
      auto parameters = func_graph->parameters();
      (void)std::transform(parameters.begin(), parameters.end(), std::back_inserter(args_spec),
                           [](const AnfNodePtr &p) -> AbstractBasePtr { return p->abstract(); });
      FuncGraphPtr new_fg = Renormalize(res, func_graph, args_spec);
      res->set_func_graph(new_fg);
      res->set_args_spec(args_spec);
    }
    if (opt::python_pass::PyPassManager::GetInstance()->ShouldReOpt()) {
      return GeOptimizeAction(res);
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

  auto multi_graphs = parallel::CostModelContext::GetInstance()->is_multi_subgraphs();
  if (!multi_graphs && pipeline::GetJitLevel() != "o0") {
    (void)actions.emplace_back(std::make_pair("combine_like_graphs", CombineLikeGraphs));
  }

  (void)actions.emplace_back(std::make_pair("inference_opt_prepare", InferenceOptPrepareAction));
  // Eliminate unused parameters before renormalize.
  (void)actions.emplace_back(std::make_pair("elininate_unused_parameter", EliminateUnusedParameterAction));
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
  // optimize
  (void)actions.emplace_back(std::make_pair("optimize", GeOptimizeAction));
  // Add opt-stage python pass stub
  (void)actions.emplace_back(std::make_pair("py_opt", OptActionGePyStub));
  (void)actions.emplace_back(std::make_pair("remove_value_node_duplications", RemoveValueNodeDuplicationsAction));
  (void)actions.emplace_back(std::make_pair("auto_monad_reorder", OrderEnforceAction));
  (void)actions.emplace_back(std::make_pair("ge_specialized_prepare", GeSpecializedAction));
  (void)actions.emplace_back(std::make_pair("validate", ValidateAction));
  return actions;
}

std::vector<ActionItem> VmPipeline(const ResourcePtr &resource) {
  std::vector<ActionItem> actions;
  // If enable compilation cache and the cache is read successfully, only do the backend actions.
  if (!resource->EnableCompileCache() || resource->func_graph() == nullptr) {
    actions = CommonPipeline();

    // optimize
    (void)actions.emplace_back(std::make_pair("optimize", VmOptimizeAction));

    // Add opt-stage python pass stub
    (void)actions.emplace_back(std::make_pair("py_opt", OptActionVmPyStub));

    (void)actions.emplace_back(std::make_pair("auto_monad_reorder", OrderEnforceAction));

    // eliminate forward cnode for grad graph
    (void)actions.emplace_back(std::make_pair("eliminate_forward_cnode", EliminateForwardCNode));

    (void)actions.emplace_back(std::make_pair("validate", ValidateAction));
  }

#if ((defined ENABLE_CPU) && (!defined _WIN32))
  if (ps::PSContext::instance()->is_worker()) {
    if (distributed::cluster::ClusterContext::instance()->initialized()) {
      MS_LOG(INFO) << "This worker is initialized. No need to add worker action.";
    } else {
      std::string server_mode = ps::PSContext::instance()->server_mode();
      if (server_mode == ps::kServerModeFL || server_mode == ps::kServerModeHybrid) {
        (void)actions.emplace_back(std::make_pair("worker", StartFLWorkerAction));
      } else {
        (void)actions.emplace_back(std::make_pair("worker", StartPSWorkerAction));
      }
    }
  }
#endif
  // compile the ANF graph
  (void)actions.emplace_back(std::make_pair("task_emit", TaskEmitAction));

  // to execute the graph
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
  // compile the ANF graph
  (void)actions.emplace_back(std::make_pair("task_emit", TaskEmitAction));
  // to execute the graph
  (void)actions.emplace_back(std::make_pair("execute", ExecuteAction));
  return actions;
}

#if ((defined ENABLE_CPU) && (!defined _WIN32))
std::vector<ActionItem> ServerPipeline(const ResourcePtr &resource) {
  if (resource->EnableCompileCache() && resource->func_graph() != nullptr) {
    return {std::make_pair("server", StartServerAction)};
  }
  auto actions = CommonPipeline();
  (void)actions.emplace_back(std::make_pair("optimize", VmOptimizeAction));
  (void)actions.emplace_back(std::make_pair("validate", ValidateAction));
  (void)actions.emplace_back(std::make_pair("server", StartServerAction));
  return actions;
}

std::vector<ActionItem> PServerPipeline(const ResourcePtr &resource) {
  if (resource->EnableCompileCache() && resource->func_graph() != nullptr) {
    return {std::make_pair("pserver", StartPSServerAction)};
  }
  auto actions = CommonPipeline();
  (void)actions.emplace_back(std::make_pair("optimize", VmOptimizeAction));
  (void)actions.emplace_back(std::make_pair("auto_monad_reorder", OrderEnforceAction));
  (void)actions.emplace_back(std::make_pair("validate", ValidateAction));
  (void)actions.emplace_back(std::make_pair("pserver", StartPSServerAction));
  return actions;
}

std::vector<ActionItem> PSchedulerPipeline(const ResourcePtr &resource) {
  if (resource->EnableCompileCache() && resource->func_graph() != nullptr) {
    return {std::make_pair("scheduler", StartPSSchedulerAction)};
  }
  auto actions = CommonPipeline();
  (void)actions.emplace_back(std::make_pair("optimize", VmOptimizeAction));
  (void)actions.emplace_back(std::make_pair("auto_monad_reorder", OrderEnforceAction));
  (void)actions.emplace_back(std::make_pair("validate", ValidateAction));
  (void)actions.emplace_back(std::make_pair("scheduler", StartPSSchedulerAction));
  return actions;
}
#endif
}  // namespace pipeline
}  // namespace mindspore
