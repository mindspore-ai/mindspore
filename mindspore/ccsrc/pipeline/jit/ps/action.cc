/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/ps/action.h"

#include <memory>
#include <utility>
#include <vector>
#include <set>
#include <string>
#include <algorithm>
#include <functional>

#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "ir/anf.h"
#include "ir/func_graph_cloner.h"
#include "ir/param_info.h"
#include "ir/cell.h"
#include "include/common/utils/python_adapter.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "abstract/abstract_value.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/operator/composite/composite.h"
#include "frontend/parallel/step_auto_parallel.h"
#include "frontend/parallel/graph_util/graph_splitter.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "pipeline/jit/ps/pipeline.h"
#include "pipeline/jit/ps/pass.h"
#include "pipeline/jit/ps/parse/parse_base.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "pipeline/jit/ps/static_analysis/auto_monad.h"
#include "pipeline/jit/ps/static_analysis/order_enforce.h"
#include "pipeline/jit/ps/static_analysis/static_analysis.h"
#include "pipeline/jit/ps/static_analysis/async_eval_result.h"
#include "pipeline/jit/ps/static_analysis/program_specialize.h"
#include "pipeline/jit/ps/resource.h"
#include "pipeline/jit/ps/remove_value_node_dup.h"
#include "pipeline/pynative/pynative_execute.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/ad/grad.h"
#include "frontend/expander/pack/packfunc.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "utils/phase.h"
#include "backend/graph_compiler/transform.h"
#include "load_mindir/infer_mindir.h"
#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "include/backend/debug/profiler/profiling.h"
#include "frontend/optimizer/fallback_rewriter.h"
#include "pipeline/jit/ps/load_mindir.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/backend/distributed/cluster/cluster_context.h"
#include "include/backend/distributed/ps/ps_context.h"
#include "include/backend/distributed/ps/util.h"
#endif

namespace mindspore {
namespace pipeline {
namespace {
const auto kFirstInput = 1;
const auto kSecondInput = 2;

bool ExistControlFlow(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  return !func_graph->func_graphs_used_total().empty();
}

bool EnableGradForScalar(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  return MsContext::GetInstance()->get_param<bool>(MS_CTX_GRAD_FOR_SCALAR) && abs->BuildType() != nullptr &&
         abs->BuildType()->isa<Number>();
}

bool EnableSequenceBroaden(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  return abs->isa<abstract::AbstractSequence>() &&
         abs->cast<abstract::AbstractSequencePtr>()->ContainsAllBroadenTensors();
}

bool ContainsAbstractFunction(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractFunction>()) {
    return true;
  }
  if (abs->isa<abstract::AbstractSequence>()) {
    const auto &abs_list = abs->cast<abstract::AbstractSequencePtr>()->elements();
    return std::any_of(abs_list.cbegin(), abs_list.cend(),
                       [](const auto &elem) { return ContainsAbstractFunction(elem); });
  }
  if (abs->isa<abstract::AbstractDictionary>()) {
    const auto &abs_pair_list = abs->cast<abstract::AbstractDictionaryPtr>()->elements();
    return std::any_of(abs_pair_list.cbegin(), abs_pair_list.cend(),
                       [](const auto &pair) { return ContainsAbstractFunction(pair.second); });
  }
  return false;
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
    if ((param_abs->BuildValue() == kValueAny && !ContainsAbstractFunction(param_abs)) ||
        EnableGradForScalar(param_abs) || EnableSequenceBroaden(param_abs)) {
      new_paras.push_back(param_node);
    } else {
      MS_LOG(INFO) << "Remove the " << i << "th parameter, since it's passed a constant argument.";
    }
  }
  func_graph->set_parameters(new_paras);
}

bool IsDynamicShapeGraph(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return(), SuccDeeperSimple);
  return std::any_of(node_list.begin(), node_list.end(), [](const AnfNodePtr &node) {
    if (common::AnfAlgo::IsCallNode(node)) {
      return false;
    }
    return common::AnfAlgo::IsDynamicShape(node);
  });
}

// Exist ScalarAdd ScalarSub etc OPS which will backoff to CPU
bool IsNeedBackoffGraph(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return(), SuccDeeperSimple);
  return std::any_of(node_list.begin(), node_list.end(), [](const AnfNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      MS_LOG(DEBUG) << "Node is not a cnode.";
      return false;
    }
    auto abs = node->abstract();
    if (abs == nullptr) {
      MS_LOG(DEBUG) << "Node " << node->fullname_with_scope()
                    << " has no abstract, Debug String: " << node->DebugString();
      return false;
    }
    return abs->isa<abstract::AbstractScalar>() && abs->BuildValue()->ContainsValueAny() &&
           !IsPrimitiveCNode(node, prim::kPrimDepend);
  });
}

// Disable mindRT in the heterogeneous scenario + dynamic_shape scenario.
void DisableMindRT(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!context_ptr->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    return;
  }
#if defined(__linux__) && defined(WITH_BACKEND)
  if (ps::PSContext::instance()->cache_enable()) {
    return;
  }
#endif
}

void TaskEmitActionForMindRT(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  // Get the mindRT backend.
  auto bc_ptr = resource->GetBackend();
  // In pyexecute kernel, the input data would be stored in user data which is a python object, this converter
  // is used to convert user data to device ptr in device address.
  compile::set_pydata_converter([](const py::object &obj, ValuePtr *value) { return parse::ConvertData(obj, value); });
  auto mindrt_bc_ptr = std::dynamic_pointer_cast<compile::MindRTBackend>(bc_ptr);
  MS_EXCEPTION_IF_NULL(mindrt_bc_ptr);
  MS_EXCEPTION_IF_NULL(resource->func_graph());
  auto actor_info = mindrt_bc_ptr->CompileGraphs(resource->func_graph());
  resource->SetResult(kOutput, actor_info);
  resource->SetResult(kActorInfo, actor_info);
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

FuncGraphPtr ConstructGraphForEval(const ValuePtr &func, const abstract::AbstractBasePtrList &args_abs) {
  auto func_abs = func->ToAbstract();
  if (!func_abs->isa<abstract::AbstractFunction>()) {
    MS_LOG(EXCEPTION) << "The value : " << func->ToString() << " is not a callable object.";
  }
  // construct a function graph.
  auto infer_graph = std::make_shared<FuncGraph>();
  std::vector<AnfNodePtr> inputs = {std::make_shared<ValueNode>(func)};
  std::transform(args_abs.begin(), args_abs.end(), std::back_inserter(inputs),
                 [infer_graph](const AbstractBasePtr &) -> AnfNodePtr { return infer_graph->add_parameter(); });
  auto infer_node = infer_graph->NewCNode(inputs);
  infer_graph->set_return(infer_node);
  return infer_graph;
}
}  // namespace
using CompileGraphs = compile::CompileGraphs;
using abstract::AnalysisResult;
using mindspore::abstract::AnalysisContextPtr;

// Whether this process in a MindSpore cluster.
static bool is_cluster_initialized = false;

abstract::AnalysisResult AbstractAnalyze(const abstract::AnalysisEnginePtr &engine, const FuncGraphPtr &func_graph,
                                         const abstract::AbstractBasePtrList &args_abs, bool is_load_resoure,
                                         bool clear) {
  MS_LOG(DEBUG) << "AbstractAnalyze start";
  py::gil_scoped_acquire gil;
  MS_EXCEPTION_IF_NULL(engine);
  if (clear || is_load_resoure) {
    auto manager = engine->func_graph_manager();
    MS_EXCEPTION_IF_NULL(manager);
    engine->Clear();
    static const auto enable_eliminate_unused_element = (common::GetEnv("MS_DEV_ENABLE_DDE") != "0");
    for (auto &node : manager->all_nodes()) {
      MS_EXCEPTION_IF_NULL(node);
      // Handle previous inferred value for CNode if is loaded from MindIR
      // If the primitive is not defined in front end, keep the inferred value loaded from MindIR.
      if (is_load_resoure) {
        auto primitive = GetCNodePrimitive(node);
        if (primitive != nullptr) {
          auto is_load = primitive->GetAttr("is_load");
          if (abstract::GetPrimEvaluator(primitive, engine) == nullptr && is_load != nullptr &&
              GetValue<bool>(is_load)) {
            MS_LOG(INFO) << "The primitive is not defined in front end. Primitive: " << primitive->ToString();
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
  MS_LOG(INFO) << "function call depth: " << abstract::FunctionCallDepth()
               << ", simulate call depth: " << abstract::StackFrameDepth();
  MS_LOG(DEBUG) << "AbstractAnalyze end";
  return res;
}

abstract::AnalysisResult AbstractAnalyze(const ValuePtr &func, const abstract::AbstractBasePtrList &args_abs,
                                         bool clear) {
  auto infer_graph = ConstructGraphForEval(func, args_abs);
  auto manager = Manage(infer_graph, true);
  auto engine = std::make_shared<abstract::AnalysisEngine>(abstract::GetPrimEvaluatorConstructors(), manager);
  return AbstractAnalyze(engine, infer_graph, args_abs, false, clear);
}

FuncGraphPtr ProgramSpecialize(const abstract::AnalysisEnginePtr &engine, const FuncGraphPtr &func_graph,
                               const abstract::AnalysisContextPtr &context) {
  MS_EXCEPTION_IF_NULL(engine);
  MS_LOG(DEBUG) << "ProgramSpecialize start";
  abstract::ProgramSpecializer specializer(engine);
  FuncGraphPtr result = specializer.Run(func_graph, context);
  auto manager = engine->func_graph_manager();
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
  auto engine = resource->engine();
#ifdef ENABLE_PROFILE
  double t1 = GetTime();
#endif
  abstract::AnalysisResult result = AbstractAnalyze(engine, func_graph, args_abs, resource->is_load(), true);
#ifdef ENABLE_PROFILE
  double t2 = GetTime();
#endif
  auto res = ProgramSpecialize(engine, func_graph, result.context);
  resource->set_func_graph(res);
#ifdef ENABLE_PROFILE
  double t3 = GetTime();
  MsProfile::StatTime("renormalize.infer", t2 - t1);
  MsProfile::StatTime("renormalize.specialize", t3 - t2);
#endif

  MS_LOG(DEBUG) << "Renormalize end";

  return res;
}

FuncGraphPtr Renormalize(const ValuePtr &func, const abstract::AbstractBasePtrList &args_abs) {
  auto func_abs = func->ToAbstract();
  if (!func_abs->isa<abstract::AbstractFunction>()) {
    MS_LOG(EXCEPTION) << "The value: " << func->ToString() << " is not a callable object.";
  }
  auto infer_graph = ConstructGraphForEval(func, args_abs);
  auto manager = Manage(infer_graph, true);
  auto engine = std::make_shared<abstract::AnalysisEngine>(abstract::GetPrimEvaluatorConstructors(), manager);
#ifdef ENABLE_PROFILE
  double t1 = GetTime();
#endif
  auto res = AbstractAnalyze(engine, infer_graph, args_abs, false);
#ifdef ENABLE_PROFILE
  double t2 = GetTime();
#endif

  auto spec_graph = ProgramSpecialize(engine, infer_graph, res.context);
#ifdef ENABLE_PROFILE
  double t3 = GetTime();
  MsProfile::StatTime("renormalize.infer", t2 - t1);
  MsProfile::StatTime("renormalize.specialize", t3 - t2);
#endif
  return spec_graph;
}

void SetMindIRLoadFlag(const ResourcePtr &resource) {
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
  TraceManager::OpenParserDebugInfoFlag();
  if (!resource->source_input()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Parse error";
  }

  py::object input = resource->source_input();
  // CellList need convert to FuncGraph in Parse, add flag for input from top graph.
  if (py::hasattr(input, PYTHON_CELL_AS_LIST)) {
    py::setattr(input, PYTHON_CELL_LIST_FROM_TOP, py::bool_(true));
  }
  parse::Parser::InitParserEnvironment(input);
  parse::Parser::EnableDeferResolve(false);
  py::module path = py::module::import("os.path");
  auto dir = path.attr("dirname")(py::globals()["__file__"]).cast<std::string>();

  python_adapter::set_python_env_flag(true);
  python_adapter::SetPythonPath(dir);

  ValuePtrList args_value_list;
  (void)std::transform(resource->args_abs().begin(), resource->args_abs().end(), std::back_inserter(args_value_list),
                       [](const AbstractBasePtr &abs) { return abs->BuildValue(); });
  parse::DataConverter data_converter(args_value_list, true);
  auto converted_ret = data_converter.ConvertData(input);
  if (converted_ret == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Attribute convert error with type:" << std::string(py::str(input));
  }

  auto top_graph = converted_ret->cast<FuncGraphPtr>();
  if (top_graph == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Object to parse " << std::string(py::str(input)) << " is not function or cell.";
  }
  if (py::hasattr(input, parse::PYTHON_PARSE_METHOD) || py::hasattr(input, "__jit_function__")) {
    (void)std::for_each(top_graph->parameters().begin(), top_graph->parameters().end(),
                        [](const AnfNodePtr &param) { param->cast<ParameterPtr>()->set_is_top_graph_param(true); });
  }
  parse::Parser::UpdateTopFuncGraph(top_graph);
  resource->set_func_graph(top_graph);
  FuncGraphManagerPtr manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(top_graph);

  parse::Parser::EnableDeferResolve(true);
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
      MS_LOG(INTERNAL_EXCEPTION) << "Clone func graph failed! " << fg->ToString();
    }
    auto base_graph = cloned_fg_iter->second;
    MS_LOG(DEBUG) << "Basegraph:" << base_graph->ToString();

    if (fg->parameter_obj_nodes().empty() || graphs.size() <= 1 || fg->has_flag(FUNC_GRAPH_OUTPUT_NO_RECOMPUTE) ||
        fg->stage() != -1) {
      continue;
    }
    auto &cloned_nodes = cloner.cloned_nodes();
    for (auto &fv : fg->parameter_obj_nodes()) {
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
    MS_LOG(DEBUG) << "Fg0 parameter_obj_nodes size :" << fg->parameter_obj_nodes().size();

    for (auto &g : graphs) {
      auto &fvs = g->parameter_obj_nodes();
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
// Get all the trainable parameters of the reusable cell.
void GetTrainableParameters(const FuncGraphPtr &fg, std::vector<AnfNodePtr> *parameters) {
  if (common::GetEnv("MS_DEV_DISABLE_TRACE") != "on" && expander::IsPackGraph(fg)) {
    return expander::GetPackGraphParams(fg, parameters);
  }
  MS_EXCEPTION_IF_NULL(parameters);
  if (fg->manager() == nullptr) {
    MS_LOG(INFO) << fg->ToString() << " manager is null. This Cell init should not be assigned cell_attr_register.";
    return;
  }
  auto used_fgs = fg->func_graphs_used_total();
  std::set<const AnfNode *> memo;
  for (auto &g : used_fgs) {
    for (auto &item : g->parameter_obj_nodes()) {
      MS_LOG(DEBUG) << fg->ToString() << " has_default: " << item->cast<ParameterPtr>()->has_default()
                    << " parameter: " << item->cast<ParameterPtr>()->ToString();
      if (item->cast<ParameterPtr>()->has_default() && memo.emplace(item.get()).second) {
        parameters->push_back(item);
      }
    }
    if (common::GetEnv("MS_DEV_DISABLE_TRACE") != "on" && expander::IsPackGraph(g)) {
      expander::GetSubPackGraphParams(fg, g, parameters, &memo);
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
    MS_LOG(INTERNAL_EXCEPTION) << "Clone func graph failed! " << fg->ToString();
  }
  auto reusing_graph = cloned_fg_iter->second;
  if (common::GetEnv("MS_DEV_DISABLE_TRACE") != "on" && expander::IsPackGraph(fg)) {
    return expander::UpdateReusingGraphForPack(reusing_graph, parameters);
  }

  // Make the reusable graph to be the no_inline status.
  reusing_graph->set_flag(FUNC_GRAPH_FLAG_NO_INLINE, true);
  reusing_graph->set_flag(FUNC_GRAPH_FLAG_CELL_REUSE, true);

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
  origin_graph->erase_flag(FUNC_GRAPH_OUTPUT_NO_RECOMPUTE);
}

void SetCalledSubGraphMixedPrecisionFlag(const FuncGraphPtr &func_graph) {
  FuncGraphPtr fp16_mixed_precision_fg;
  FuncGraphPtr fp32_mixed_precision_fg;
  FuncGraphPtr bf16_mixed_precision_fg;
  // Find the first subgraph which has mixed precision flag.
  for (auto &item : func_graph->func_graphs_used()) {
    if (item.first->has_flag(GRAPH_FLAG_MIX_PRECISION_FP16)) {
      fp16_mixed_precision_fg = item.first;
    }
    if (item.first->has_flag(GRAPH_FLAG_MIX_PRECISION_FP32)) {
      fp32_mixed_precision_fg = item.first;
    }
    if (item.first->has_flag(GRAPH_FLAG_MIX_PRECISION_BF16)) {
      bf16_mixed_precision_fg = item.first;
    }
    if ((fp32_mixed_precision_fg != nullptr) || (fp16_mixed_precision_fg != nullptr) ||
        (bf16_mixed_precision_fg != nullptr)) {
      break;
    }
  }

  // Add mixed precision flag to new subgraph which call subgraph in set.
  if (fp16_mixed_precision_fg != nullptr) {
    for (auto sub_fg : fp16_mixed_precision_fg->func_graphs_used_total()) {
      sub_fg->set_flag(GRAPH_FLAG_MIX_PRECISION_FP16, true);
    }
  }
  if (fp32_mixed_precision_fg != nullptr) {
    for (auto sub_fg : fp32_mixed_precision_fg->func_graphs_used_total()) {
      sub_fg->set_flag(GRAPH_FLAG_MIX_PRECISION_FP32, true);
    }
  }
  if (bf16_mixed_precision_fg != nullptr) {
    for (auto sub_fg : bf16_mixed_precision_fg->func_graphs_used_total()) {
      sub_fg->set_flag(GRAPH_FLAG_MIX_PRECISION_BF16, true);
    }
  }
}
}  // namespace

// Make the reusable cell to be the reusable function graph.
bool GraphReusingAction(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  const auto &obj_map = parse::data_converter::GetObjGraphs();
  bool cell_reused = false;
  for (const auto &[cell_key, graphs] : obj_map) {
    MS_LOG(DEBUG) << "Start to handle the reusable graph: " << cell_key << ", size: " << graphs.size();
    const auto &fg = graphs[0];
    // fg->parameter_obj_nodes().empty() have been handled by combine like.
    if (!fg->parameter_obj_nodes().empty()) {
      MS_LOG(INFO) << "Finish handling the reusable graph: " << cell_key;
      continue;
    }
    if (cell_key.find("lazy_inline") == cell_key.npos) {
      continue;
    }
    auto reusing_graph = GenerateReusingGraph(fg);
    if (reusing_graph == nullptr) {
      MS_LOG(WARNING) << "Failed to generate reused graph for cell_key: " << cell_key;
      continue;
    }
    // Let the original cell graph call the reusable graph.
    (void)std::for_each(graphs.begin(), graphs.end(), [&reusing_graph](const auto &origin_graph) {
      ReplaceWithReusingGraph(reusing_graph, origin_graph);
    });
    cell_reused = true;
    MS_LOG(DEBUG) << "Finish handling the reusable graph: " << cell_key;
  }
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  const bool enable_ge = context->backend_policy() == "ge";
  const bool force_no_inline = common::GetEnv("MS_FORCE_NO_INLINE") == "1";
  context->SetCellReuseLevel(CellReuseLevel::kNoCellReuse);
  if (cell_reused) {
    MS_LOG(INFO) << "Cell reuse(@lazy_inline) actually takes effect.";
    auto cell_reuse_level =
      (enable_ge && !context->IsKByKExecutorMode()) ? CellReuseLevel::kNoInline : CellReuseLevel::kLazyInline;
    if (force_no_inline) {
      cell_reuse_level = CellReuseLevel::kNoInline;
    }
    context->SetCellReuseLevel(cell_reuse_level);
  }
  return true;
}

// Used for excluding the func graphs in VMap.
bool UsedByVmap(const FuncGraphPtr &func_graph) {
  const auto &cnodes_index = func_graph->func_graph_cnodes_index();
  if (cnodes_index.empty()) {
    return false;
  }
  const auto matcher = [&func_graph](const std::pair<const CNodeIndexPairPtr, int64_t> &cnode_index) {
    const auto &cnode = cnode_index.first->first;
    const auto &vmap_meta = GetCNodeValueWithoutDoSignature(cnode);
    if (vmap_meta != nullptr && vmap_meta->isa<prim::VmapOperation>()) {
      MS_LOG(DEBUG) << "Found VMap CNode: " << cnode->DebugString();
      return true;
    }
    // The func graph is used in MakeTuple or UnpackGraph.
    const auto user_matcher = [](const FuncGraphPtr &func_graph, const AnfNodePtr &cnode) {
      auto manager = func_graph->manager();
      MS_EXCEPTION_IF_NULL(manager);
      auto &users = manager->node_users()[cnode];
      for (const auto &user : users) {
        const auto &user_vmap_meta = GetCNodeValueWithoutDoSignature(user.first);
        if (user_vmap_meta != nullptr && user_vmap_meta->isa<prim::VmapOperation>()) {
          MS_LOG(DEBUG) << "Found VMap CNode: " << user.first->DebugString();
          return true;
        }
      }
      return false;
    };
    const auto unpack_graph_prim = GetCNodePrimitive(cnode);
    if (unpack_graph_prim != nullptr && unpack_graph_prim->isa<prim::UnpackGraphPrimitive>()) {
      return user_matcher(func_graph, cnode);
    }
    if (IsPrimitiveCNode(cnode, prim::kPrimMakeTuple)) {
      return user_matcher(func_graph, cnode);
    }
    // Deal with F.vmap(fn, ...) in construct().
    // Not check fn passed from nested func graph calls.
    if (cnode_index.first->second == 1) {
      const auto vmap_func = GetCNodeFuncGraph(cnode);
      if (vmap_func == nullptr) {
        return false;
      }
      auto first_param = vmap_func->parameters()[0];
      return user_matcher(func_graph, first_param);
    }
    return false;
  };
  return std::any_of(cnodes_index.cbegin(), cnodes_index.cend(), matcher);
}

bool PreCConvAction(const ResourcePtr &resource) {
  static const bool enable_pre_lift = (common::GetEnv("MS_DEV_PRE_LIFT") == "1");
  if (!enable_pre_lift) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(resource);
  MS_EXCEPTION_IF_NULL(resource->func_graph());
  FuncGraphPtr func_graph = resource->func_graph();
  FuncGraphPtr new_fg = LiftingClone(func_graph, false, UsedByVmap);
  resource->set_func_graph(new_fg);
  return GradPartialTransformPass(resource);
}

bool SymbolResolveAction(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (resource->manager() == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "SymbolResolve error, manager is null";
  }
  auto func_graph = resource->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "SymbolResolve error, graph is null";
  }
  bool ret = parse::ResolveFuncGraph(func_graph, resource);
  // Remove unused nodes in cnode order list,
  // and check isolated side-effect nodes.
  if (func_graph != nullptr) {
    func_graph->EraseUnusedNodeInOrder();
    for (auto fg : func_graph->func_graphs_used_total()) {
      if (fg != nullptr) {
        fg->EraseUnusedNodeInOrder();
      }
    }
  }
  return ret;
}

bool SetMixedPrecisionAction(const ResourcePtr &resource) {
  if (resource->manager() == nullptr) {
    MS_LOG(EXCEPTION) << "SetMixedPrecisionAction error, manager is null";
  }
  auto func_graph = resource->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "SetMixedPrecisionAction error, graph is null";
  }
  SetCalledSubGraphMixedPrecisionFlag(func_graph);
  MS_LOG(DEBUG) << "Finish set mixed Precision flag in subgraph. ";
  return true;
}

bool AutoMonadAction(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (resource->manager() == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Auto-Monad failed, manager is null";
  }
  auto func_graph = resource->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Auto-Monad failed, graph is null";
  }
  (void)pipeline::AutoMonad(func_graph);
  return true;
}

bool OrderEnforceAction(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (resource->manager() == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Order-Enforce error, manager is null";
  }
  auto func_graph = resource->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Order-Enforce error, graph is null";
  }
  pipeline::OrderEnforce(func_graph);
  return true;
}

bool MetaUnpackPrepareAction(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (resource->manager() == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "MetaUnpackPrepareAction error, manager is null.";
  }
  if (resource->func_graph() == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "MetaUnpackPrepareAction error, graph is null.";
  }
  return MetaUnpackPreparePass(resource);
}

// Get abstract of the default value in the given parameter.
AbstractBasePtr GetDefaultValueAbstract(const ParameterPtr &param) {
  auto value = param->default_param();
  MS_EXCEPTION_IF_NULL(value);
  auto value_abs = value->ToAbstract();
  MS_EXCEPTION_IF_NULL(value_abs);
  if (value_abs->isa<abstract::AbstractMapTensor>()) {
    // Return AbstractMapTensor for map parameter.
    return value_abs;
  }
  // Make an AbstractRefTensor for the tensor value.
  auto abs_tensor = value_abs->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(abs_tensor);
  auto ref_key = std::make_shared<RefKey>(param->name());
  return std::make_shared<abstract::AbstractRefTensor>(abs_tensor, ref_key);
}

namespace {
abstract::AbstractBasePtrList GetArgsAbs(const ResourcePtr &resource) {
  FuncGraphPtr func_graph = resource->func_graph();
  abstract::AbstractBasePtrList args_abs = resource->args_abs();

  // Parallel checking.
  auto context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel::ParallelContext::GetInstance());

  // Handle the Parameter from FV inputs.
  for (const auto &param : func_graph->parameters()) {
    auto param_node = std::static_pointer_cast<Parameter>(param);
    MS_EXCEPTION_IF_NULL(param_node);
    if (param_node->has_default()) {
      auto param_abs = GetDefaultValueAbstract(param_node);
      context->ParallelParameterContextRestoreShape(func_graph, param_node, param_abs);
      (void)args_abs.emplace_back(param_abs);
    }
  }
  return args_abs;
}
}  // namespace

bool AbstractSpecializeAction(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (resource->func_graph() == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "AbstractSpecialize error";
  }
  SetMindIRLoadFlag(resource);
  // Abstract analyze
  auto engine = resource->engine();
  MS_EXCEPTION_IF_NULL(engine);

  // Check isolated side-effect nodes.
  engine->set_check_side_effect(true);
  // Analyze
  (void)profiler::CollectHostInfo(kCompiler, kAbstractSpecialize, kAbstractAnalyze, 0, 0, 0);
  AnalysisResult result =
    AbstractAnalyze(resource->engine(), resource->func_graph(), GetArgsAbs(resource), resource->is_load());
  (void)profiler::CollectHostInfo(kCompiler, kAbstractSpecialize, kAbstractAnalyze, 0, 0, 1);
  // Specialize
  (void)profiler::CollectHostInfo(kCompiler, kAbstractSpecialize, kProgramSpecialize, 0, 0, 0);
  FuncGraphPtr new_fg = ProgramSpecialize(resource->engine(), result.context->func_graph(), result.context);
  (void)profiler::CollectHostInfo(kCompiler, kAbstractSpecialize, kProgramSpecialize, 0, 0, 1);
  // Update the top func graph with the specialized graph.
  parse::Parser::UpdateTopFuncGraph(new_fg);
  resource->set_func_graph(new_fg);
  engine->set_check_side_effect(false);

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
    ProcessStatus::GetInstance().RecordStart(pass.first);
    (void)profiler::CollectHostInfo(kCompiler, kOptimize, pass.first, 0, 0, 0);
    auto profile_context = MsProfile::GetProfile()->Step(pass.first);
    auto pass_func = [&pass, &resource, &counter]() {
      MS_LOG(DEBUG) << "Pass " << pass.first << " start ...";
      auto result = pass.second(resource);
      if (!result) {
        MS_LOG(INTERNAL_EXCEPTION) << "Pass running to end, failed in pass:" << pass.first;
      }
#ifdef ENABLE_DUMP_IR
      auto context = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(context);
      if (context->CanDump(kIntroductory) && resource->func_graph() != nullptr) {
        auto fg_name = "opt_pass_" + std::to_string(counter) + "_" + pass.first;
        auto func_graph = resource->func_graph();
        MS_EXCEPTION_IF_NULL(func_graph);
        static const auto switch_order = (common::GetEnv("MS_DEV_SAVE_GRAPHS_SORT_MODE") == "1");
        if (switch_order) {
          ExportIR(fg_name + ".ir", func_graph);
        } else {
          DumpIR(fg_name + ".ir", func_graph);
        }
        if (context->CanDump(kFully)) {
          draw::Draw(fg_name + ".dot", func_graph);
        }
        MS_LOG(DEBUG) << "Dump " << fg_name << " func graph.";
      }
#endif
      counter++;
      MS_LOG(DEBUG) << "Pass " << pass.first << " end.";
    };
    ProfileExecute(profile_context, pass_func);
    (void)profiler::CollectHostInfo(kCompiler, kOptimize, pass.first, 0, 0, 1);
    ProcessStatus::GetInstance().RecordEnd();
  }

  return true;
}

bool PackExpandAction(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (resource->manager() == nullptr) {
    MS_LOG(EXCEPTION) << "PackExpandAction error, manager is null.";
  }
  if (resource->func_graph() == nullptr) {
    MS_LOG(EXCEPTION) << "PackExpandAction error, graph is null.";
  }
  return PackExpandPass(resource);
}

bool OptInlineAction(const ResourcePtr &resource) {
  if (parallel::ParallelContext::GetInstance()->parallel_mode() == "semi_auto_parallel" ||
      parallel::ParallelContext::GetInstance()->parallel_mode() == "auto_parallel") {
    return OptimizeAction(resource, kInlinePasses);
  }
  return true;
}

bool VmOptimizeAction(const ResourcePtr &resource) {
#if defined(__linux__) && defined(WITH_BACKEND)
  if (ps::PSContext::instance()->is_ps_mode()) {
    (void)kVmPasses.emplace_back(PassItem("server_communication_op_fusion", [](const ResourcePtr &res) -> bool {
      MS_EXCEPTION_IF_NULL(res);
      return ps::Util::FuseServerCommOps(res->func_graph());
    }));
  }
#endif
  auto ret = OptimizeAction(resource, kVmPasses);
  TraceManager::CloseParserDebugInfoFlag();
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

bool GetJitBpropGraph(const ResourcePtr &resource) {
  // This function only works in Pynative mode. The func_graph is decorated with 'jit'.
  if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode) {
    return true;
  }
  return pynative::PyNativeExecutor::GetInstance()->grad_executor()->jit()->GetJitGradGraph(resource);
}

bool RewriterAfterOptAPassAfterJitBprop(const ResourcePtr &resource) {
  // This function is only used to convert unsupported syntax into PyExecute nodes through Fallback,
  // when the forward graph is decorated with 'jit', and is derivative in pynative mode.
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->not_convert_jit()) {
    context->set_not_convert_jit(false);
    MS_EXCEPTION_IF_NULL(resource);
    FuncGraphPtr func_graph = resource->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    (void)mindspore::opt::RewriterAfterOptA(func_graph, resource);
    UpdateArgsSpec(func_graph, resource);
  }
  return true;
}

bool EliminateSpecialOpNode(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (resource->manager() == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "PynativeElimOpt error, manager is null.";
  }
  if (resource->func_graph() == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "PynativeElimOpt error, graph is null.";
  }
  return EliminateSpecialOpOptPass(resource);
}

bool SupportInlinePartial(const AnfNodePtr &input0) {
  // inline partial
  if (IsPrimitiveCNode(input0, prim::kPrimTupleGetItem)) {
    auto tuple_get_node = input0->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(tuple_get_node);
    auto get_from_node = tuple_get_node->input(kFirstInput);
    auto idx = common::AnfAlgo::GetTupleGetItemOutIndex(tuple_get_node);
    MS_EXCEPTION_IF_NULL(get_from_node);
    // tuple get item from a call subgraph output
    if (get_from_node->isa<CNode>() && IsValueNode<FuncGraph>(get_from_node->cast<CNodePtr>()->input(0))) {
      auto call_graph = GetValueNode<FuncGraphPtr>(get_from_node->cast<CNodePtr>()->input(0));
      MS_EXCEPTION_IF_NULL(call_graph);
      auto graph_out = call_graph->output();
      MS_EXCEPTION_IF_NULL(graph_out);
      size_t tuple_input_num = common::AnfAlgo::GetInputTensorNum(graph_out);
      // the partial must be the last output
      if (graph_out->isa<CNode>() && tuple_input_num == idx + 1) {
        int partial_cnt = 0;
        for (size_t i = 0; i < tuple_input_num; i++) {
          auto input = graph_out->cast<CNodePtr>()->input(i + 1);
          if (IsPrimitiveCNode(input, prim::kPrimPartial)) {
            partial_cnt++;
          }
        }
        auto partial = graph_out->cast<CNodePtr>()->input(idx + 1);
        MS_EXCEPTION_IF_NULL(partial);
        // we only support one partial func at the last return value now
        if (partial_cnt != 1 || !IsPrimitiveCNode(partial, prim::kPrimPartial)) {
          if (partial_cnt != 0) {
            MS_LOG(INFO) << "Partial func cnt: " << partial_cnt
                         << ", last return value: " << partial->fullname_with_scope();
          }
          return false;
        }
        auto partial_inputs = partial->cast<CNodePtr>()->inputs();
        // the input of partial can't be FuncGraph/Partial
        bool has_illegal_input =
          std::any_of(partial_inputs.begin() + kSecondInput, partial_inputs.end(), [](const AnfNodePtr &partial_input) {
            return IsValueNode<FuncGraph>(partial_input) || IsPrimitiveCNode(partial_input, prim::kPrimPartial);
          });
        return !has_illegal_input;
      }
    }
  }
  return false;
}

bool HasAbstractFunction(const AbstractBasePtr &abs) {
  if (abs->isa<abstract::AbstractSequence>() && !abs->isa<abstract::AbstractSparseTensor>()) {
    auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    if (abs_seq->dynamic_len()) {
      return HasAbstractFunction(abs_seq->dynamic_len_element_abs());
    }
    return std::any_of(abs_seq->elements().cbegin(), abs_seq->elements().cend(), HasAbstractFunction);
  }
  // if abs it not AbstractSequence.
  return abs->isa<abstract::AbstractFunction>();
}

bool IsCellReuse(const AnfNodePtr &input) {
  if (IsValueNode<FuncGraph>(input)) {
    auto fg = GetValueNode<FuncGraphPtr>(input);
    MS_EXCEPTION_IF_NULL(fg);
    auto debug_str = fg->ToString();
    if (fg->has_flag(FUNC_GRAPH_FLAG_CELL_REUSE)) {
      return true;
    }
  }
  return false;
}

bool AcceptableReturnValue(const CNodePtr &cnode, const AnfNodePtr &input0) {
  if (IsCellReuse(input0)) {
    return true;
  }
  auto func_graphs = abstract::GetFuncGraphsFromCallNode(cnode);
  auto graph_has_function_output = [](const FuncGraphPtr &fg) { return HasAbstractFunction(fg->output()->abstract()); };
  if (std::all_of(func_graphs.cbegin(), func_graphs.cend(), std::not_fn(graph_has_function_output))) {
    return true;
  }
  return false;
}

bool HasIncorporateCallNode(const CNodePtr &cnode) {
  if (!IsValueNode<Primitive>(cnode->input(0))) {  // If cnode is a call node.
    auto input0 = cnode->input(0);
    if (IsPrimitiveCNode(input0, prim::kPrimSwitch) || IsPrimitiveCNode(input0, prim::kPrimSwitchLayer) ||
        IsValueNode<FuncGraph>(input0)) {
      if (IsCellReuse(input0) && IsEnableRefMode()) {
        MS_LOG(INFO) << "Use cell reuse when enable ge mode: " << cnode->DebugString();
        return true;
      }
      if (AcceptableReturnValue(cnode, input0)) {
        return false;
      }
    }
    if (SupportInlinePartial(input0)) {
      return false;
    }
    MS_LOG(INFO) << "Call has indirect call: " << cnode->DebugString();
    return true;
  }
  return false;
}

bool HasIncorporateCall(const std::vector<AnfNodePtr> &all_nodes) {
  for (const auto &node : all_nodes) {
    if (node == nullptr || !node->isa<CNode>()) {
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
        MS_LOG(INTERNAL_EXCEPTION) << "SwitchLayer input2 should be make_tuple, but got: " << make_tuple->DebugString();
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
    if (HasIncorporateCallNode(cnode)) {
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
            MS_LOG(WARNING) << "On the Ascend platform, if you read-only access to the parameter, "
                            << "you can take the value of the parameter, so that the system can do more optimization. "
                            << "For example, change 'return param' to 'return param.value()'\n"
                            << "Please check your code:" << trace::GetDebugInfoStr(user_node->debug_info());
          }
          return true;
        }
      }
    }
  }
  return false;
}

bool SetModeForControlFlow(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &all_nodes, bool pynative_mode,
                           compile::Backend *backend_ptr) {
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
  // GRAPH | Closure\ENV\While scenario : KernelByKernel path in MindRT.
  auto graphs = func_graph->func_graphs_used_total();
  (void)graphs.insert(func_graph);
  bool exist_control_flow = ExistControlFlow(func_graph);
  bool exist_func = exist_control_flow && HasIncorporateCall(all_nodes);
  if (exist_func) {
    if (!pynative_mode) {
      MS_LOG(INFO) << "Run graph mode with sub graph sink because graph exist control flow and incorporate call.";
      set_ctx(true, false, false);
    } else {
      MS_LOG(INFO) << "Run graph mode with kernel by kernel because graph exist control flow and incorporate call.";
      set_ctx(false, false, false);
    }
    return false;
  }
  bool exist_while =
    std::any_of(graphs.cbegin(), graphs.cend(), [](const FuncGraphPtr &fg) { return fg->recursive(); });
  MS_LOG(INFO) << func_graph->ToString() << " exist_while: " << exist_while;
  if (exist_while || ExistSwitchRef(func_graph, all_nodes)) {
    if (!pynative_mode) {
      MS_LOG(INFO) << "Run graph mode with sub graph sink because graph exist while or switch ref.";
      set_ctx(true, false, false);
    } else {
      MS_LOG(INFO) << "Run graph mode with kernel by kernel because graph exist while or switch ref.";
      set_ctx(false, false, false);
    }
    return false;
  }
  // Multiple device targets scenario.
  if (func_graph->exist_multi_target()) {
    // Heterogeneous scenario + ControlFlow : KernelByKernel path in MindRT.
    if (exist_control_flow && pynative_mode) {
      MS_LOG(INFO) << "Run graph mode with kernel by kernel because graph exist multi device target and control flow.";
      set_ctx(false, false, false);
      return false;
    }
    // GRAPH | Heterogeneous scenario : No control flow, subgraph sink path in MindRT.
    MS_LOG(INFO) << "Run graph mode with subgraph sink because graph exist multi device target.";
    set_ctx(true, false, false);
    return false;
  }
  return true;
}

void SetRunMode(const FuncGraphPtr &func_graph, compile::Backend *backend_ptr, std::string *kbk_reason) {
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
  graphkernel::GraphKernelFlags::SaveJitConfig(PhaseManager::GetInstance().jit_config());

  const bool pynative_mode = context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode;
  const auto &device_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (pynative_mode && device_target != kAscendDevice) {
    return;
  }
  const auto &all_nodes = TopoSort(func_graph->return_node(), SuccDeeperSimple, AlwaysInclude);
  // GPU/CPU no need set any context.
  if (!ExistTarget(all_nodes, kAscendDevice)) {
    return;
  }

  // GRAPH | Single Op : KernelByKernel path in MindRT.
  if (context_ptr->IsKByKExecutorMode()) {
    if (kbk_reason != nullptr) {
      *kbk_reason = "Run graph mode with kernel by kernel by configuration.";
      MS_LOG(INFO) << *kbk_reason;
    }
    set_ctx(false, false, false);
    return;
  }

  // GRAPH | Dynamic Shape : KernelByKernel path in MindRT.
  if (IsDynamicShapeGraph(func_graph) && (context_ptr->backend_policy() != "ge")) {
    if (kbk_reason != nullptr) {
      *kbk_reason =
        "Run graph mode with kernel by kernel because graph exist dynamic shape. Call "
        "'set_context(save_graphs=True)' to check graph irs.";
      MS_LOG(INFO) << *kbk_reason;
    }
    set_ctx(false, false, false);
    return;
  }

  // GRAPH | Dynamic Scalar : Dynamic scalar ops in graph.
  if (IsNeedBackoffGraph(func_graph) && !IsDynamicShapeGraph(func_graph)) {
    if (kbk_reason != nullptr) {
      *kbk_reason = "Run graph mode with kernel by kernel because graph exist dynamic scalar ops.";
      MS_LOG(INFO) << *kbk_reason;
    }
    set_ctx(false, false, false);
    return;
  }
  if (!SetModeForControlFlow(func_graph, all_nodes, pynative_mode, backend_ptr)) {
    return;
  }

#if defined(__linux__) && defined(WITH_BACKEND)
  if (ps::PSContext::instance()->cache_enable()) {
    MS_LOG(INFO) << "Run graph mode with subgraph sink because PS cache enable.";
    set_ctx(true, false, false);
    return;
  }
#endif

  // GRAPH | normal network and if/for/switch scenario etc : MultiGraph path in MindRT.
  MS_LOG(INFO) << "Run graph mode with multi graph sink.";
  set_ctx(true, true, !pynative_mode);
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
  // The root cause of KernelByKernel mode should be returned.
  std::string kbk_reason = "";
  if (context_ptr->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    SetRunMode(resource->func_graph(), resource->GetBackend().get(), &kbk_reason);
  } else {
    OriginSetRunMode(resource);
  }
  auto mode = context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE);
  auto is_task_sink = context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
  auto enable_hccl = context_ptr->get_param<bool>(MS_CTX_ENABLE_HCCL);
  bool using_cm = common::UseDynamicCluster() && common::GetEnv("MS_HCCL_CM_INIT") == "1";
  if (using_cm && common::GetEnv("GRAPH_OP_RUN") == "1") {
    MS_LOG(INTERNAL_EXCEPTION)
      << "You are setting 'MS_HCCL_CM_INIT' and 'GRAPH_OP_RUN' to 1 at the same time, which will cause confilct "
         "because 'MS_HCCL_CM_INIT' means running in sink mode, but 'GRAPH_OP_RUN' means running kernel by kernel. "
         "Please unset either of them and rerun the task.";
  }
  if (!is_task_sink && mode == kGraphMode && enable_hccl && (!common::UseHostCollective() || using_cm)) {
    MS_LOG(INTERNAL_EXCEPTION) << "Current execution mode is 'kernelbykernel', reason: " << kbk_reason
                               << ", but you're launching job using 'ranktable', which "
                                  "does not support 'kernelbykernel' mode.\n Please refer to link: "
                                  "https://www.mindspore.cn/tutorials/experts/en/master/parallel/startup_method.html "
                                  "and use 'Dynamic cluster'(suggested) or 'mpirun' to launch your job.";
  }
}

bool TaskEmitAction(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr func_graph = resource->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "TaskEmit args error";
  }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto mode = context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE);
  if (mode == kGraphMode && CheckGraphOutputConstOrParameter(func_graph)) {
    return true;
  }

  // In PyNative mode, multi target will generate in -1 shape in jit. But, jit in -1 shape will run as a call graph;
  // control flow not has flag kFlagJitCallGraph
  bool is_control_flow = !func_graph->func_graphs_used_total().empty();
  if (mode == kGraphMode || (mode == kPynativeMode && (func_graph->has_flag(kFlagJitCallGraph) || is_control_flow))) {
    func_graph->SetMultiTarget();
    if (func_graph->exist_multi_target() && DumpJsonParser::GetInstance().IsDumpEnabled()) {
      MS_LOG(WARNING) << "Multi device target is detected, CPU data is dumped in rank_0 directory";
    }
  }
  DisableMindRT(resource);

  SetRunMode(resource);
  auto bc_ptr = resource->GetBackend();
  MS_EXCEPTION_IF_NULL(bc_ptr);
  const auto &backend = context_ptr->backend_policy();
  // The graph compiling of mindRT.
  if ((backend == kMsConvert || backend == kGeVm) && context_ptr->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    TaskEmitActionForMindRT(resource);
    return true;
  }
  // The graph compiling of control sink.
  if (IsCtrlSink() && (backend == kMsConvert || backend == kGeVm)) {
    auto graph_id = bc_ptr->CompileGraph(NOT_NULL(func_graph));
    resource->SetResult(kOutput, graph_id);
    return true;
  }
  std::vector<PrimitivePtr> cut_list = compile::GetNonlinearOps();
  if (bc_ptr->name() == kMsConvert || bc_ptr->name() == kGeVm) {
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
    MS_LOG(INTERNAL_EXCEPTION) << "Execute args error";
  }
  std::string backend = MsContext::GetInstance()->backend_policy();
  // The graph running of mindRT.
  if ((backend == kMsConvert || backend == kGeVm) && MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    ExecuteActionForMindRT(resource);
    return true;
  }

  // The graph running of control sink.
  if (IsCtrlSink() && (backend == kMsConvert || backend == kGeVm)) {
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

#if defined(__linux__) && defined(WITH_BACKEND)
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
    MS_LOG(INTERNAL_EXCEPTION) << "Remove value node duplications error.";
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

bool AutoParallelAction(const ResourcePtr &resource) { return AutoParallelPass(resource); }

bool ValidateAction(const ResourcePtr &resource) {
  auto res = ValidatePass(resource);
#ifdef DEBUG
  FuncGraphLoopBreaker::Inst().Dump();
#endif
  return res;
}

bool SetMindIRGraphAction(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  resource->set_is_load(true);
  auto cell = py::cast<CellPtr>(resource->source_input());
  if (cell == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "The graph loaded from mindir is null.";
  }
  const std::string mindir_graph = "graph_load_from_mindir";
  auto obj = cell->GetAttr(mindir_graph);
  if (obj == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "The graph loaded from mindir is null. The cell has not attribute: " << mindir_graph;
  }
  auto fg = GetValue<FuncGraphPtr>(obj);
  if (fg == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "The graph loaded from mindir is null.";
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
  const auto &args_abs_list = resource->args_abs();
  (void)std::transform(args_abs_list.begin(), args_abs_list.end(), std::back_inserter(broaded_args),
                       [](const AbstractBasePtr &arg) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(arg);
                         if (arg->GetValueTrack() != kValueAny) {
                           return arg->Broaden();
                         }
                         return arg;
                       });

  abstract::AbstractBasePtrList func_args;
  const auto inputs = fg->get_inputs();
  (void)std::transform(inputs.begin(), inputs.end(), std::back_inserter(func_args),
                       [](const AnfNodePtr &arg) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(arg);
                         auto abs = arg->abstract();
                         MS_EXCEPTION_IF_NULL(abs);
                         return abs->Broaden();
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
    (void)InferMindir(resource->func_graph(), args_abs_list, true);
  }
  return true;
}

static std::vector<ActionItem> CommonPipeline(bool trace_flag) {
  std::vector<ActionItem> actions;

  if (!trace_flag) {
    // Parse the python ast to ANF graph
    (void)actions.emplace_back(std::make_pair(kParse, ParseAction));

    // Resolve the python func
    static auto boost_parse = common::GetEnv("MS_DEV_GREED_PARSE");
    if (boost_parse != "1") {
      (void)actions.emplace_back(std::make_pair(kSymbolResolve, SymbolResolveAction));
    }

    // Notice: Temporary solution, to be implemented using Python Rewriter in the future.
    // Set mixed Precision flag in subgraph.
    static bool enable_set_mixed_precision_flag = (common::GetEnv("MS_DEV_AMP_ENABLE_ALL_FG") == "1");
    if (enable_set_mixed_precision_flag) {
      (void)actions.emplace_back(std::make_pair(kSetMixedPrecisionFlag, SetMixedPrecisionAction));
    }

    auto parallel_context = parallel::ParallelContext::GetInstance();
    MS_EXCEPTION_IF_NULL(parallel_context);
    auto parallel_mode = parallel_context->parallel_mode();
    const bool is_parallel_mode =
      parallel_mode == parallel::kSemiAutoParallel || parallel_mode == parallel::kAutoParallel;
    static const auto combine_like_graphs = (common::GetEnv("COMBINE_LIKE_GRAPHS") == "1");
    if (!is_cluster_initialized && (!is_parallel_mode || combine_like_graphs)) {
      (void)actions.emplace_back(std::make_pair(kCombineLikeGraphs, CombineLikeGraphs));
    }

    // Make the reusable cell to be the reusable function graph
    (void)actions.emplace_back(std::make_pair(kGraphReusing, GraphReusingAction));
    (void)actions.emplace_back(std::make_pair(kMetaUnpackPrepare, MetaUnpackPrepareAction));
    // Pre-Lift the func graphs.
    (void)actions.emplace_back(std::make_pair(kPreCConv, PreCConvAction));
  }
  // Evaluate type and shape, and specialize.
  (void)actions.emplace_back(std::make_pair(kAbstractSpecialize, AbstractSpecializeAction));
  // PackFunc Expand.
  if (common::GetEnv("MS_DEV_DISABLE_TRACE") != "on") {
    (void)actions.emplace_back(std::make_pair(kPackExpand, PackExpandAction));
  }
  // Auto-monad for side-effects handling.
  (void)actions.emplace_back(std::make_pair(kAutoMonad, AutoMonadAction));
  // Do data structure simplifications and inline.
  (void)actions.emplace_back(std::make_pair(kInline, OptInlineAction));
  // Do prepositive auto parallel.
  (void)actions.emplace_back(std::make_pair(kPreAutoParallel, AutoParallelAction));
  // Do PipelineSplit action.
  (void)actions.emplace_back(std::make_pair(kPipelineSplit, PipelineSplitAction));

  return actions;
}

std::vector<ActionItem> VmPipeline(const ResourcePtr &resource, bool trace_flag) {
  is_cluster_initialized = distributed::cluster::ClusterContext::instance()->initialized();
  std::vector<ActionItem> actions;
  // If enable compilation cache and the cache is read successfully, only do the backend actions.
  if (IsPhaseLoadFromMindIR(PhaseManager::GetInstance().phase())) {
    actions = MindIRPipeline();
  } else if (!resource->EnableCompileCache() || resource->func_graph() == nullptr) {
    actions = CommonPipeline();

    // Optimize
    (void)actions.emplace_back(std::make_pair(kOptimize, VmOptimizeAction));

    (void)actions.emplace_back(std::make_pair(kAutoMonadReorder, OrderEnforceAction));

    // Eliminate forward cnode for grad graph
    (void)actions.emplace_back(std::make_pair(kGetJitBpropGraph, GetJitBpropGraph));

    // Rewriter(dict convert pyexecute) after jit bprop.
    (void)actions.emplace_back(std::make_pair(kRewriterAfterJitBprop, RewriterAfterOptAPassAfterJitBprop));

    // Eliminate the virtual mirror node
    (void)actions.emplace_back(std::make_pair(kEliminateSpecialOpNode, EliminateSpecialOpNode));
    (void)actions.emplace_back(std::make_pair(kValidate, ValidateAction));

#if defined(__linux__) && defined(WITH_BACKEND)
    (void)actions.emplace_back(std::make_pair(kDistribtuedSplit, DistributedSplitAction));
    if (ps::PSContext::instance()->is_worker()) {
      if (distributed::cluster::ClusterContext::instance()->initialized()) {
        MS_LOG(INFO) << "This worker is initialized. No need to add worker action.";
      } else {
        std::string server_mode = ps::PSContext::instance()->server_mode();
      }
    }
#endif
  }

  auto is_precompile_only = MsContext::GetInstance()->get_param<bool>(MS_CTX_PRECOMPILE_ONLY);
  if (is_precompile_only) {
    MS_LOG(INFO) << "PrecompileOnly, stop run graph";
    return actions;
  }

  if (common::GetEnv(kSimulationLevel) == kSimulationLevelCompileGraph) {
    return actions;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
#ifndef WITH_BACKEND
  if (ms_context->backend_policy() != "ge") {
#endif
    // Compile the ANF graph
    (void)actions.emplace_back(std::make_pair(kTaskEmit, TaskEmitAction));

    // Execute the graph
    (void)actions.emplace_back(std::make_pair(kExecute, ExecuteAction));
#ifndef WITH_BACKEND
  }
#endif
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
  (void)actions.emplace_back(std::make_pair(kLoadMindir, SetMindIRGraphAction));
  (void)actions.emplace_back(std::make_pair(kModifyMindirGraph, ModifyGraphGeneratedByMindIR));
  (void)actions.emplace_back(std::make_pair(kInferMindir, InferMindIR));
  (void)actions.emplace_back(std::make_pair(kValidate, ValidateAction));
  return actions;
}
}  // namespace pipeline
}  // namespace mindspore
