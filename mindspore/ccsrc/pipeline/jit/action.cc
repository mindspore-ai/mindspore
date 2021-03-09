/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "frontend/parallel/costmodel_context.h"
#include "frontend/parallel/context.h"
#include "pipeline/jit/pass.h"
#include "pipeline/jit/parse/parse_base.h"
#include "pipeline/jit/parse/data_converter.h"
#include "pipeline/jit/static_analysis/auto_monad.h"
#include "abstract/abstract_value.h"
#include "pipeline/jit/static_analysis/static_analysis.h"
#include "pipeline/jit/static_analysis/program_specialize.h"
#include "pipeline/jit/resource.h"
#include "utils/ms_context.h"
#include "pipeline/jit/remove_value_node_dup.h"
#include "frontend/optimizer/optimizer.h"
#include "vm/transform.h"
#include "parse/python_adapter.h"
#include "frontend/optimizer/py_pass_manager.h"
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
#include "ps/parameter_server.h"
#include "ps/scheduler.h"
#include "ps/worker.h"
#endif

namespace mindspore {
namespace pipeline {
using CompileGraphs = compile::CompileGraphs;
using abstract::AnalysisResult;
using mindspore::abstract::AnalysisContextPtr;

abstract::AnalysisResult AbstractAnalyze(const ResourcePtr &res, const FuncGraphPtr &func_graph,
                                         const abstract::AbstractBasePtrList &args_spec, bool clear) {
  MS_LOG(DEBUG) << "AbstractAnalyze start";
  auto engine = res->engine();
  MS_EXCEPTION_IF_NULL(engine);
  if (clear) {
    auto manager = res->manager();
    MS_EXCEPTION_IF_NULL(manager);
    engine->Clear();
    for (auto &node : manager->all_nodes()) {
      MS_EXCEPTION_IF_NULL(node);
      const AbstractBasePtr &prev_inferred = node->abstract();
      // Keep previous inferred value for CNode if is loaded from MindIR.
      if (node->isa<CNode>() && node->cast<CNodePtr>()->get_load_flag()) {
        continue;
      }
      // Keep previous inferred value for ValueNode if the inferred value is not AbstractFunction.
      if (!node->isa<ValueNode>() || (prev_inferred != nullptr && prev_inferred->isa<abstract::AbstractFunction>())) {
        node->set_abstract(nullptr);
        MS_LOG(DEBUG) << "Abstract of node " << node->ToString() << " is set to nullptr";
      }
    }
  }
  auto ret = engine->Run(func_graph, args_spec);
  MS_LOG(DEBUG) << "AbstractAnalyze end";
  return ret;
}

FuncGraphPtr ProgramSpecialize(const ResourcePtr &res, const FuncGraphPtr &func_graph,
                               const abstract::AnalysisContextPtr &context) {
  MS_LOG(DEBUG) << "ProgramSpecialize start";
  abstract::ProgramSpecializer spc(res->engine());
  FuncGraphPtr result = spc.Run(func_graph, context);
  auto manager = res->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->KeepRoots({result});
  MS_LOG(DEBUG) << "ProgramSpecialize end";
  return result;
}

FuncGraphPtr Renormalize(const ResourcePtr &res, const FuncGraphPtr &func_graph,
                         const abstract::AbstractBasePtrList &args_spec) {
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
    }
  }
  if (loaded_graph_num == 0) {
    return nullptr;
  }
  if (loaded_graph_num == 1) {
    return loaded_graph;
  }
  MS_LOG(EXCEPTION) << "The loaded sub graph currently should less than 2, but got " << loaded_graph_num;
}

void CheckRootInputShapeAndType(const ResourcePtr &res, const FuncGraphPtr &loaded_graph) {
  MS_EXCEPTION_IF_NULL(res);
  auto manager = res->manager();
  MS_EXCEPTION_IF_NULL(manager);
  FuncGraphPtr root_graph = *(manager->roots().begin());
  auto root_inputs = root_graph->get_inputs();
  auto loaded_inputs = loaded_graph->get_inputs();

  size_t root_inputs_num = root_inputs.size();
  size_t loaded_inputs_num = loaded_inputs.size();
  if (root_inputs_num != loaded_inputs_num) {
    MS_LOG(EXCEPTION) << "The inputs number " << root_inputs_num << " not equal to the inputs number of loaded graph "
                      << loaded_inputs_num;
  }
  for (size_t index = 0; index < root_inputs_num; index++) {
    auto root_input = root_inputs[index];
    auto loaded_input = loaded_inputs[index];

    auto root_shape = root_input->Shape() == nullptr ? nullptr : dyn_cast<abstract::Shape>(root_input->Shape());
    auto loaded_shape = loaded_input->Shape() == nullptr ? nullptr : dyn_cast<abstract::Shape>(loaded_input->Shape());
    auto root_type = root_input->Type() == nullptr ? nullptr : dyn_cast<Type>(root_input->Type());
    auto loaded_type = loaded_input->Type() == nullptr ? nullptr : dyn_cast<Type>(loaded_input->Type());
    MS_EXCEPTION_IF_NULL(root_shape);
    MS_EXCEPTION_IF_NULL(loaded_shape);
    MS_EXCEPTION_IF_NULL(root_type);
    MS_EXCEPTION_IF_NULL(loaded_type);

    if (root_shape->shape() != loaded_shape->shape()) {
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
  if (!res->input()) {
    MS_LOG(EXCEPTION) << "Parse error";
  }

  py::object input = res->input();
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
  auto &obj_map = parse::data_converter::GetObjGraphs();

  for (auto it : obj_map) {
    auto &graphs = it.second;
    MS_LOG(DEBUG) << "Start combine like graph:" << it.first << ", size:" << graphs.size();
    auto fg = graphs[0];
    FuncGraphVector func_graphs = {fg};
    ClonerPtr cloner = std::make_shared<Cloner>(func_graphs, false, false, true, std::make_shared<TraceCopy>(),
                                                std::make_shared<TraceCombileLikeGraphs>());
    cloner->Run();
    auto base_graph = cloner->cloned_func_graph()[fg];
    MS_LOG(DEBUG) << "Basegraph:" << base_graph->ToString();

    if (fg->used_global_parameters().empty() || graphs.size() <= 1) {
      continue;
    }
    auto &cloned_nodes = *cloner->cloned_node();
    for (auto &fv : fg->used_global_parameters()) {
      TraceGuard guard(std::make_shared<TraceCombileLikeGraphs>(fv->debug_info()));
      auto param = base_graph->add_parameter();
      auto &node_users = res->manager()->node_users()[fv];
      for (auto &n : node_users) {
        // If the user is not in this graph, no need to change.
        auto cloned = cloned_nodes[n.first];
        if (cloned == nullptr) {
          continue;
        }
        auto repl_n = cloned->cast<CNodePtr>();
        repl_n->set_input(n.second, param);
      }
    }
    MS_LOG(DEBUG) << "Fg0 used_global_parameters size :" << fg->used_global_parameters().size();

    for (auto &g : graphs) {
      auto &fvs = g->used_global_parameters();
      std::vector<AnfNodePtr> new_node_inputs;
      new_node_inputs.push_back(NewValueNode(base_graph));
      for (auto &p : g->parameters()) {
        AnfNodePtr para_after_cast = parse::GetMixedPrecisionCastHelp(g, p);
        new_node_inputs.push_back(para_after_cast);
      }
      (void)new_node_inputs.insert(new_node_inputs.end(), fvs.begin(), fvs.end());
      AnfNodePtr out = g->NewCNodeBefore(g->get_return(), new_node_inputs);
      g->set_output(out);
      MS_LOG(DEBUG) << "Combine graph newout:" << out->DebugString(4);
    }
    MS_LOG(DEBUG) << "End combine graph:" << it.first;
  }
  return true;
}

bool SymbolResolveAction(const ResourcePtr &res) {
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

bool InferenceOptPrepareAction(const ResourcePtr &res) {
  if (res->manager() == nullptr) {
    MS_LOG(EXCEPTION) << "InferenceOptPrepare error, manager is null.";
  }
  if (res->func_graph() == nullptr) {
    MS_LOG(EXCEPTION) << "InferenceOptPrepare error, graph is null.";
  }
  return InferenceOptPreparePass(res);
}

bool AbstractSpecializeAction(const ResourcePtr &res) {
  if (res->func_graph() == nullptr) {
    MS_LOG(EXCEPTION) << "AbstractSpecialize error";
  }
  FuncGraphPtr func_graph = res->func_graph();
  abstract::AbstractBasePtrList args_spec = res->args_spec();
  auto context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel::ParallelContext::GetInstance());
  context->ParallelParameterContextInitShape(func_graph);

  // get original loaded graph to check inputs later
  auto loaded_graph_ptr = GetLoadedGraph(res);
  // suppose that there is not KeywordArgument for the top graph
  // get the hyper parameter
  for (const auto &param : func_graph->parameters()) {
    auto param_node = std::static_pointer_cast<Parameter>(param);
    if (param_node->has_default()) {
      auto value = param_node->default_param();
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
  // check input after abstract when there is a loaded graph
  if (loaded_graph_ptr != nullptr) {
    CheckRootInputShapeAndType(res, loaded_graph_ptr);
  }
  MS_LOG(DEBUG) << "End graph: " << new_fg->ToString() << ", return: " << new_fg->get_return()->DebugString(true);
  return true;
}

bool OptimizeAction(const ResourcePtr &res, const std::vector<PassItem> &passes) {
  size_t counter = 0;
  for (auto &pass : passes) {
    WITH(MsProfile::GetProfile()->Step(pass.first))[&pass, &res, &counter]() {
      MS_LOG(DEBUG) << "Pass " << pass.first << " start ...";
      auto result = pass.second(res);
      if (!result) {
        MS_LOG(EXCEPTION) << "Pass running to end, failed in pass:" << pass.first;
      }
      if (MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG) && res->func_graph() != nullptr) {
        auto fg_name = "opt_pass_" + std::to_string(counter) + "_" + pass.first;
        auto func_graph = res->func_graph();
        MS_EXCEPTION_IF_NULL(func_graph);
        func_graph->DumpFuncGraph(fg_name);
        DumpIR(fg_name + ".ir", func_graph);
        ExportIR(fg_name + ".dat", "", func_graph);
        MS_LOG(DEBUG) << "Dump " << fg_name << " func graph.";
      }
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

bool VmOptimizeAction(const ResourcePtr &res) { return OptimizeAction(res, kVmPasses); }

bool PynativeOptimizeAction(const ResourcePtr &resource) {
  WITH(MsProfile::GetProfile())[&resource]() { (void)OptimizeAction(resource, kPynativePasses); };
#ifdef ENABLE_PROFILE
  MsProfile::Print();
  MsProfile::Reset();
#endif
  return true;
}

bool PynativeElimOpt(const ResourcePtr &res) {
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

bool TaskEmitAction(const ResourcePtr &res) {
  if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode &&
      CheckGraphOutputConstOrParameter(res->func_graph())) {
    return true;
  }
  if (res->func_graph() == nullptr) {
    MS_LOG(EXCEPTION) << "TaskEmit args error";
  }
  FuncGraphPtr func_graph = res->func_graph();
  auto bc_ptr = res->results()[kBackend].cast<compile::BackendPtr>();
  auto context_ptr = MsContext::GetInstance();
  std::string backend = MsContext::GetInstance()->backend_policy();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (func_graph->ContainMultiTarget()) {
    bc_ptr->set_is_multi_graph_sink(false);
    context_ptr->set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, false);
    context_ptr->set_param<bool>(MS_CTX_ENABLE_LOOP_SINK, false);
  } else if (context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode) {
    std::string device_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    if (device_target == kAscendDevice && backend != kMsVm) {
      bc_ptr->set_is_multi_graph_sink(true);
      context_ptr->set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, true);
    }
  }

  if (IsCtrlSink() && backend == kMsConvert) {
    res->results()[kOutput] = bc_ptr->CompileGraph(NOT_NULL(func_graph));
    return true;
  }
  std::vector<PrimitivePtr> cut_list = compile::nonlinear_ops;
  if (bc_ptr->name() == kMsConvert) {
    cut_list = compile::GetMsNonlinearOps();
  }
  std::shared_ptr<CompileGraphs> compile = std::make_shared<CompileGraphs>(bc_ptr, cut_list);
  res->results()[kOutput] = compile->CompileAndLink(func_graph);
  return true;
}

bool ExecuteAction(const ResourcePtr &res) {
  if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode &&
      CheckGraphOutputConstOrParameter(res->func_graph())) {
    return true;
  }
  if (res->results().count(kOutput) == 0) {
    MS_LOG(EXCEPTION) << "Execute args error";
  }
  std::string backend = MsContext::GetInstance()->backend_policy();
  if (IsCtrlSink() && backend == kMsConvert) {
    if (!res->results()[kOutput].is<GraphId>()) {
      MS_LOG(EXCEPTION) << "Execute args error";
    }
    auto graph_id = res->results()[kOutput].cast<GraphId>();
    std::shared_ptr<compile::Backend> bc_ptr = res->results()[kBackend].cast<std::shared_ptr<compile::Backend>>();
    compile::MsBackend *msbc_ptr = std::dynamic_pointer_cast<compile::MsBackend>(bc_ptr).get();
    MS_EXCEPTION_IF_NULL(msbc_ptr);
    compile::VmEvalFuncPtr run =
      std::make_shared<compile::VmEvalFunc>([msbc_ptr, graph_id](const VectorRef &args) -> BaseRef {
        MS_LOG(INFO) << "Execute args size " << args.size();
        auto outs = msbc_ptr->RunGraph(graph_id, args);
        MS_LOG(DEBUG) << "out size " << outs.size();
        return outs[0];
      });
    res->results()[kOutput] = run;
    return true;
  }

  if (!res->results()[kOutput].is<compile::FinalVMPtr>()) {
    MS_LOG(EXCEPTION) << "Execute args error";
  }
  compile::FinalVMPtr vm = res->results()[kOutput].cast<compile::FinalVMPtr>();
  if (vm == nullptr) {
    MS_LOG(INFO) << "Call GE to Run the func_graph instead of VM";
    return true;
  }
  compile::VmEvalFuncPtr run =
    std::make_shared<compile::VmEvalFunc>(std::bind(&compile::FinalVM::Eval, vm, std::placeholders::_1));
  res->results()[kOutput] = run;
  return true;
}

#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
bool StartPSWorkerAction(const ResourcePtr &res) {
  ps::Worker::GetInstance().Run();
  return true;
}

bool StartPSServerAction(const ResourcePtr &res) {
  FuncGraphPtr func_graph = res->func_graph();
  auto &ps = ps::ParameterServer::GetInstance();
  ps.Run(func_graph);
  return true;
}

bool StartPSSchedulerAction(const ResourcePtr &res) {
  ps::Scheduler::GetInstance().Run();
  return true;
}
#endif

// The parallel primitive related valuenode might be partitioned so that its value changes by device,
// that will result in a synchronization error due to different executing order.
// Here we temporarily avoid the problem by skipping valuenode merging used by parallel related primitive,
// the final solution will be proposed later as a parallel feature.
bool KeepValueNodeDuplication(const AnfNodePtr &value_node, const ResourcePtr &res) {
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
        // value_node is referenced by some parallel primitive
        return prim->HasAttr("keep_value_node_input");
      }
      return false;
    });
  return used_by_keep_value_prim;
}

bool RemoveValueNodeDuplicationsAction(const ResourcePtr &res) {
  if (res->func_graph() == nullptr) {
    MS_LOG(EXCEPTION) << "Remove value node duplications error.";
  }
  FuncGraphPtr func_graph = res->func_graph();
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
      MS_EXCEPTION_IF_NULL(res->func_graph());
      FuncGraphPtr func_graph = res->func_graph();
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
      MS_EXCEPTION_IF_NULL(res->func_graph());
      FuncGraphPtr func_graph = res->func_graph();
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
  actions.emplace_back(std::make_pair("parse", ParseAction));

  // Resolve the python func
  actions.emplace_back(std::make_pair("symbol_resolve", SymbolResolveAction));

  auto multi_graphs = parallel::CostModelContext::GetInstance()->is_multi_subgraphs();
  if (!multi_graphs) {
    actions.emplace_back(std::make_pair("combine_like_graphs", CombineLikeGraphs));
  }

  actions.emplace_back(std::make_pair("inference_opt_prepare", InferenceOptPrepareAction));
  // Evaluate type and shape, and specialize
  actions.emplace_back(std::make_pair("abstract_specialize", AbstractSpecializeAction));
  // Auto-monad for side-effects handling.
  actions.emplace_back(std::make_pair("auto_monad", AutoMonadAction));
  // Do data structure simplifications and inline
  actions.emplace_back(std::make_pair("inline", OptInlineAction));
  // Add pre-ad, post-inline python pass stub
  actions.emplace_back(std::make_pair("py_pre_ad", PreAdActionPyStub));
  // Do PipelineSplit
  actions.emplace_back(std::make_pair("pipeline_split", PipelineSplitAction));

  return actions;
}

std::vector<ActionItem> GePipeline() {
  auto actions = CommonPipeline();
  // optimize
  actions.emplace_back(std::make_pair("optimize", GeOptimizeAction));
  // Add opt-stage python pass stub
  actions.emplace_back(std::make_pair("py_opt", OptActionGePyStub));
  actions.emplace_back(std::make_pair("remove_value_node_duplications", RemoveValueNodeDuplicationsAction));
  actions.emplace_back(std::make_pair("validate", ValidateAction));
  return actions;
}

std::vector<ActionItem> VmPipeline() {
  auto actions = CommonPipeline();

  // optimize
  actions.emplace_back(std::make_pair("optimize", VmOptimizeAction));

  // Add opt-stage python pass stub
  actions.emplace_back(std::make_pair("py_opt", OptActionVmPyStub));

  actions.emplace_back(std::make_pair("validate", ValidateAction));
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
  if (ps::PSContext::instance()->is_worker()) {
    actions.emplace_back(std::make_pair("worker", StartPSWorkerAction));
  }
#endif
  // compile the ANF graph
  actions.emplace_back(std::make_pair("task_emit", TaskEmitAction));

  // to execute the graph
  actions.emplace_back(std::make_pair("execute", ExecuteAction));

  return actions;
}

#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
std::vector<ActionItem> PServerPipeline() {
  auto actions = CommonPipeline();
  actions.emplace_back(std::make_pair("optimize", VmOptimizeAction));
  actions.emplace_back(std::make_pair("validate", ValidateAction));
  actions.emplace_back(std::make_pair("pserver", StartPSServerAction));
  return actions;
}

std::vector<ActionItem> PSchedulerPipeline() {
  std::vector<ActionItem> actions;
  actions.emplace_back(std::make_pair("scheduler", StartPSSchedulerAction));
  return actions;
}
#endif
}  // namespace pipeline
}  // namespace mindspore
