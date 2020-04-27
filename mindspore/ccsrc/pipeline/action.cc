/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "pipeline/action.h"

#include <memory>
#include <utility>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>

#include "ir/func_graph_cloner.h"
#include "parallel/costmodel_context.h"
#include "pipeline/pass.h"
#include "pipeline/parse/parse_base.h"
#include "pipeline/parse/data_converter.h"
#include "pipeline/static_analysis/abstract_value.h"
#include "pipeline/static_analysis/static_analysis.h"
#include "pipeline/static_analysis/program_specialize.h"
#include "pipeline/resource.h"
#include "utils/context/ms_context.h"
#include "pipeline/remove_value_node_dup.h"
#include "optimizer/optimizer.h"
#include "vm/transform.h"

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

  FuncGraphPtr fg = parse::ConvertToFuncGraph(input);
  if (fg == nullptr) {
    MS_LOG(EXCEPTION) << "Parse error.";
  }
  res->set_func_graph(fg);

  FuncGraphManagerPtr manager = res->manager();
  if (manager == nullptr) {
    MS_LOG(EXCEPTION) << "Manager is nullptr.";
  }
  manager->AddFuncGraph(fg);
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
    FuncGraphPtrList func_graphs = {fg};
    ClonerPtr cloner = std::make_shared<Cloner>(func_graphs, false, false, true, std::make_shared<TraceCopy>(),
                                                std::make_shared<TraceCombileLikeGraphs>());
    cloner->Run();
    auto base_graph = cloner->cloned_func_graph()[fg];
    MS_LOG(DEBUG) << "Basegraph:" << base_graph->ToString();

    if (fg->paramter_obj_nodes().size() == 0 || graphs.size() <= 1) {
      continue;
    }
    for (auto &fv : fg->paramter_obj_nodes()) {
      TraceManager::DebugTrace(std::make_shared<TraceCombileLikeGraphs>(fv->debug_info()));
      auto param = base_graph->add_parameter();
      TraceManager::EndTrace();
      auto &node_users = res->manager()->node_users()[fv];
      for (auto &n : node_users) {
        auto repl_n = (*cloner->cloned_node())[n.first]->cast<CNodePtr>();
        repl_n->set_input(n.second, param);
      }
    }
    MS_LOG(DEBUG) << "Fg0 paramter_obj_nodes size :" << fg->paramter_obj_nodes().size();

    for (auto &g : graphs) {
      auto fvs = g->paramter_obj_nodes();
      std::vector<AnfNodePtr> new_node_inputs;
      new_node_inputs.push_back(NewValueNode(base_graph));
      for (auto &p : g->parameters()) {
        AnfNodePtr para_after_cast = parse::GetMixedPrecisionCastHelp(g, p);
        new_node_inputs.push_back(para_after_cast);
      }
      (void)new_node_inputs.insert(new_node_inputs.end(), fvs.begin(), fvs.end());
      AnfNodePtr out = g->NewCNode(new_node_inputs);
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
  if (res->func_graph() == nullptr) {
    MS_LOG(EXCEPTION) << "SymbolResolve error, graph is null";
  }
  FuncGraphPtr func_graph = res->func_graph();
  auto succ = parse::ResolveFuncGraph(func_graph, res);

  // Remove unused nodes in cnode order list.
  func_graph->EraseUnusedNodeInOrder();
  func_graph->ReleaseFullOrderToEffectOrder();
  for (auto fg : func_graph->func_graphs_used_total()) {
    MS_EXCEPTION_IF_NULL(fg);
    fg->EraseUnusedNodeInOrder();
    fg->ReleaseFullOrderToEffectOrder();
  }
  return succ;
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

  // suppose that there is not KeywordArgument for the top graph
  // get the hyper parameter
  for (const auto &param : func_graph->parameters()) {
    auto param_node = std::static_pointer_cast<Parameter>(param);
    if (param_node->has_default()) {
      AbstractBasePtr ptr =
        abstract::FromValue(parse::data_converter::PyDataToValue(param_node->default_param()), true);
      args_spec.push_back(ptr);
    }
  }
  // Analyze
  AnalysisResult result = AbstractAnalyze(res, func_graph, args_spec);
  // The top graph may be replaced by infer, update the top graph when the infer is done
  parse::Parser::UpdateTopFuncGraph(result.context->func_graph());

  // Specialize
  FuncGraphPtr new_fg = ProgramSpecialize(res, result.context->func_graph(), result.context);
  res->set_func_graph(new_fg);

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
      if (MsContext::GetInstance()->save_graphs_flag() && res->func_graph() != nullptr) {
        auto fg_name = "opt_pass_" + std::to_string(counter) + "_" + pass.first;
        auto func_graph = res->func_graph();
        MS_EXCEPTION_IF_NULL(func_graph);
        func_graph->DumpFuncGraph(fg_name);
        DumpIR(fg_name + ".ir", func_graph);
        MS_LOG(DEBUG) << "Dump " << fg_name << " func graph.";
      }
      counter++;
      MS_LOG(DEBUG) << "Pass " << pass.first << " end.";
    };
  }

  return true;
}

bool GeOptimizeAction(const ResourcePtr &res) { return OptimizeAction(res, kGePasses); }

bool VmOptimizeAction(const ResourcePtr &res) { return OptimizeAction(res, kVmPasses); }

bool TaskEmitAction(const ResourcePtr &res) {
  if (res->func_graph() == nullptr) {
    MS_LOG(EXCEPTION) << "TaskEmit args error";
  }
  FuncGraphPtr func_graph = res->func_graph();

  auto bc_ptr = res->results()[kBackend].cast<compile::BackendPtr>();
  std::vector<PrimitivePtr> cut_list = compile::nonlinear_ops;
  if (bc_ptr->name() == kMsConvert) {
    cut_list = compile::GetMsNonlinearOps();
  }
  std::shared_ptr<CompileGraphs> compile = std::make_shared<CompileGraphs>(bc_ptr, cut_list);
  res->results()[kOutput] = compile->CompileAndLink(func_graph);
  return true;
}

bool ExecuteAction(const ResourcePtr &res) {
  if (res->results().count(kOutput) == 0 || !res->results()[kOutput].is<compile::FinalVMPtr>()) {
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

// The parallel primitive related valuenode might be partitioned so that its value changes by device,
// that will result in a syncronization error due to different executing order.
// Here we temporarily avoid the problem by skipping valuenode merging used by parallel related primitive,
// the final solution will be proposed later as a parallel feature.
bool KeepValueNodeDuplication(const AnfNodePtr &value_node, const ResourcePtr &res) {
  auto &node_users = res->manager()->node_users();
  auto &users = node_users[value_node];
  auto used_by_keep_value_prim =
    std::any_of(users.begin(), users.end(), [](const std::pair<AnfNodePtr, int> &user) -> bool {
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
  auto value_nodes = manager->valuenodes()[func_graph];
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

bool ValidateAction(const ResourcePtr &res) { return ValidatePass(res); }

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

  return actions;
}

std::vector<ActionItem> GePipeline() {
  auto actions = CommonPipeline();
  // optimize
  actions.emplace_back(std::make_pair("optimize", GeOptimizeAction));
  actions.emplace_back(std::make_pair("remove_value_node_duplications", RemoveValueNodeDuplicationsAction));
  actions.emplace_back(std::make_pair("validate", ValidateAction));
  return actions;
}

std::vector<ActionItem> VmPipeline() {
  auto actions = CommonPipeline();

  // optimize
  actions.emplace_back(std::make_pair("optimize", VmOptimizeAction));

  actions.emplace_back(std::make_pair("validate", ValidateAction));

  // compile the ANF graph
  actions.emplace_back(std::make_pair("task_emit", TaskEmitAction));

  // to execute the graph
  actions.emplace_back(std::make_pair("execute", ExecuteAction));

  return actions;
}
}  // namespace pipeline
}  // namespace mindspore
