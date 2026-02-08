/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/ad/pynative_jit_grad.h"

#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include <set>
#include <map>

#include "include/pynative/utils/pynative_utils.h"
#include "include/utils/operator/primitive_utils.h"
#include "include/utils/pynative/common_utils.h"
#include "frontend/jit/ps/pass.h"
#include "ir/func_graph_cloner.h"
#include "primitive/sequence_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/structure_ops.h"
#include "primitive/other_ops.h"
#include "frontend/jit/ps/pipeline.h"
#include "frontend/jit/ps/parse/data_converter.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_u.h"
#include "primitive/auto_generate/gen_ops_primitive_v.h"
#include "include/utils/compile_cache_context.h"
#include "frontend/jit/ps/compile_cache_manager.h"
#include "utils/config_manager.h"
#include "include/frontend/jit/ps/executor/graph_executor_py.h"
#include "include/frontend/jit/ps/executor/jit_executor_py.h"
#include "include/frontend/jit/ps/action_interface.h"
#include "include/frontend/jit/ps/pipeline_interface.h"
#include "include/frontend/optimizer/fallback_rewriter_opt.h"

namespace mindspore {
namespace ad {
mindspore::HashMap<std::string, std::pair<FuncGraphPtr, FuncGraphPtr>> pass_grad_graph_;
mindspore::HashMap<std::string, pipeline::ResourcePtr> jit_forward_resource;
std::set<std::string> check_invalid_dout_bprop_graph;
mindspore::HashMap<std::string, FuncGraphPtr> origin_grad_graph_;
mindspore::HashMap<std::string, mindspore::HashMap<size_t, FuncGraphPtr>> filtered_grad_graph;

std::pair<FuncGraphPtr, FuncGraphPtr> GetGradAndForwardGraph(const std::string &key) {
  auto iter = pass_grad_graph_.find(key);
  if (iter == pass_grad_graph_.end()) {
    return std::make_pair(nullptr, nullptr);
  }
  return iter->second;
}

void StoreOriginGradGraph(const std::string &key, const FuncGraphPtr &fg) {
  auto iter = origin_grad_graph_.find(key);
  if (iter != origin_grad_graph_.end()) {
    MS_LOG(EXCEPTION) << "Key " << key << " has already set origin graph.";
  }
  origin_grad_graph_[key] = fg;
}

FuncGraphPtr GetOriginGradGraph(const std::string &key) {
  auto iter = origin_grad_graph_.find(key);
  if (iter == origin_grad_graph_.end()) {
    MS_LOG(EXCEPTION) << "Key " << key << " can not find origin graph.";
  }
  return iter->second;
}

bool HasOriginGradGraph(const std::string &key) {
  auto iter = origin_grad_graph_.find(key);
  return iter != origin_grad_graph_.end();
}

size_t StoreFilteredGradGraph(const std::string &cache_key, size_t hash_key, const FuncGraphPtr &fg) {
  auto cache_key_iter = filtered_grad_graph.find(cache_key);
  if (cache_key_iter == filtered_grad_graph.end()) {
    mindspore::HashMap<size_t, FuncGraphPtr> new_filtered_map = {
      {hash_key, fg},
    };
    filtered_grad_graph[cache_key] = new_filtered_map;
    return 1;
  }
  auto &cur_filtered_map = cache_key_iter->second;
  auto iter = cur_filtered_map.find(hash_key);
  if (iter != cur_filtered_map.end()) {
    MS_LOG(EXCEPTION) << "Hash key " << hash_key << " has already set filtered grad graph.";
  }
  cur_filtered_map[hash_key] = fg;
  return cur_filtered_map.size();
}

FuncGraphPtr GetFilteredGradGraph(const std::string &cache_key, size_t hash_key) {
  auto cache_key_iter = filtered_grad_graph.find(cache_key);
  if (cache_key_iter == filtered_grad_graph.end()) {
    MS_LOG(INFO) << "Can not find cache key " << cache_key << " in filtered_grad_graph";
    return nullptr;
  }
  const auto &cur_filter_map = cache_key_iter->second;
  auto iter = cur_filter_map.find(hash_key);
  if (iter == cur_filter_map.end()) {
    MS_LOG(INFO) << "Can not find filtered grad graph by hash Key " << hash_key;
    return nullptr;
  }
  return iter->second;
}

namespace {
using VmEvalPtr = std::shared_ptr<std::function<BaseRef(const VectorRef &)>>;
static const std::vector<PrimitivePtr> UNREUSED_PRIM_LIST = {prim::kPrimStopGradient,   prim::kPrimUpdateState,
                                                             prim::kPrimMirror,         prim::kPrimVirtualDiv,
                                                             prim::kPrimMutable,        prim::kPrimInsertGradientOf,
                                                             prim::kPrimHookBackward,   prim::kPrimCellBackwardHook,
                                                             prim::kPrimPrintShapeType, prim::kPrimLoad};

// Optimizes the forward function graph.
FuncGraphPtr OptimizeForwardGraph(const FuncGraphPtr &bprop_func_graph, bool need_renormalize = false) {
  auto resource = std::make_shared<pipeline::Resource>();
  resource->set_func_graph(bprop_func_graph);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(bprop_func_graph);
  if (need_renormalize) {
    // Renormalize, infer shape and set abstract for all nodes in graph
    abstract::AbstractBasePtrList args_abs;
    const auto &parameters = bprop_func_graph->parameters();
    (void)std::transform(parameters.begin(), parameters.end(), std::back_inserter(args_abs),
                         [](const AnfNodePtr &p) -> AbstractBasePtr { return p->abstract(); });
    MS_LOG(INFO) << "Start renormalizing for graph: " << bprop_func_graph->ToString();
    FuncGraphPtr new_fg = pipeline::Renormalize(resource, bprop_func_graph, args_abs);
    MS_EXCEPTION_IF_NULL(new_fg);
    MS_LOG(INFO) << "Finish renormalizing for graph: " << bprop_func_graph->ToString();
    resource->set_func_graph(new_fg);
    resource->set_args_abs(args_abs);
    manager->AddFuncGraph(new_fg);
  }
  (void)mindspore::opt::RewriterAfterOptA(resource->func_graph(), resource);
  (void)OptAfterJitGradPass(resource);
  return resource->func_graph();
}

// Optimizes the bprop function graph using certain passes
FuncGraphPtr OptimizeBpropGraph(const FuncGraphPtr &bprop_func_graph, const pynative::GradParamPtr &grad_param) {
  pipeline::ResourcePtr resource = std::make_shared<pipeline::Resource>();
  resource->set_func_graph(bprop_func_graph);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(bprop_func_graph);
  auto after_opt_bg = pipeline::JitBpropGraphPass(resource, true);
  auto is_dynamic_shape_control_flow = grad_param->is_jit_graph && grad_param->is_control_flow;
  if (is_dynamic_shape_control_flow) {
    for (const auto &g : manager->func_graphs()) {
      g->set_flag(kFlagJitCallGraph, true);
    }
  }
  return after_opt_bg;
}

void ClearFuncGraphCNodeAbstract(const FuncGraphPtr &func_graph) {
  std::vector<AnfNodePtr> nodes = TopoSort(func_graph->get_return(), SuccDeeperSimple);
  for (const auto &node : nodes) {
    if (node == nullptr || node->isa<Parameter>() || node->isa<mindspore::ValueNode>()) {
      continue;
    }
    const AbstractBasePtr &prev_inferred = node->abstract();
    // Keep previous inferred value for parameter and ValueNode if the inferred value is not AbstractFunction.
    if (prev_inferred != nullptr && prev_inferred->isa<abstract::AbstractFunction>()) {
      continue;
    }
    node->set_abstract(nullptr);
    MS_LOG(DEBUG) << "Abstract of node " << node->DebugString() << " is set to nullptr";
  }
}

void PlantFuncGradBpropGraphDout(const FuncGraphPtr &graph, size_t dout_index,
                                 const abstract::AbstractBasePtr &out_abstract) {
  MS_EXCEPTION_IF_NULL(graph);
  if (const size_t param_size = graph->parameters().size(); param_size <= dout_index) {
    MS_LOG(EXCEPTION) << "Invalid dout index for bprop_func_graph: " << graph->ToString()
                      << " , total param size: " << param_size << " , dout_index: " << dout_index;
  }
  // Plant dout tuple or dict
  // Parameters for bprop graph: {original_inputs, dout}
  if (out_abstract->isa<abstract::AbstractSequence>()) {
    pynative::CommonUtils::ProcessTupleParam(graph, dout_index);
  } else if (out_abstract->isa<abstract::AbstractDictionary>()) {
    pynative::CommonUtils::ProcessDictParam(graph, dout_index);
  }
}

bool IsUnSupportPrim(const AnfNodePtr &node) {
  // Check if a cnode
  auto cnode = dyn_cast_ptr<CNode>(node);
  if (cnode == nullptr || cnode->size() == 0) {
    return true;
  }
  // Check if a prim cnode
  const auto &input = cnode->input(0);
  MS_EXCEPTION_IF_NULL(input);
  if (!GetValuePtr<Primitive>(input)) {
    return true;
  }
  // Filter unsupported prim
  return std::any_of(UNREUSED_PRIM_LIST.begin(), UNREUSED_PRIM_LIST.end(),
                     [&node](const auto &primitive) { return IsPrimitiveCNode(node, primitive); });
}

bool JitOutputHasDict(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractDictionary>()) {
    return true;
  }
  if (abs->isa<abstract::AbstractSequence>()) {
    const auto &abs_sequence = abs->cast<abstract::AbstractSequencePtr>();
    return std::any_of(abs_sequence->elements().begin(), abs_sequence->elements().end(),
                       [](const abstract::AbstractBasePtr &item) { return JitOutputHasDict(item); });
  }
  return false;
}

BaseRef GetGraphResult(const FuncGraphPtr &fg, const VectorRef &arg_list, bool cache_hit,
                       const std::string &cache_key) {
  pipeline::ResourcePtr resource;
  const auto &it = jit_forward_resource.find(cache_key);
  constexpr auto need_repeat_task_emit_key = "need_repeat_task_emit";
  bool need_repeat_task_emit = fg->has_flag(need_repeat_task_emit_key);
  if (it == jit_forward_resource.end() || need_repeat_task_emit) {
    if (need_repeat_task_emit) {
      fg->erase_flag(need_repeat_task_emit_key);
    } else if (cache_hit) {
      MS_LOG(WARNING) << "Can not find cached resource for func graph: " << fg->ToString();
    }
    resource = std::make_shared<pipeline::Resource>();
    resource->set_func_graph(fg);
    auto manager = resource->manager();
    manager->AddFuncGraph(resource->func_graph(), true);
    pipeline::JitCompilingScope jit_compiling_scope;
    (void)TaskEmitAction(resource);
    (void)ExecuteAction(resource);
    jit_forward_resource[cache_key] = resource;
  } else {
    resource = it->second;
  }
  pipeline::JitRunningScope jit_running_scope;
  VectorRef outputs;
  if (common::AnfAlgo::IsGraphOutputValueNodeOrParameter(fg->output(), arg_list, &outputs)) {
    if (outputs.empty()) {
      return VectorRef();
    } else {
      return outputs[0];
    }
  }
  VmEvalPtr run = resource->GetResult(pipeline::kOutput).cast<VmEvalPtr>();
  auto result = (*run)(arg_list);
  MS_LOG(INFO) << "Finish running funcgraph: " << fg->ToString() << " , result: " << result.ToString();
  return result;
}

AnfNodePtrList ProcessParam(const FuncGraphPtr &source_fg, const abstract::AbstractBasePtrList &input_abs,
                            const std::vector<ValuePtr> &input_values) {
  MS_EXCEPTION_IF_NULL(source_fg);
  AnfNodePtrList param_list;
  if (input_abs.size() != input_values.size()) {
    MS_LOG(EXCEPTION) << "Got unmatched input abstract and value.";
  }
  for (size_t index = 0; index < input_abs.size(); ++index) {
    auto param = source_fg->add_parameter();
    param->set_abstract(input_abs[index]);
    (void)param_list.emplace_back(param);
    const auto &input_value = input_values[index];
    MS_EXCEPTION_IF_NULL(input_value);
    if (!input_value->isa<tensor::Tensor>()) {
      continue;
    }
    const auto &tensor = input_value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    if (!tensor->is_parameter()) {
      continue;
    }
    const auto &param_info = tensor->param_info();
    if (param_info) {
      const auto &parameter = param_info->parameter();
      if (parameter && parameter->has_default()) {
        param->set_default_param(parameter->default_param_raw());
      }
    }
  }
  return param_list;
}

// Helper function to handle forward result
py::object HandleForwardResult(const BaseRef &forward_result, const FuncGraphPtr &forward_fg,
                               const AbstractBasePtr &origin_forward_output_abs,
                               const pynative::GradParamPtr &grad_param, bool need_reuse_forward_node) {
  MS_EXCEPTION_IF_NULL(forward_result);
  MS_EXCEPTION_IF_NULL(forward_fg);
  if (!need_reuse_forward_node) {
    return pipeline::BaseRefToPyDataWithUserData(forward_result, origin_forward_output_abs);
  }
  grad_param->added_args.clear();
  if (utils::isa<VectorRef>(forward_result)) {
    MS_LOG(INFO) << "Run forward graph: " << forward_fg->ToString() << " in sync pipeline mode.";
    auto vector_result = utils::cast<VectorRef>(forward_result);
    auto result = vector_result[kIndex0];
    VectorRef add_args(vector_result.begin() + 1, vector_result.end());
    grad_param->added_args = add_args;
    return pipeline::BaseRefToPyDataWithUserData(result, origin_forward_output_abs);
  } else {
    MS_LOG(INFO) << "Run forward graph: " << forward_fg->ToString() << " in async pipeline mode.";
    const auto &output = forward_fg->output();
    MS_EXCEPTION_IF_NULL(output);
    const auto &output_abs = output->abstract();
    MS_EXCEPTION_IF_NULL(output_abs);
    auto py_forward_result = pipeline::BaseRefToPyDataWithUserData(forward_result, output_abs);
    py::tuple ret_tuple = py::cast<py::tuple>(py_forward_result);
    if (!py::isinstance<py::tuple>(ret_tuple) || !ret_tuple.size()) {
      MS_LOG(EXCEPTION) << "Forward output is not valid for fg: " << forward_fg->ToString()
                        << " , output: " << py::str(py_forward_result);
    }
    std::transform(
      ret_tuple.begin() + 1, ret_tuple.end(), std::back_inserter(grad_param->added_args),
      [](const auto &element) { return parse::data_converter::PyObjToValue(py::cast<py::object>(element)); });
    return ret_tuple[kIndex0];
  }
}

bool IsValidAbstract(const AbstractBasePtr &prim_abstract) {
  if (prim_abstract == nullptr) {
    return false;
  } else if (prim_abstract->isa<abstract::AbstractRefTensor>()) {
    const auto ref_abs = prim_abstract->cast_ptr<abstract::AbstractRefTensor>();
    MS_EXCEPTION_IF_NULL(ref_abs);
    return !ref_abs->is_view() && !ref_abs->is_inplace();
  } else if (prim_abstract->isa<abstract::AbstractTensor>()) {
    return true;
  } else if (prim_abstract->isa<abstract::AbstractSequence>()) {
    const auto &elements = prim_abstract->cast<abstract::AbstractSequencePtr>()->elements();
    return std::all_of(elements.begin(), elements.end(),
                       [](const AbstractBasePtr &element) { return IsValidAbstract(element); });
  }
  return false;
}

bool WithRecomputedScope(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto full_name_with_scope = node->fullname_with_scope();
  return full_name_with_scope.find(kAttrRecompute) == 0;
}

bool HasRecomputedScope(const CNodePtr &node) {
  // Exclude nodes without recompute scope
  if (!WithRecomputedScope(node)) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto recompute_attr = cnode->GetAttr(kAttrRecompute);
  return recompute_attr != nullptr && recompute_attr->isa<BoolImm>() && GetValue<bool>(recompute_attr);
}

bool IsViewInplaceAbs(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (!abs->isa<abstract::AbstractRefTensor>()) {
    return false;
  }
  const auto ref_abs = abs->cast_ptr<abstract::AbstractRefTensor>();
  MS_EXCEPTION_IF_NULL(ref_abs);
  return ref_abs->is_view() || ref_abs->is_inplace();
}
}  // namespace

std::pair<FuncGraphPtr, pipeline::CompileCacheManagerPtr> GetCompileCacheResource(const py::dict &weights,
                                                                                  const std::string &extension,
                                                                                  size_t compile_cache_id,
                                                                                  bool backward_graph) {
  pipeline::CompileCacheManagerPtr compile_cache_manager =
    std::make_shared<pipeline::CompileCacheManager>(compile_cache_id);
  compile_cache_manager->set_id_extension(extension);
  compile_cache_manager->InitParallelGroupCkptSaveFile();
  const bool force_use_compile_cache = (common::GetEnv("MS_DEV_FORCE_USE_COMPILE_CACHE") == "1");
  auto &context = CompileCacheContext::GetInstance();
  auto jit_executor = pipeline::JitExecutorPy::GetInstance();
  const py::list &compile_cache_dep_files = jit_executor->compile_cache_dep_files();
  // When enabling compile cache, it is possible to enable it even without Python script.
  if (force_use_compile_cache || compile_cache_dep_files.empty()) {
    context.set_init_compile_cache(true);
    MS_LOG(WARNING)
      << "The env MS_DEV_FORCE_USE_COMPILE_CACHE has been set. It will force to use the compile cache without "
         "checking whether the network has been changed. Please note the correctness.";
  } else {
    MsProfileStatGuard stat_guard("InitCompileCache", "compile_cache", true);
    if (!common::UseHostCollective()) {
      context.set_init_compile_cache(true);
    }
    bool compile_cache_consistent = jit_executor->compile_cache_consistent();
    if (!compile_cache_consistent) {
      MS_LOG(WARNING) << "Check the consistency of dependency files hash failed. Execute all the compilation actions.";
      return std::make_pair(nullptr, compile_cache_manager);
    }
  }
  auto manager = MakeManager({}, true, true);
  FuncGraphPtr func_graph = compile_cache_manager->GetCachedFuncGraph(
    manager, weights, ConfigManager::GetInstance().QueueName(), backward_graph);
  return std::make_pair(func_graph, compile_cache_manager);
}

VectorRef ExecuteForward(const pynative::GradParamPtr &grad_param, const FuncGraphPtr &forward_fg,
                         const bool need_forward_result, const bool need_reuse_forward_node, const bool cache_hit) {
  // 2. Execute forward graph if needed
  // Prepare argument list for graph execution
  VectorRef arg_list;
  std::transform(grad_param->op_grad_info->input_value.begin(), grad_param->op_grad_info->input_value.end(),
                 std::back_inserter(arg_list), [](const ValuePtr &value) { return value; });
  ValuePtr forward_output_value = grad_param->op_grad_info->out_value;
  AbstractBasePtr origin_forward_output_abs = grad_param->op_grad_info->out_abs;
  MS_EXCEPTION_IF_NULL(origin_forward_output_abs);
  MS_EXCEPTION_IF_NULL(forward_fg);
  if (need_forward_result) {
    MS_LOG(INFO) << "Start run forward graph result";
    const auto &output = forward_fg->output();
    MS_EXCEPTION_IF_NULL(output);
    const auto &output_abs = output->abstract();
    MS_EXCEPTION_IF_NULL(output_abs);
    if (need_reuse_forward_node) {
      // {prim::kPrimMakeTuple, origin_forward_output, {prim::kPrimMakeTuple, reuse_cnode1, reuse_cnode2, ...}}
      auto tuple_output_abstract = output_abs->cast<abstract::AbstractTuplePtr>();
      if (tuple_output_abstract == nullptr || tuple_output_abstract->size() == 0) {
        MS_LOG(INTERNAL_EXCEPTION) << "Invalid output abstract: " << output_abs->ToString();
      }
      auto node_abstracts = tuple_output_abstract->elements();
      node_abstracts[kIndex0] = origin_forward_output_abs;
      output->set_abstract(std::make_shared<abstract::AbstractTuple>(node_abstracts));
    } else {
      output->set_abstract(origin_forward_output_abs);
    }
    if (grad_param->source_fg->has_user_data("jit_config")) {
      forward_fg->set_user_data<std::map<std::string, std::string>>(
        "jit_config", grad_param->source_fg->user_data<std::map<std::string, std::string>>("jit_config"));
    }
    auto forward_result = GetGraphResult(forward_fg, arg_list, cache_hit, grad_param->graph_cache_key);
    py::object py_forward_result =
      HandleForwardResult(forward_result, forward_fg, origin_forward_output_abs, grad_param, need_reuse_forward_node);
    MS_LOG(DEBUG) << "Run forward graph get result: " << py::str(py_forward_result);
    forward_output_value = parse::data_converter::PyObjToValue(py_forward_result);
    grad_param->op_grad_info->out_value = forward_output_value;
  }
  return arg_list;
}

void CacheFuncGraph(const pipeline::CompileCacheManagerPtr &compile_cache_manager, const FuncGraphPtr &fg, bool loaded,
                    bool cache_hit) {
  if (CompileCacheEnable() && !loaded && !cache_hit) {
    {
      MsProfileStatGuard stat_guard("SaveCacheFuncGraph", "compile_cache", true);
      compile_cache_manager->CacheFuncGraph(fg, nullptr, false, true);
    }
  }
}

std::pair<bool, FuncGraphPtr> GetBpropGraph(const pynative::GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(grad_param);
  MS_EXCEPTION_IF_NULL(grad_param->op_grad_info);

  FuncGraphPtr after_opt_fg = nullptr;
  FuncGraphPtr forward_fg = nullptr;
  BpropGeneratorPtr jit_adgrad_processer = nullptr;

  // Determine if forward result is needed, eg: second grad for high grad no need
  const bool need_forward_result = (grad_param->op_grad_info->out_value == nullptr);
  // Determine if forward node reuse is needed, first grad for high grad is ir_grad, no need reuse forward node
  const bool need_reuse_forward_node = need_forward_result && !grad_param->is_high_order;
  grad_param->graph_cache_key = grad_param->graph_cache_key + (grad_param->is_high_order ? ".high_order" : "");
  MS_LOG(INFO) << "Get Bprop from fprop, need forward result: " << need_forward_result
               << " , need reuse forward node: " << need_reuse_forward_node
               << " , is high order: " << grad_param->is_high_order << " , cache key: " << grad_param->graph_cache_key;

  // 1. Check cache for existing graphs
  const auto it = pass_grad_graph_.find(grad_param->graph_cache_key);
  bool cache_hit = it != pass_grad_graph_.end();
  pipeline::CompileCacheManagerPtr compile_cache_manager = nullptr;
  pipeline::CompileCacheManagerPtr compile_cache_manager_forward = nullptr;
  bool loaded = false;
  py::dict weights;
  if (CompileCacheEnable() && !cache_hit) {
    auto graph_executor = pipeline::JitExecutorPy::GetInstance();
    weights = graph_executor->weights();
    {
      MsProfileStatGuard stat_guard("LoadCachedFuncGraph");
      static size_t idx_forward = 0;
      auto pair_forward = GetCompileCacheResource(weights, "grad_forward", idx_forward++, false);
      forward_fg = pair_forward.first;
      compile_cache_manager_forward = pair_forward.second;
    }
    loaded = forward_fg != nullptr;
    auto &context = CompileCacheContext::GetInstance();
    context.SetUseCompileCache(CompileCacheEnable() && loaded);
  }
  if (cache_hit) {
    MS_LOG(DEBUG) << "Get ad grad graph by cache, cache key: " << grad_param->graph_cache_key;
    std::tie(forward_fg, after_opt_fg) = it->second;
  } else if (!loaded) {
    // Generate backward graph and forward graph with reused cnode as output
    jit_adgrad_processer = std::make_shared<BpropGenerator>(
      BasicClone(grad_param->fg), grad_param->op_grad_info->input_abs, grad_param->op_grad_info->input_value,
      grad_param->op_grad_info->out_abs, need_reuse_forward_node);

    // Generating backward_graph
    MS_LOG(INFO) << "Start generating brop graph.";
    after_opt_fg = jit_adgrad_processer->GenerateBpropGraph();
    MS_LOG(INFO) << "Start optimizing brop graph.";
    pynative::CommonUtils::DumpGraphIR("opt_backward_before_opt.ir", after_opt_fg);
    after_opt_fg = OptimizeBpropGraph(after_opt_fg, grad_param);
    pynative::CommonUtils::DumpGraphIR("opt_backward_after_opt.ir", after_opt_fg);
    jit_adgrad_processer->EraseUnusedReuseCNode(after_opt_fg);
    MS_LOG(INFO) << "Bprop graph generated successfully.";

    // Generating forward_graph
    MS_LOG(INFO) << "Start generating forward graph.";
    forward_fg = jit_adgrad_processer->GenerateForwardGraph(grad_param->source_fg, grad_param->is_control_flow);
    MS_LOG(INFO) << "Forward graph generated successfully.";
    pynative::CommonUtils::DumpGraphIR("opt_forward.ir", forward_fg);
  }
  CacheFuncGraph(compile_cache_manager_forward, forward_fg, loaded, cache_hit);

  VectorRef arg_list = ExecuteForward(grad_param, forward_fg, need_forward_result, need_reuse_forward_node, cache_hit);
  ValuePtr forward_output_value = grad_param->op_grad_info->out_value;
  AbstractBasePtr origin_forward_output_abs = grad_param->op_grad_info->out_abs;

  if (CompileCacheEnable() && !cache_hit) {
    {
      MsProfileStatGuard stat_guard("LoadCachedFuncGraph");
      static size_t idx = 0;
      auto pair = GetCompileCacheResource(weights, "grad", idx++, true);
      after_opt_fg = loaded ? pair.first : after_opt_fg;
      compile_cache_manager = pair.second;
    }
  }

  // 3. Update grad_param info about forward output value
  grad_param->args = arg_list;
  MS_EXCEPTION_IF_NULL(forward_output_value);
  MS_EXCEPTION_IF_NULL(grad_param->op_grad_info->out_value);
  AbstractBasePtr real_forward_output_abs = forward_output_value->ToAbstract();
  MS_EXCEPTION_IF_NULL(origin_forward_output_abs);
  if (origin_forward_output_abs->isa<abstract::AbstractAny>()) {
    grad_param->op_grad_info->out_abs = pynative::CommonUtils::SetAbstractValueToAnyValue(real_forward_output_abs);
  }
  grad_param->jit_out_has_dict = JitOutputHasDict(grad_param->op_grad_info->out_abs);

  if (grad_param->source_fg->has_user_data("jit_config")) {
    after_opt_fg->set_user_data<std::map<std::string, std::string>>(
      "jit_config", grad_param->source_fg->user_data<std::map<std::string, std::string>>("jit_config"));
  }

  // 4. Store forward_graph and bprop
  if (!cache_hit) {
    if (!CompileCacheEnable() || !loaded) {
      jit_adgrad_processer->SetForwardOutputAbs(grad_param->op_grad_info->out_abs, after_opt_fg);
      pynative::CommonUtils::DumpGraphIR("opt_backward.ir", after_opt_fg);
    }
    CacheFuncGraph(compile_cache_manager, after_opt_fg, loaded, cache_hit);
    if (grad_param->is_jit_graph) {
      pass_grad_graph_[grad_param->graph_cache_key] = {forward_fg, after_opt_fg};
    }
  }
  return std::make_pair(cache_hit, after_opt_fg);
}

std::pair<FuncGraphPtr, FuncGraphPtr> CacheFuncGraphBeforeOpt(const FuncGraphPtr &jit_grad_graph,
                                                              const FuncGraphPtr &jit_primal_graph) {
  pipeline::CompileCacheManagerPtr compile_cache_manager = nullptr;
  pipeline::CompileCacheManagerPtr compile_cache_manager_forward = nullptr;
  FuncGraphPtr grad_graph_before_opt = nullptr;
  FuncGraphPtr forward_graph_before_opt = nullptr;
  bool loaded = false;
  if (CompileCacheEnable()) {
    auto graph_executor = pipeline::JitExecutorPy::GetInstance();
    const auto &weights = graph_executor->weights();
    {
      MsProfileStatGuard stat_guard("LoadCachedFuncGraph");
      static size_t idx = 0;
      auto pair = GetCompileCacheResource(py::dict(), "grad_before_opt", idx++, true);
      grad_graph_before_opt = pair.first;
      compile_cache_manager = pair.second;
    }
    {
      MsProfileStatGuard stat_guard("LoadCachedFuncGraph");
      static size_t idx_forward = 0;
      auto pair_forward = GetCompileCacheResource(weights, "grad_forward_before_opt", idx_forward++, false);
      forward_graph_before_opt = pair_forward.first;
      compile_cache_manager_forward = pair_forward.second;
    }
    loaded = grad_graph_before_opt != nullptr && forward_graph_before_opt != nullptr;
  }
  if (!loaded) {
    grad_graph_before_opt = jit_grad_graph;
    forward_graph_before_opt = jit_primal_graph;
  }
  CacheFuncGraph(compile_cache_manager, jit_grad_graph, loaded, false);
  CacheFuncGraph(compile_cache_manager_forward, jit_primal_graph, loaded, false);
  auto jit_config = PhaseManager::GetInstance().jit_config();
  MS_EXCEPTION_IF_NULL(forward_graph_before_opt);
  forward_graph_before_opt->set_user_data<std::map<std::string, std::string>>(
    "jit_config", std::make_shared<std::map<std::string, std::string>>(jit_config));
  return std::pair(grad_graph_before_opt, forward_graph_before_opt);
}

void ClearGradCache() {
  pass_grad_graph_.clear();
  jit_forward_resource.clear();
  check_invalid_dout_bprop_graph.clear();
  origin_grad_graph_.clear();
  filtered_grad_graph.clear();
  CompileCacheContext::GetInstance().Clear();
}

void BpropGenerator::ReuseCustomBpropForwardOutput(const FuncGraphPtr &k_fg, const FuncGraphPtr &top_fg) {
  const auto &forward_fg_iter = k_fg->transforms().find("custom_bprop_primal");
  if (forward_fg_iter == k_fg->transforms().end()) {
    return;
  }
  auto primal_forward_fg = forward_fg_iter->second.func_graph();
  for (auto node : TopoSort(top_fg->output())) {
    if (node == nullptr) {
      continue;
    }
    auto forward_output = node->cast<CNodePtr>();
    if (forward_output == nullptr) {
      continue;
    }
    if (GetValueNode<FuncGraphPtr>(forward_output->input(0)) != primal_forward_fg) {
      continue;
    }
    auto &forward_output_abs = forward_output->abstract();
    MS_EXCEPTION_IF_NULL(forward_output_abs);
    MS_LOG(INFO) << "Reuse custom bprop's forward output node: " << forward_output->DebugString()
                 << ", with index: " << fprop_sub_fgs_.size();
    (void)fprop_sub_fgs_.emplace_back(k_fg);
    (void)replace_nodes_.emplace_back(forward_output);
    (void)replace_nodes_abs_.emplace_back(forward_output_abs);
  }
}

void BpropGenerator::ReusePrimalCNode(const FuncGraphPtr &k_fg, const FuncGraphPtr &top_fg,
                                      bool top_cell_do_recompute) {
  // Find primal cnode for this fprop
  const auto &primal_cnode_iter = k_fg->transforms().find("primal_cnode");
  if (primal_cnode_iter == k_fg->transforms().end()) {
    return;
  }
  // Filter control flow graph and unsupported prim
  const auto &primal_cnode = primal_cnode_iter->second.primal_cnode();
  MS_EXCEPTION_IF_NULL(primal_cnode);
  if (primal_cnode->func_graph() != top_fg || IsUnSupportPrim(primal_cnode)) {
    return;
  }
  // Do not reuse recompute sub cell's cnode
  if (!top_cell_do_recompute && HasRecomputedScope(primal_cnode)) {
    MS_LOG(DEBUG) << "Need recompute cnode: " << primal_cnode->DebugString();
    return;
  }
  // Process primal abstract
  const auto &prim_abstract = primal_cnode->abstract();
  if (!IsValidAbstract(prim_abstract)) {
    return;
  }
  MS_LOG(INFO) << "Reuse forward output node: " << primal_cnode->DebugString()
               << ", with index: " << fprop_sub_fgs_.size();
  (void)fprop_sub_fgs_.emplace_back(k_fg);
  (void)replace_nodes_.emplace_back(primal_cnode);
  (void)replace_nodes_abs_.emplace_back(prim_abstract);
}

void BpropGenerator::Init() {
  basic_graph_ = std::make_shared<FuncGraph>();
  basic_graph_->debug_info()->set_name("bprop_builder");

  // Generate bprop function: basic_graph_(inputs, dout) ==> dins
  // (result, bprop) = fprop_graph_(inputs)
  auto fprop_app_inputs = ProcessParam(basic_graph_, input_abs_, input_value_);
  fprop_app_inputs.insert(fprop_app_inputs.begin(), NewValueNode(fprop_graph_));
  // Get bprop from fprop_fg, it is 2nd output of fprop_fg
  auto fprop_app = basic_graph_->NewCNode(fprop_app_inputs);
  auto get_bprop = basic_graph_->NewCNode(
    {NewValueNode(prim::kPrimTupleGetItem), fprop_app, NewValueNode(static_cast<int64_t>(kIndex1))});

  // (df, dinputs) = bprop(dout)
  // Get dinputs from calling bprop funcgraph
  AnfNodePtrList node_list{get_bprop};
  auto dout = basic_graph_->add_parameter();
  dout->set_abstract(out_abs_);
  (void)node_list.emplace_back(dout);
  auto call_bprop = basic_graph_->NewCNode(node_list);
  AnfNodePtrList actual_out{NewValueNode(prim::kPrimMakeTuple)};
  for (size_t i = 0; i < input_abs_.size(); ++i) {
    // Index 0 env, skip
    auto out =
      basic_graph_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), call_bprop, NewValueNode(SizeToLong(i + 1))});
    (void)actual_out.emplace_back(out);
  }
  basic_graph_->set_output(basic_graph_->NewCNode(actual_out));
  ClearFuncGraphCNodeAbstract(basic_graph_);
  pynative::CommonUtils::DumpGraphIR("opt_before.ir", basic_graph_);

  if (!need_reuse_forward_node_) {
    return;
  }

  // Find necessary sub fprop graphs
  auto primal_fg_iter = fprop_graph_->transforms().find("primal");
  if (primal_fg_iter == fprop_graph_->transforms().end()) {
    return;
  }
  auto primal_fg = primal_fg_iter->second.func_graph();
  MS_EXCEPTION_IF_NULL(primal_fg);

  // Check whether top cell do recompute
  bool top_cell_do_recompute = primal_fg->has_flag(kTopCellWithRecompute);

  for (const auto &node : TopoSort(basic_graph_->return_node(), SuccDeeperSimple)) {
    // Check fprop graph for each prim
    auto k_fg = GetValueNode<FuncGraphPtr>(node);
    if (!k_fg) {
      continue;
    }
    ReuseCustomBpropForwardOutput(k_fg, primal_fg);
    ReusePrimalCNode(k_fg, primal_fg, top_cell_do_recompute);
  }
  // Check param, if modified by inplace ops, need insert tensor move
  const auto &bprop_params = basic_graph_->parameters();
  const auto &forword_params = primal_fg->parameters();
  for (size_t i = 0; i < input_abs_.size(); ++i) {
    if (IsViewInplaceAbs(input_abs_[i])) {
      (void)fprop_modified_params_.emplace_back(bprop_params[i]);
      (void)replace_nodes_.emplace_back(forword_params[i]);
      (void)replace_nodes_abs_.emplace_back(input_abs_[i]);
    }
  }
  MS_LOG(INFO) << "Finish init generating basic bprop func graph for " << fprop_graph_->ToString() << ", there are "
               << fprop_sub_fgs_.size() << " forward nodes could be reused.";
}

FuncGraphPtr BpropGenerator::GenerateBpropGraph() {
  if (need_reuse_forward_node_) {
    bprop_origin_param_size_ = basic_graph_->parameters().size();
    auto back_manager = Manage({basic_graph_}, false);
    size_t index = 0;
    for (const auto &k_fg : fprop_sub_fgs_) {
      auto param = basic_graph_->add_parameter();
      auto output = k_fg->output();
      MS_EXCEPTION_IF_NULL(output);
      auto output_cnode = output->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(output_cnode);
      auto forward_output_node = output_cnode->input(kIndex1);
      back_manager->Replace(forward_output_node, param);
      param->set_abstract(replace_nodes_abs_[index++]);
    }
    for (const auto &bprop_param : fprop_modified_params_) {
      auto param = basic_graph_->add_parameter();
      back_manager->Replace(bprop_param, param);
      param->set_abstract(replace_nodes_abs_[index++]);
    }
  }
  return basic_graph_;
}

// Erase unused forward_reused params after bprop_sub_fg expanded
void BpropGenerator::EraseUnusedReuseCNode(const FuncGraphPtr &bprop_fg) {
  if (!need_reuse_forward_node_) {
    return;
  }
  MS_EXCEPTION_IF_NULL(bprop_fg);
  auto manager = Manage({bprop_fg}, false);
  auto params = bprop_fg->parameters();
  auto node_users = manager->node_users();
  AnfNodePtrList new_params;
  for (size_t index = 0; index < params.size(); ++index) {
    auto param = params[index];
    // Add original forward inputs and dout
    if (index < bprop_origin_param_size_) {
      (void)new_params.emplace_back(param);
      continue;
    }
    // Add params have actual users in bprop fg
    auto use_node_size = node_users[param].size();
    if (use_node_size != 0) {
      (void)new_params.emplace_back(param);
    } else {
      size_t origin_reuse_index = index - bprop_origin_param_size_;
      MS_EXCEPTION_IF_CHECK_FAIL(origin_reuse_index < replace_nodes_.size(),
                                 "Reused nodes vector size should less than param size");
      MS_LOG(DEBUG) << "Unused primal cnode in bprop graph: " << replace_nodes_[origin_reuse_index]->DebugString();
      replace_nodes_[origin_reuse_index] = nullptr;
      replace_nodes_abs_[origin_reuse_index] = nullptr;
      if (fprop_sub_fgs_.size() > origin_reuse_index) {
        fprop_sub_fgs_[origin_reuse_index] = nullptr;
      } else {
        fprop_modified_params_[origin_reuse_index - fprop_sub_fgs_.size()] = nullptr;
      }
    }
  }
  bprop_fg->set_parameters(new_params);
}

FuncGraphPtr BpropGenerator::GenerateForwardGraph(const FuncGraphPtr &jit_forward_graph, bool do_renormalize) {
  if (!need_reuse_forward_node_) {
    return OptimizeForwardGraph(BasicClone(jit_forward_graph), do_renormalize);
  }
  auto primal_fg_iter = fprop_graph_->transforms().find("primal");
  if (primal_fg_iter == fprop_graph_->transforms().end()) {
    return OptimizeForwardGraph(BasicClone(jit_forward_graph), do_renormalize);
  }
  // Need modify forward output
  // From {kPrimReturn, original_output} ==> {kPrimReturn, {kPrimMakeTuple, original_output, reused_cnodes}}
  const auto &primal_fg = primal_fg_iter->second.func_graph();
  MS_EXCEPTION_IF_NULL(primal_fg);
  pynative::CommonUtils::DumpGraphIR("primal_graph.ir", primal_fg);
  const auto &params = primal_fg->parameters();
  if (params.size() != input_abs_.size()) {
    MS_LOG(EXCEPTION) << "Unmatched param size for primal_fg: " << primal_fg->ToString();
  }
  for (size_t index = 0; index < input_abs_.size(); ++index) {
    auto param = params[index]->cast<ParameterPtr>();
    param->set_abstract(input_abs_[index]);
  }

  MS_LOG(INFO) << "Start appending reused nodes to forward graph output.";
  // {Primal_fg(inputs) = foward_result} ==> {Primal_fg(inputs) = (foward_result, reused nodes)}
  // Get original output node and abstract, and merge original output node and used forward nodes to return node.
  auto original_output_node = primal_fg->output();
  MS_EXCEPTION_IF_NULL(original_output_node);
  AnfNodePtrList fprop_forward_outputs{NewValueNode(prim::kPrimMakeTuple), original_output_node};
  auto primal_graph_manager = MakeManager({primal_fg}, false);
  for (size_t i = 0; i < replace_nodes_.size(); ++i) {
    if (replace_nodes_[i] == nullptr) {
      continue;
    }
    auto &node = replace_nodes_[i];
    if (!node->isa<Parameter>()) {
      (void)fprop_forward_outputs.emplace_back(node);
      continue;
    }
    // Reuse load param, insert a tensor move node
    auto insert_tensor_move = primal_fg->NewCNode({NewValueNode(prim::kPrimTensorMove), node});
    auto insert_depend_move = primal_fg->NewCNode({NewValueNode(prim::kPrimDepend), node, insert_tensor_move});
    primal_graph_manager->Replace(node, insert_depend_move);
    (void)fprop_forward_outputs.emplace_back(insert_tensor_move);
  }
  auto merge_node = primal_fg->NewCNode(std::move(fprop_forward_outputs));
  primal_fg->set_output(merge_node);
  auto forward_fg = BasicClone(primal_fg);
  primal_fg->set_output(original_output_node);
  MS_LOG(INFO) << "Finish appending reused nodes to forward graph output.";
  return OptimizeForwardGraph(forward_fg, true);
}

void BpropGenerator::SetForwardOutputAbs(const abstract::AbstractBasePtr &forward_abs,
                                         const FuncGraphPtr &bprop_graph) {
  if (bprop_graph->parameters().empty()) {
    return;
  }
  auto input_value_size = input_value_.size();
  auto &dout_param = bprop_graph->parameters()[input_value_size];
  dout_param->set_abstract(forward_abs);
  PlantFuncGradBpropGraphDout(bprop_graph, input_value_size, forward_abs);
}

bool CheckTupleNeedGrad(const ValueSequencePtr &seq) {
  const auto &elements = seq->value();
  for (const auto &element : elements) {
    if (element->isa<ValueSequence>()) {
      const auto &arg_tuple = element->cast<ValueSequencePtr>();
      if (CheckTupleNeedGrad(arg_tuple)) {
        return True;
      }
    } else if (element->isa<tensor::Tensor>()) {
      const auto &tensor = element->cast<tensor::TensorPtr>();
      if (pynative::autograd::impl::RequiresGrad(tensor)) {
        return True;
      }
    }
  }
  return false;
}

std::vector<bool> GetNeedGradIndexes(const VectorRef &args) {
  std::vector<bool> need_grad_indexes;
  std::transform(args.begin(), args.end(), std::back_inserter(need_grad_indexes), [](const auto &arg) {
    if (utils::isa<ValueSequence>(arg)) {
      const auto &arg_tuple = utils::cast<ValueSequencePtr>(arg);
      return CheckTupleNeedGrad(arg_tuple);
    }
    if (!utils::isa<tensor::Tensor>(arg)) {
      return false;
    }
    const auto &tensor = utils::cast<tensor::TensorPtr>(arg);
    return pynative::autograd::impl::RequiresGrad(tensor);
  });
  return need_grad_indexes;
}

size_t GetArgLength(const BaseRef &arg) {
  if (utils::isa<ValueSequence>(arg)) {
    const auto &arg_sequence = utils::cast<ValueSequencePtr>(arg);
    auto real_arg = pynative::CommonUtils::FlattenOnlyTensor(arg_sequence);
    return real_arg.size();
  } else if (utils::isa<ValueDictionary>(arg)) {
    const auto &arg_sequence = utils::cast<ValueDictionaryPtr>(arg);
    auto real_arg = pynative::CommonUtils::FlattenOnlyTensor(arg_sequence);
    return real_arg.size();
  }
  return 1;
}

bool FilterGradOutput(const std::vector<bool> &need_grad, const FuncGraphPtr &func_graph, const VectorRef &args,
                      std::vector<pynative::autograd::Edge> *next_edges) {
  MS_LOG(INFO) << "Start filter grad function graph output";
  MS_EXCEPTION_IF_NULL(func_graph->output());
  auto graph_output = func_graph->output()->cast<CNodePtr>();
  if (graph_output == nullptr) {
    MS_LOG(INFO) << "Do not filter grad output for constant output " << func_graph->output()->DebugString();
    return false;
  }
  MS_EXCEPTION_IF_NULL(graph_output);
  const auto &graph_output_element = graph_output->inputs();
  MS_EXCEPTION_IF_CHECK_FAIL(graph_output_element.size() - 1 == need_grad.size(), "Size not match");
  AnfNodePtrList new_graph_output_element = {NewValueNode(prim::kPrimMakeTuple)};
  AbstractBasePtrList new_graph_output_abstract_element;
  bool need_filter = false;
  std::vector<pynative::autograd::Edge> new_edge;
  for (size_t i = 0; i < need_grad.size(); ++i) {
    size_t arg_size = GetArgLength(args[i]);
    if (i + arg_size - 1 >= next_edges->size()) {
      MS_LOG(EXCEPTION) << "Number of args excced the number of edges";
    }
    auto cur_node = graph_output_element[i + 1];
    MS_EXCEPTION_IF_NULL(cur_node);
    auto cur_abstract = cur_node->abstract();
    MS_EXCEPTION_IF_NULL(cur_abstract);
    if (need_grad[i]) {
      (void)new_graph_output_element.emplace_back(cur_node);
      (void)new_graph_output_abstract_element.emplace_back(cur_abstract);
      for (size_t j = 0; j < arg_size; ++j) {
        (void)new_edge.emplace_back((*next_edges)[i + j]);
      }
      i += arg_size - 1;
      continue;
    }
    i += arg_size - 1;
    need_filter = true;
  }
  constexpr auto need_grad_key = "need_grad";
  func_graph->set_attr(need_grad_key, MakeValue<std::vector<bool>>(need_grad));
  if (!need_filter) {
    if (MsContext::GetInstance()->CanDump(kIntroductory)) {
      DumpIR("filtered_output_grad_fg.ir", func_graph);
    }
    return need_filter;
  }
  MS_LOG(INFO) << "Do filter for grad function graph output";
  next_edges->clear();
  next_edges->insert(next_edges->begin(), new_edge.begin(), new_edge.end());
  auto new_graph_output = func_graph->NewCNode(new_graph_output_element);
  auto new_graph_output_abstract = std::make_shared<abstract::AbstractTuple>(new_graph_output_abstract_element);
  new_graph_output->set_abstract(new_graph_output_abstract);
  func_graph->set_output(new_graph_output);
  if (MsContext::GetInstance()->CanDump(kIntroductory)) {
    DumpIR("filtered_output_grad_fg.ir", func_graph);
  }
  return need_filter;
}

void FilterGradInput(const std::vector<bool> &need_filter, const FuncGraphPtr &func_graph, size_t add_args_size,
                     size_t skip_filter_size) {
  const auto &bprop_parameters = func_graph->parameters();
  AnfNodePtrList new_bprop_parameters;
  for (size_t i = 0; i < skip_filter_size; ++i) {
    (void)new_bprop_parameters.emplace_back(bprop_parameters[i]);
  }
  for (size_t i = 0; i < add_args_size; ++i) {
    bool cur_need_filter = need_filter[i];
    if (!cur_need_filter) {
      (void)new_bprop_parameters.emplace_back(bprop_parameters[i + skip_filter_size]);
    }
  }
  func_graph->set_parameters(new_bprop_parameters);
  if (MsContext::GetInstance()->CanDump(kIntroductory)) {
    DumpIR("filtered_bprop_fg.ir", func_graph);
  }
}

VectorRef RefreshAddedArgs(const VectorRef &added_args, const std::vector<bool> &need_filter, size_t add_args_size) {
  std::vector<BaseRef> new_added_args_element;
  for (size_t i = 0; i < add_args_size; ++i) {
    bool cur_need_filter = need_filter[i];
    if (!cur_need_filter) {
      (void)new_added_args_element.emplace_back(added_args[i]);
    }
  }
  return VectorRef(new_added_args_element);
}

void FilterForwardOutput(const std::vector<bool> &need_filter, const std::string &cache_key, size_t add_args_size) {
  auto forward_graph = GetGradAndForwardGraph(cache_key).first;
  MS_EXCEPTION_IF_NULL(forward_graph);
  auto forward_graph_output = forward_graph->output();
  MS_EXCEPTION_IF_CHECK_FAIL(IsPrimitiveCNode(forward_graph_output, prim::kPrimMakeTuple), "Invalid output");
  const auto &forward_graph_output_elements = forward_graph_output->cast<CNodePtr>()->inputs();
  // one for kPrimMakeTuple, one for real graph output.
  constexpr auto output_arg_diff = 2;
  MS_EXCEPTION_IF_CHECK_FAIL(forward_graph_output_elements.size() - output_arg_diff == add_args_size, "Size not match");
  AnfNodePtrList new_forward_output_elements = {NewValueNode(prim::kPrimMakeTuple), forward_graph_output_elements[1]};
  auto forward_graph_output_elements_abstract = forward_graph_output->abstract();
  MS_EXCEPTION_IF_NULL(forward_graph_output_elements_abstract);
  MS_EXCEPTION_IF_CHECK_FAIL(forward_graph_output_elements_abstract->isa<abstract::AbstractTuple>(), "cast failed");
  const auto &forward_graph_output_elements_abstract_elements =
    forward_graph_output_elements_abstract->cast<abstract::AbstractTuplePtr>()->elements();
  AbstractBasePtrList new_forward_output_abstract_elements = {forward_graph_output_elements_abstract_elements[0]};
  for (size_t i = 0; i < add_args_size; ++i) {
    bool cur_need_filter = need_filter[i];
    if (!cur_need_filter) {
      (void)new_forward_output_elements.emplace_back(forward_graph_output_elements[i + output_arg_diff]);
      (void)new_forward_output_abstract_elements.emplace_back(forward_graph_output_elements_abstract_elements[i + 1]);
    }
  }
  auto new_forward_output = forward_graph->NewCNode(new_forward_output_elements);
  new_forward_output->set_abstract(std::make_shared<abstract::AbstractTuple>(new_forward_output_abstract_elements));
  forward_graph->set_output(new_forward_output);
  constexpr auto need_repeat_task_emit_key = "need_repeat_task_emit";
  forward_graph->set_flag(need_repeat_task_emit_key, true);
  if (MsContext::GetInstance()->CanDump(kIntroductory)) {
    DumpIR("filtered_forward_fg.ir", forward_graph);
  }
}

std::pair<std::vector<bool>, int> CollectFilterMsg(const VectorRef &added_args, const FuncGraphPtr &func_graph) {
  const auto &bprop_parameters = func_graph->parameters();
  auto add_args_size = added_args.size();
  MS_LOG(INFO) << "add_args_size: " << add_args_size;
  auto skip_filter_size = bprop_parameters.size() - add_args_size;
  MS_LOG(INFO) << "Skip filter size: " << skip_filter_size;

  ud_chain::Preprocess(func_graph);
  std::vector<bool> need_filter(add_args_size);
  for (size_t i = 0; i < add_args_size; ++i) {
    auto cur_bprop_parameters = bprop_parameters[i + skip_filter_size];
    const auto &cur_users = ud_chain::GetUsers(cur_bprop_parameters);
    need_filter[i] = cur_users.empty();
  }
  return std::make_pair(need_filter, skip_filter_size);
}

void UpdateNextEdge(std::vector<pynative::autograd::Edge> *next_edges, const FuncGraphPtr &func_graph,
                    const VectorRef &args) {
  constexpr auto need_grad_key = "need_grad";
  const auto &need_grad_value = func_graph->attrs()[need_grad_key];
  const auto &need_grad = GetValue<std::vector<bool>>(need_grad_value);
  std::vector<pynative::autograd::Edge> new_edge;
  for (size_t i = 0; i < need_grad.size(); ++i) {
    size_t arg_size = GetArgLength(args[i]);
    if (i + arg_size - 1 >= next_edges->size()) {
      MS_LOG(EXCEPTION) << "Number of args excced the number of edges";
    }
    if (need_grad[i]) {
      for (size_t j = 0; j < arg_size; ++j) {
        (void)new_edge.emplace_back((*next_edges)[i + j]);
      }
      i += arg_size - 1;
      continue;
    }
    i += arg_size - 1;
  }
  next_edges->clear();
  next_edges->insert(next_edges->begin(), new_edge.begin(), new_edge.end());
}

FuncGraphPtr FilterGraphOutput(const bool is_filtered, const std::pair<VectorRef, VectorRef> arg_pair,
                               const FuncGraphPtr &func_graph, const std::string &cache_key,
                               std::vector<pynative::autograd::Edge> *next_edges) {
  const auto &args = arg_pair.first;
  const auto &added_args = arg_pair.second;
  const auto &need_grad = GetNeedGradIndexes(args);
  size_t need_grad_hash = std::hash<std::vector<bool>>()(need_grad);
  FuncGraphPtr new_graph = func_graph;
  if (is_filtered) {
    auto cache_filtered_graph = GetFilteredGradGraph(cache_key, need_grad_hash);
    if (cache_filtered_graph != nullptr) {
      MS_LOG(INFO) << "Found cached filtered grad graph for hash key " << need_grad_hash;
      MS_EXCEPTION_IF_NULL(cache_filtered_graph->output());
      auto graph_output = cache_filtered_graph->output()->cast<CNodePtr>();
      if (graph_output == nullptr) {
        MS_LOG(INFO) << "Do not filter grad output for constant output "
                     << cache_filtered_graph->output()->DebugString();
        return cache_filtered_graph;
      }
      UpdateNextEdge(next_edges, func_graph, args);
      return func_graph;
    }
    MS_LOG(INFO) << "Cache find graph failed, filter grad graph again.";
    const auto &cloned_graph = BasicClone(GetOriginGradGraph(cache_key));
    auto resource = std::make_shared<pipeline::Resource>();
    resource->set_func_graph(cloned_graph);
    auto manager = resource->manager();
    MS_EXCEPTION_IF_NULL(manager);
    manager->AddFuncGraph(cloned_graph);
    abstract::AbstractBasePtrList args_abs;
    const auto &parameters = cloned_graph->parameters();
    (void)std::transform(parameters.begin(), parameters.end(), std::back_inserter(args_abs),
                         [](const AnfNodePtr &p) -> AbstractBasePtr { return p->abstract(); });
    new_graph = pipeline::Renormalize(resource, cloned_graph, args_abs);
    MS_EXCEPTION_IF_NULL(new_graph);
  }
  MS_LOG(INFO) << "Start to filter grad jit graph output.";
  (void)FilterGradOutput(need_grad, new_graph, args, next_edges);
  auto cur_size = StoreFilteredGradGraph(cache_key, need_grad_hash, new_graph);

  auto forward_input_size = new_graph->parameters().size() - added_args.size() - 1;
  constexpr size_t capacity_factor = 2;
  if (cur_size > forward_input_size * capacity_factor) {
    MS_LOG(WARNING) << "Cache filtered grad graph size is " << cur_size << " exceed expected maximum capacity "
                    << forward_input_size * capacity_factor;
  }
  constexpr auto need_grad_hash_key = "need_grad_hash";
  new_graph->set_attr(need_grad_hash_key, MakeValue<size_t>(need_grad_hash));
  MS_LOG(INFO) << "Finish to filter grad jit graph output.";
  return new_graph;
}

VectorRef FilterGraphInputOutput(bool is_filtered, const std::pair<VectorRef, VectorRef> arg_pair,
                                 const FuncGraphPtr &func_graph, const std::string &cache_key,
                                 std::vector<pynative::autograd::Edge> *next_edges) {
  const auto &args = arg_pair.first;
  const auto &added_args = arg_pair.second;
  if (is_filtered) {
    MS_LOG(INFO) << "Grad graph is filtered.";
    UpdateNextEdge(next_edges, func_graph, args);
    return added_args;
  }
  MS_LOG(INFO) << "Start to filter grad jit graph.";
  const auto &need_grad = GetNeedGradIndexes(args);
  auto filtered = FilterGradOutput(need_grad, func_graph, args, next_edges);
  if (!filtered) {
    MS_LOG(INFO) << "No need to filter grad jit graph.";
    return added_args;
  }
  MS_LOG(INFO) << "Finish filter grad jit graph.";
  const auto &filter_msg = CollectFilterMsg(added_args, func_graph);
  const auto &need_filter = filter_msg.first;
  auto skip_filter_size = filter_msg.second;
  auto add_args_size = need_filter.size();
  if (add_args_size == 0 || std::all_of(need_filter.begin(), need_filter.end(), [](auto e) { return !e; })) {
    MS_LOG(INFO) << "No need to filter grad input";
    return added_args;
  }
  MS_LOG(INFO) << "Start to filter grad input.";
  FilterGradInput(need_filter, func_graph, add_args_size, skip_filter_size);
  const auto &new_added_args = RefreshAddedArgs(added_args, need_filter, add_args_size);
  FilterForwardOutput(need_filter, cache_key, add_args_size);
  MS_LOG(INFO) << "Finish filter grad input.";
  return new_added_args;
}

std::pair<FuncGraphPtr, VectorRef> FilterGraph(const VectorRef &args, const VectorRef &added_args,
                                               const FuncGraphPtr &func_graph, const std::string &cache_key,
                                               std::vector<pynative::autograd::Edge> *next_edges) {
  const auto &filter_level = common::GetCompileConfig("GRAD_JIT_FILTER");
  if (filter_level != "1" && filter_level != "2") {
    return std::pair(func_graph, added_args);
  }
  bool is_filtered = HasOriginGradGraph(cache_key);
  if (!is_filtered) {
    MS_LOG(INFO) << "Store origin bprop graph for jit.";
    StoreOriginGradGraph(cache_key, BasicClone(func_graph));
  }
  if (filter_level == "1") {
    MS_LOG(INFO) << "Filter grad graph output.";
    const auto &new_graph =
      FilterGraphOutput(is_filtered, std::pair(args, added_args), func_graph, cache_key, next_edges);
    if (func_graph->has_user_data("jit_config")) {
      new_graph->set_user_data<std::map<std::string, std::string>>(
        "jit_config", func_graph->user_data<std::map<std::string, std::string>>("jit_config"));
    }
    return std::pair(new_graph, added_args);
  } else if (filter_level == "2") {
    MS_LOG(INFO) << "Filter grad graph input and output.";
    const auto &new_added_args =
      FilterGraphInputOutput(is_filtered, std::pair(args, added_args), func_graph, cache_key, next_edges);
    return std::pair(func_graph, new_added_args);
  }
  return std::pair(func_graph, added_args);
}
}  // namespace ad
}  // namespace mindspore
