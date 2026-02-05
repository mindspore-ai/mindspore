/**
 * Copyright 2019-2026 Huawei Technologies Co., Ltd
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

#include "frontend/jit/ps/action.h"

#include <memory>
#include <map>
#include <utility>
#include <vector>
#include <set>
#include <string>
#include <algorithm>
#include <functional>

#include "include/utils/ir_dump/draw_interface.h"
#include "include/utils/ir_dump/anf_ir_dump_interface.h"
#include "ir/core_ops_primitive.h"
#include "primitive/sequence_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/structure_ops.h"
#include "ir/anf.h"
#include "ir/func_graph_cloner.h"
#include "ir/param_info.h"
#include "ir/cell.h"
#include "device_address/device_address.h"
#include "include/frontend/jit/ps/action_interface.h"
#include "include/utils/pynative/grad_state.h"
#include "include/backend/common/kernel_graph/py_execute_utils.h"
#include "include/utils/python_adapter.h"
#include "include/utils/anfalgo.h"
#include "include/utils/utils.h"
#include "include/utils/tensor_py_wrapper.h"
#include "include/utils/parallel_context.h"
#include "include/utils/pipeline/fallback.h"
#include "include/runtime/utils/runtime_conf/runtime_env.h"
#include "abstract/abstract_value.h"
#include "frontend/operator/composite/composite.h"
#include "frontend/parallel/step_auto_parallel.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/shard/shard.h"
#include "include/frontend/jit/ps/executor/graph_executor_py.h"
#include "include/frontend/jit/ps/executor/jit_executor_py.h"
#include "frontend/jit/ps/parse/parse_base.h"
#include "frontend/jit/ps/parse/data_converter.h"
#include "frontend/jit/ps/static_analysis/auto_monad.h"
#include "frontend/jit/ps/static_analysis/order_enforce.h"
#include "frontend/jit/ps/static_analysis/static_analysis.h"
#include "frontend/jit/ps/static_analysis/async_eval_result.h"
#include "frontend/jit/ps/static_analysis/program_specialize.h"
#include "frontend/jit/ps/static_analysis/event_method.h"
#include "frontend/jit/ps/static_analysis/parameter_forward_hook.h"
#include "frontend/jit/ps/resource.h"
#include "frontend/jit/ps/remove_value_node_dup.h"
#include "frontend/jit/ps/event_message_print.h"
#include "include/frontend/optimizer/optimizer.h"
#include "frontend/optimizer/ad/grad.h"
#include "frontend/optimizer/custom_pass/custom_pass_executor.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "utils/trace_info.h"
#include "utils/phase.h"
#include "utils/compile_config.h"
#include "load_mindir/infer_mindir.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "tools/profiler/profiling.h"
#include "tools/profiler/mstx/mstx_impl.h"
#include "frontend/optimizer/fallback_rewriter.h"
#include "frontend/jit/ps/load_mindir.h"
#include "frontend/jit/ps/pass_config.h"

#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/cluster/topology/cluster_context.h"
#else
#include "include/cluster/topology/dummy_cluster_context.h"
#endif
#include "tools/profiler/profiling_framework_data.h"
#include "tools/profiler/profiler.h"
#include "include/backend/backend_manager/backend_manager.h"
#include "include/utils/pynative/adapter.h"
#include "include/backend/backend_manager/backend_jit_config.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_p.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "ir/func_graph_flag.h"

namespace mindspore {
namespace pipeline {
namespace {
const auto kFirstInput = 1;
const auto kSecondInput = 2;
const auto kLazyInlineThershold = 64;
using VmEvalFunc = std::function<BaseRef(const VectorRef &)>;
using VmEvalPtr = std::shared_ptr<std::function<BaseRef(const VectorRef &)>>;

bool ExistControlFlow(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  return !func_graph->func_graphs_used_total().empty();
}

bool EnableGradForScalar(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  return (MsContext::GetInstance()->get_param<bool>(MS_CTX_GRAD_FOR_SCALAR) ||
          common::GetCompileConfig("GRAD_FOR_SCALAR") == "1") &&
         abs->BuildType() != nullptr && abs->BuildType()->isa<Number>();
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

void SetParamFormatFromTensor(const ParameterPtr &param_node, const ValuePtr &param_value) {
  MS_EXCEPTION_IF_NULL(param_value);
  if (param_value->isa<tensor::Tensor>()) {
    auto param_tensor = param_value->cast<tensor::TensorPtr>();
    if (param_tensor->device_address()) {
      auto param_deviceaddress = std::dynamic_pointer_cast<device::DeviceAddress>(param_tensor->device_address());
      MS_EXCEPTION_IF_NULL(param_deviceaddress);
      MS_LOG(DEBUG) << "Update param format from tensor format: " << param_tensor->format()
                    << ", param: " << param_node->DebugString();
      param_node->set_format(kernel::GetFormatFromEnumToStr(param_tensor->format()));
    }
  } else if (param_value->isa<ValueSequence>()) {
    // For kv-cache scene, suppose all tensors in kv-cache is same format.
    auto param_tuple = param_value->cast<ValueSequencePtr>();
    if (param_tuple->size() != 0) {
      SetParamFormatFromTensor(param_node, (*param_tuple)[0]);
    }
  }
}

void UpdateFuncGraphParameter(const FuncGraphPtr &func_graph, const std::vector<ValuePtr> &arguments,
                              const std::vector<ValuePtr> &real_arguments) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> new_paras;
  for (size_t i = 0; i < func_graph->parameters().size(); ++i) {
    const auto &param = func_graph->parameters()[i];
    auto param_node = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_node);

    if (param_node->has_default()) {
      new_paras.push_back(param_node);
      auto default_param = param_node->default_param();
      MS_EXCEPTION_IF_NULL(default_param);
      SetParamFormatFromTensor(param_node, default_param);
      continue;
    }

    // Handle the Parameter from input arguments.
    if (i < arguments.size()) {
      auto param_value = tensor::GetMetaTensorFromValue(arguments[i]);
      if (param_value != nullptr && param_value->is_parameter()) {
        param_node->set_default_param(arguments[i]);
      }
    }
    // Set param's format from real arguments, since cannot get format from arguments in dynamic-shape scene.
    if (i < real_arguments.size()) {
      SetParamFormatFromTensor(param_node, real_arguments[i]);
    }

    AbstractBasePtr param_abs = param_node->abstract();
    MS_EXCEPTION_IF_NULL(param_abs);
    if ((param_abs->BuildValue() == kValueAny && !ContainsAbstractFunction(param_abs)) ||
        EnableGradForScalar(param_abs) || EnableSequenceBroaden(param_abs) ||
        param_abs->isa<abstract::AbstractEvent>()) {
      new_paras.push_back(param_node);
    } else {
      MS_LOG(INFO) << "Remove the " << i << "th parameter, since it's passed a constant argument.";
    }
  }
  func_graph->set_parameters(new_paras);
}

bool CheckDuplicatedParameterName(const ValuePtr &arg, std::map<std::string, ValuePtr> *param_names) {
  if (arg->isa<ValueSequence>()) {
    const auto &elements = arg->cast<ValueSequencePtr>()->value();
    return std::all_of(elements.begin(), elements.end(),
                       [param_names](const auto &elem) { return CheckDuplicatedParameterName(elem, param_names); });
  }

  if (arg->isa<ValueDictionary>()) {
    const auto &key_values = arg->cast<ValueDictionaryPtr>()->value();
    return std::all_of(key_values.begin(), key_values.end(), [param_names](const auto &pair) {
      return CheckDuplicatedParameterName(pair.second, param_names);
    });
  }

  if (!arg->isa<tensor::Tensor>()) {
    return true;
  }

  auto arg_tensor = arg->cast<tensor::TensorPtr>();
  if (!arg_tensor->is_parameter()) {
    return true;
  }

  auto param_info = arg_tensor->param_info();
  MS_EXCEPTION_IF_NULL(param_info);
  auto param_name = param_info->name();
  auto found = param_names->find(param_name);
  if (found != param_names->end()) {
    if (found->second == arg) {
      return true;
    }
    MS_EXCEPTION(ValueError)
      << "The parameter " << arg->ToString() << " , its name '" << param_name
      << "' already exists. Please set a unique name for the parameter."
      << "\nFor more details with the name of parameter, please refer to "
      << "https://mindspore.cn/search?inputValue=Please%20set%20a%20unique%20name%20for%20the%20parameter";
  }

  param_names->emplace(param_name, arg);
  return true;
}

// Check parameters in cell and out cell has same name
void CheckDuplicatedParameterName(const AnfNodePtrList &parameters) {
  std::map<std::string, ValuePtr> param_names;
  for (auto &parameter : parameters) {
    auto param_node = parameter->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_node);

    if (!param_node->has_default()) {
      continue;
    }

    (void)CheckDuplicatedParameterName(param_node->default_param(), &param_names);
  }
}

// Exist ScalarAdd ScalarSub etc OPS which will backoff to CPU
bool IsNeedBackoffGraph(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return(), SuccDeeperSimple);
  return std::any_of(node_list.begin(), node_list.end(),
                     [](const AnfNodePtr &node) { return common::AnfAlgo::IsNodeMutableScalar(node); });
}

FuncGraphPtr ConstructGraphForEval(const ValuePtr &func, const abstract::AbstractBasePtrList &args_abs) {
  MS_EXCEPTION_IF_NULL(func);
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
using abstract::AnalysisResult;
using mindspore::abstract::AnalysisContextPtr;

// Whether this process in a MindSpore cluster.
static bool is_cluster_initialized = false;

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

abstract::AnalysisResult AbstractAnalyze(const abstract::AnalysisEnginePtr &engine, const FuncGraphPtr &func_graph,
                                         const abstract::AbstractBasePtrList &args_abs, bool is_load_resource,
                                         bool clear) {
  MS_LOG(DEBUG) << "AbstractAnalyze start";
  py::gil_scoped_acquire gil;
  MS_EXCEPTION_IF_NULL(engine);
  if (clear || is_load_resource) {
    auto manager = engine->func_graph_manager();
    MS_EXCEPTION_IF_NULL(manager);
    engine->Clear();
    static const auto enable_eliminate_unused_element = (common::GetCompileConfig("ENABLE_DDE") != "0");
    for (auto &node : manager->all_nodes()) {
      MS_EXCEPTION_IF_NULL(node);
      // Handle previous inferred value for CNode if is loaded from MindIR
      // If the primitive is not defined in front end, keep the inferred value loaded from MindIR.
      if (is_load_resource) {
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
  MS_EXCEPTION_IF_NULL(func);
  MS_LOG(DEBUG) << "Start AbstractAnalyze for callable: " << func->ToString();
  auto infer_graph = func->isa<FuncGraph>() ? func->cast<FuncGraphPtr>() : ConstructGraphForEval(func, args_abs);

  auto top_graph = parse::Parser::GetTopFuncGraph();
  auto manager =
    top_graph == nullptr || top_graph->manager() == nullptr ? Manage(infer_graph, true) : top_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(infer_graph);

  auto engine = std::make_shared<abstract::AnalysisEngine>(abstract::GetPrimEvaluatorConstructors(), manager);
  engine->set_top_func_graph(parse::Parser::GetTopFuncGraph());
  engine->set_check_side_effect(true);
  return AbstractAnalyze(engine, infer_graph, args_abs, false, clear);
}

abstract::AnalysisResult AbstractAnalyzeWithResourceClean(const ValuePtr &func,
                                                          const abstract::AbstractBasePtrList &args_abs) {
  auto infer_graph = func->isa<FuncGraph>() ? func->cast<FuncGraphPtr>() : ConstructGraphForEval(func, args_abs);

  ResourcePtr resource = std::make_shared<Resource>();
  resource->set_func_graph(infer_graph);

  auto engine = resource->engine();
  engine->set_top_func_graph(parse::Parser::GetTopFuncGraph());
  auto res = AbstractAnalyze(engine, infer_graph, args_abs, false, true);

  JitExecutorPy::GetInstance()->CleanCompileRes(resource);
  return res;
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
  specializer.SpecializeFuncGraph();
  MS_LOG(DEBUG) << "ProgramSpecialize end";
  return result;
}

void SetMindIRLoadFlag(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  FuncGraphPtr loaded_graph = nullptr;
  const auto &all_graphs = manager->func_graphs();
  for (auto &graph : all_graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    if (graph->has_attr("is_load")) {
      loaded_graph = graph;
      resource->set_is_load(true);
      return;
    }
  }
}

FuncGraphPtr Renormalize(const ResourcePtr &resource, const FuncGraphPtr &func_graph,
                         const abstract::AbstractBasePtrList &args_abs) {
  MS_EXCEPTION_IF_NULL(resource);
  MS_LOG(DEBUG) << "Renormalize start";
  auto engine = resource->engine();
  engine->set_top_func_graph(resource->func_graph());

  abstract::AnalysisResult result;
  {
    MsProfileStatGuard stat_guard("renormalize.infer");
    result = AbstractAnalyze(engine, func_graph, args_abs, resource->is_load(), true);
  }
  FuncGraphPtr res;
  {
    MsProfileStatGuard stat_guard("renormalize.specialize");
    res = ProgramSpecialize(engine, func_graph, result.context);
    resource->set_func_graph(res);
  }

  MS_LOG(DEBUG) << "Renormalize end";
  return res;
}

namespace {
// Get entry function/class.method name.
std::string GetFunctionName(const py::object &input) {
  // Get Cell.construct() or @jit function name.
  std::string function_name;
  if (py::hasattr(input, parse::PYTHON_PARSE_METHOD)) {
    // The class type string format is like: <class 'x.x.xxx'>
    std::string class_type_name = py::cast<std::string>(py::str(input.get_type()));
    constexpr auto class_type_prefix_len = 8;  // <class '
    constexpr auto class_type_suffix_len = 2;  // '>
    const auto class_type_len = class_type_name.length();
    // Exclude class prefix and suffix.
    auto class_name =
      class_type_name.substr(class_type_prefix_len, class_type_len - class_type_prefix_len - class_type_suffix_len);
    auto method_name = py::cast<std::string>(input.attr(parse::PYTHON_PARSE_METHOD));
    function_name = class_name + '.' + method_name;
  } else if (py::hasattr(input, "__jit_function__") && py::hasattr(input, "__name__")) {
    // Get @jit decorated function name.
    auto jit_name = py::cast<std::string>(input.attr("__name__"));
    function_name = jit_name;
  } else {
    MS_EXCEPTION(NotSupportError) << "Entry Python object for JIT is invalid.\ninput: " << py::str(input);
  }
  MS_LOG(DEBUG) << "function_name: " << function_name;
  return function_name;
}

// Update top graph name.
void UpdateTopGraphDebugInfo(const FuncGraphPtr &func_graph, const py::object &input) {
  auto function_name = GetFunctionName(input);
  // Normalize the name.
  std::replace(function_name.begin(), function_name.end(), '.', '_');
  std::replace(function_name.begin(), function_name.end(), '<', '_');
  std::replace(function_name.begin(), function_name.end(), '>', '_');

  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(func_graph->debug_info());
  func_graph->debug_info()->set_name(function_name);
}

struct FuncArgSpec {
  AnfNodePtrList args_;
  ParameterPtr varargs_{nullptr};
  AnfNodePtrList kwonlyargs_;
  ParameterPtr varkw_{nullptr};
};

void MakeDefaultValue(const py::dict &defaults, const std::string &arg_name,
                      std::vector<std::string> *namelist_for_default_value, std::vector<AnfNodePtr> *default_values) {
  (void)namelist_for_default_value->emplace_back(arg_name);
  if (defaults.contains(arg_name)) {
    AnfNodePtr arg_node = NewValueNode(parse::data_converter::PyDataToValue(defaults[py::str(arg_name)]));
    (void)default_values->emplace_back(arg_node);
  } else {
    (void)default_values->emplace_back(NewValueNode(kNull));
  }
}

bool CheckIgnoreSelfParam(const py::object &input) {
  auto input_type = parse::data_converter::GetObjType(input);
  if (input_type == parse::ResolveType::RESOLVE_TYPE_CLASS_INSTANCE) {
    return true;
  }
  if (input_type == parse::ResolveType::RESOLVE_TYPE_METHOD) {
    py::object method_object = python_adapter::GetPyObjAttr(input, parse::PYTHON_GET_METHOD_SELF_CLASS);
    if (!py::isinstance<py::none>(method_object)) {
      return true;
    }
  }
  return false;
}

FuncArgSpec GetFuncArgSpec(const FuncGraphPtr &func_graph, const py::object &input) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto func = input;
  if (py::hasattr(input, parse::PYTHON_PARSE_METHOD)) {
    auto func_name = py::cast<std::string>(input.attr(parse::PYTHON_PARSE_METHOD));
    func = input.attr(func_name.c_str());
  }
  py::tuple obj_tuple =
    python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE, "get_arg_spec_and_default_values", func);
  auto full_arg_spec = obj_tuple[0];
  py::dict defaults = obj_tuple[1];
  std::vector<std::string> namelist_for_default_value;
  std::vector<AnfNodePtr> default_values;
  FuncArgSpec arg_spec;
  bool ignore_self_param = CheckIgnoreSelfParam(input);
  if (py::hasattr(full_arg_spec, "args")) {
    for (const auto &arg : full_arg_spec.attr("args")) {
      auto arg_name = py::cast<std::string>(arg);
      if (arg_name == "self" && ignore_self_param) {
        continue;
      }
      auto para = func_graph->add_parameter();
      para->set_is_top_graph_param(true);
      para->set_name(arg_name);
      (void)arg_spec.args_.emplace_back(para);
      MakeDefaultValue(defaults, arg_name, &namelist_for_default_value, &default_values);
    }
  }

  if (py::hasattr(full_arg_spec, "varargs")) {
    auto varargs = full_arg_spec.attr("varargs");
    if (!py::isinstance<py::none>(varargs)) {
      arg_spec.varargs_ = func_graph->add_parameter();
      arg_spec.varargs_->set_is_top_graph_param(true);
      auto arg_name = py::cast<std::string>(varargs);
      arg_spec.varargs_->set_name(arg_name);
      func_graph->set_has_vararg(true);
      MakeDefaultValue(defaults, arg_name, &namelist_for_default_value, &default_values);
    }
  }

  if (py::hasattr(full_arg_spec, "kwonlyargs")) {
    for (const auto &arg : full_arg_spec.attr("kwonlyargs")) {
      auto para = func_graph->add_parameter();
      para->set_is_top_graph_param(true);
      auto arg_name = py::cast<std::string>(arg);
      para->set_name(arg_name);
      (void)arg_spec.kwonlyargs_.emplace_back(para);
      MakeDefaultValue(defaults, arg_name, &namelist_for_default_value, &default_values);
    }
    func_graph->set_kwonlyargs_count(SizeToInt(arg_spec.kwonlyargs_.size()));
  }

  if (py::hasattr(full_arg_spec, "varkw")) {
    auto varkw = full_arg_spec.attr("varkw");
    if (!py::isinstance<py::none>(varkw)) {
      arg_spec.varkw_ = func_graph->add_parameter();
      arg_spec.varkw_->set_is_top_graph_param(true);
      auto arg_name = py::cast<std::string>(varkw);
      arg_spec.varkw_->set_name(arg_name);
      func_graph->set_has_kwarg(true);
      MakeDefaultValue(defaults, arg_name, &namelist_for_default_value, &default_values);
    }
  }
  func_graph->SetDefaultValues(namelist_for_default_value, default_values);
  return arg_spec;
}

void BuildTopGraph(const FuncGraphPtr &func_graph, const py::object &input,
                   const abstract::AbstractBasePtrList &args_abs) {
  ScopeGuard scope_guard(kDefaultScope);
  // Make Resolve for user top graph 'input'.
  auto function_name = GetFunctionName(input);
  parse::NameSpacePtr name_space =
    std::make_shared<parse::NameSpace>(parse::RESOLVE_NAMESPACE_NAME_ENTRY, py::str(function_name), input);
  parse::SymbolPtr symbol = std::make_shared<parse::Symbol>(function_name);
  MS_LOG(DEBUG) << "name_space: " << name_space->ToString() << ", symbol: " << symbol->ToString();
  ValueNodePtr module_node = NewValueNode(name_space);
  ValueNodePtr symbol_node = NewValueNode(symbol);

  bool contains_value_any = false;
  ValuePtrList args_value_list;
  (void)std::transform(args_abs.cbegin(), args_abs.cend(), std::back_inserter(args_value_list),
                       [&contains_value_any](const AbstractBasePtr &abs) {
                         auto res = abs->BuildValue();
                         if (res->isa<ValueAny>()) {
                           contains_value_any = true;
                         }
                         return res;
                       });
  CNodePtr resolve_node;
  if (contains_value_any) {
    resolve_node = func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimResolve), module_node, symbol_node});
  } else {
    ValueNodePtr args_node = NewValueNode<ValuePtrList>(args_value_list);
    resolve_node = func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimResolve), module_node, symbol_node, args_node});
  }

  auto arg_spec = GetFuncArgSpec(func_graph, input);
  bool need_unpack = false;
  if (func_graph->has_vararg() || func_graph->has_kwarg() || func_graph->kwonlyargs_count() > 0) {
    need_unpack = true;
  }
  // Call user top graph in top graph.
  AnfNodePtrList inputs;
  if (!need_unpack) {
    (void)inputs.emplace_back(resolve_node);
    std::copy(func_graph->parameters().cbegin(), func_graph->parameters().cend(), std::back_inserter(inputs));
  } else {
    (void)inputs.emplace_back(NewValueNode(prim::kPrimDoUnpackCall));
    (void)inputs.emplace_back(resolve_node);
    if (!arg_spec.args_.empty()) {
      AnfNodePtrList args_inputs = {NewValueNode(prim::kPrimMakeTuple)};
      std::copy(arg_spec.args_.cbegin(), arg_spec.args_.cend(), std::back_inserter(args_inputs));
      (void)inputs.emplace_back(func_graph->NewCNodeInOrder(args_inputs));
    }
    if (arg_spec.varargs_ != nullptr) {
      (void)inputs.emplace_back(arg_spec.varargs_);
    }
    if (arg_spec.varkw_ != nullptr) {
      (void)inputs.emplace_back(arg_spec.varkw_);
    }
    if (!arg_spec.kwonlyargs_.empty()) {
      AnfNodePtrList key_inputs = {NewValueNode(prim::kPrimMakeTuple)};
      AnfNodePtrList value_inputs = {NewValueNode(prim::kPrimMakeTuple)};
      for (const auto &kwonlyarg : arg_spec.kwonlyargs_) {
        auto param_kwonlyarg = kwonlyarg->cast<ParameterPtr>();
        MS_EXCEPTION_IF_NULL(param_kwonlyarg);
        (void)key_inputs.emplace_back(NewValueNode(param_kwonlyarg->name()));
        (void)value_inputs.emplace_back(kwonlyarg);
      }
      auto make_dict =
        func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimMakeDict), func_graph->NewCNodeInOrder(key_inputs),
                                     func_graph->NewCNodeInOrder(value_inputs)});
      (void)inputs.emplace_back(make_dict);
    }
  }
  auto output = func_graph->NewCNodeInOrder(inputs);
  MS_LOG(DEBUG) << "output: " << output->DebugString(AnfNode::DebugStringLevel::kLevel2);
  func_graph->set_output(output);
}
}  // namespace

bool BootstrapAction(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  TraceManager::OpenParserDebugInfoFlag();
  if (!resource->source_input()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Bootstrap error";
  }
  py::object input = resource->source_input();
  parse::Parser::InitParserEnvironment(input);
  parse::Parser::EnableDeferResolve(false);
  py::module path = py::module::import("os.path");
  auto dir = path.attr("dirname")(py::globals()["__file__"]).cast<std::string>();
  python_adapter::set_python_env_flag(true);
  python_adapter::SetPythonPath(dir);

  // Create fake top graph firstly.
  auto top_graph = std::make_shared<FuncGraph>();
  MS_EXCEPTION_IF_NULL(top_graph);
  auto is_top_graph = (py::hasattr(input, parse::PYTHON_PARSE_METHOD) || py::hasattr(input, "__jit_function__"));
  if (!is_top_graph) {
    MS_EXCEPTION(NotSupportError) << "Not supported Python object for JIT entry.\ninput: " << py::str(input);
  }
  UpdateTopGraphDebugInfo(top_graph, input);
  // Call the user top graph with its arguments.
  BuildTopGraph(top_graph, input, resource->args_abs());
  // Set the top graph.
  parse::Parser::UpdateTopFuncGraph(top_graph);
  resource->set_func_graph(top_graph);
  FuncGraphManagerPtr manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(top_graph);
  return true;
}

bool ParseAction(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  TraceManager::OpenParserDebugInfoFlag();
  if (!resource->source_input()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Parse error";
  }

  py::object input = resource->source_input();
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
    MS_LOG(INTERNAL_EXCEPTION) << "Attribute convert error with type: " << std::string(py::str(input));
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
    if (it->first.find("lazy_inline") != it->first.npos) {
      continue;
    }
    auto &graphs = it->second;
    MS_LOG(DEBUG) << "Start combine like graph:" << it->first << ", size:" << graphs.size();
    auto fg = graphs[0];
    FuncGraphVector func_graphs = {fg};
    Cloner cloner(func_graphs, false, false, true, MakeTraceInfo<TraceCopy>(), MakeTraceInfo<TraceCombileLikeGraphs>());
    cloner.Run();
    auto cloned_fg_iter = cloner.cloned_func_graphs().find(fg);
    if (cloned_fg_iter == cloner.cloned_func_graphs().end()) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, fg->return_node()) << "Clone func graph failed! " << fg->ToString();
    }
    auto base_graph = cloned_fg_iter->second;
    MS_LOG(DEBUG) << "Basegraph:" << base_graph->ToString();

    if (fg->parameter_obj_nodes().empty() || graphs.size() <= 1 || fg->has_attr(FUNC_GRAPH_FLAG_OUTPUT_RECOMPUTE) ||
        fg->stage() != -1) {
      continue;
    }
    auto &cloned_nodes = cloner.cloned_nodes();
    for (auto &fv : fg->parameter_obj_nodes()) {
      auto output = fg->output();
      MS_EXCEPTION_IF_NULL(output);
      TraceGuard guard(MakeTraceInfo<TraceCombileLikeGraphs>(output->debug_info()));
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
      MS_EXCEPTION_IF_NULL(fg->output());
      TraceGuard guard(MakeTraceInfo<TraceCopy>(fg->output()->debug_info()));
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
      MS_LOG(DEBUG) << "Combine graph newout:" << out->DebugString(AnfNode::DebugStringLevel::kLevel4);
    }
    MS_LOG(DEBUG) << "End combine graph:" << it->first;
  }
  return true;
}

namespace {
// Get all the trainable parameters of the reusable cell.
void GenerateTopGraphParams(const FuncGraphPtr &fg, std::vector<AnfNodePtr> *params,
                            const FuncGraphPtr &top_func_graph) {
  MS_LOG(DEBUG) << "enter GenerateTopGraphParams: " << fg->ToString();
  auto obj_value = fg->python_obj();
  MS_EXCEPTION_IF_NULL(obj_value);
  auto wrapper = dyn_cast_ptr<parse::PyObjectWrapper>(obj_value);
  MS_EXCEPTION_IF_NULL(wrapper);
  auto obj = wrapper->obj();
  auto trainable_parameters = py::getattr(obj, parse::CELL_PARAMETERS_AND_NAMES, py::none())();
  for (auto tr : trainable_parameters) {
    auto item = py::cast<py::tuple>(tr);
    auto value = item[1];
    auto par_name = item[0].cast<std::string>();
    auto parameter_name = py::getattr(value, "name", py::str(par_name)).cast<std::string>();
    auto exist_fv = top_func_graph->GetParameterByName(parameter_name);
    if (exist_fv) {
      params->push_back(exist_fv);
      MS_LOG(DEBUG) << "exist: " << parameter_name;
    } else {
      auto fv = top_func_graph->AddFvParameter(parameter_name, parse::GetParameterValue(value));
      auto context = parallel::ParallelContext::GetInstance();
      if (context != nullptr && fv->has_default()) {
        auto fv_abs = pipeline::GetDefaultValueAbstract(fv);
        context->ParallelParameterContextRestoreShape(top_func_graph, fv, fv_abs);
        fv->set_abstract(fv_abs);
      }
      MS_LOG(DEBUG) << "New: " << parameter_name;
      params->push_back(fv);
    }
  }
  MS_LOG(DEBUG) << "finish GenerateTopGraphParams: " << fg->ToString();
}

void UpdateCellFuncGraph(const FuncGraphPtr &func_graph, const FuncGraphPtr &reusing_graph,
                         const FuncGraphPtr &top_func_graph) {
  std::vector<AnfNodePtr> new_node_inputs;
  new_node_inputs.push_back(NewValueNode(reusing_graph));
  std::vector<AnfNodePtr> fvs;
  GenerateTopGraphParams(func_graph, &fvs, top_func_graph);
  (void)new_node_inputs.insert(new_node_inputs.end(), fvs.rbegin(), fvs.rend());
  auto params = func_graph->parameters();
  (void)new_node_inputs.insert(new_node_inputs.end(), params.begin(), params.end());
  AnfNodePtr out = func_graph->NewCNodeInOrder(new_node_inputs);
  MS_EXCEPTION_IF_NULL(func_graph->output());
  out->set_abstract(func_graph->output()->abstract());
  func_graph->set_output(out);
}

void GeneralizeReusingGraph(const FuncGraphPtr &func_graph, const FuncGraphPtr &top_func_graph) {
  FuncGraphPtr fg = func_graph;
  FuncGraphVector func_graphs = {fg};
  Cloner cloner(func_graphs, false, false, true, MakeTraceInfo<TraceCopy>(), MakeTraceInfo<TraceGraphReusing>());
  cloner.Run();
  auto cloned_fg_iter = cloner.cloned_func_graphs().find(fg);
  if (cloned_fg_iter == cloner.cloned_func_graphs().end()) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, fg->return_node()) << "Clone func graph failed! " << fg->ToString();
  }
  auto reusing_graph = cloned_fg_iter->second;
  auto &cloned_nodes = cloner.cloned_nodes();
  auto manager = fg->manager();
  std::vector<AnfNodePtr> fv_params;
  GenerateTopGraphParams(fg, &fv_params, top_func_graph);
  for (auto &fv : fv_params) {
    auto param = reusing_graph->InsertFrontParameter();
    const auto &top_param = fv->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(top_param);
    std::string name = "CR_" + top_param->name();
    if (param->debug_info() != nullptr) {
      param->debug_info()->set_name(name);
    }
    param->set_name(name);
    param->set_abstract(top_param->abstract());
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

  if (func_graph->has_attr(FUNC_GRAPH_FLAG_NO_INLINE)) {
    reusing_graph->set_flag(FUNC_GRAPH_FLAG_NO_INLINE, func_graph->has_flag(FUNC_GRAPH_FLAG_NO_INLINE));
  } else {
    reusing_graph->set_flag(FUNC_GRAPH_FLAG_NO_INLINE, true);
    reusing_graph->set_flag(FUNC_GRAPH_FLAG_CELL_REUSE, true);
  }

  // Update call nodes
  auto no_inline_flag = reusing_graph->has_flag(FUNC_GRAPH_FLAG_NO_INLINE);
  auto cnodes_index = fg->func_graph_cnodes_index();
  for (auto &cnode_index : cnodes_index) {
    MS_EXCEPTION_IF_NULL(cnode_index.first);
    auto old_cnode = cnode_index.first->first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(old_cnode);
    auto cell_func_graph = old_cnode->func_graph();
    MS_EXCEPTION_IF_NULL(cell_func_graph);
    UpdateCellFuncGraph(cell_func_graph, reusing_graph, top_func_graph);

    // optimize FuncGraph::scope() performance
    cell_func_graph->set_flag(FUNC_GRAPH_FLAG_NO_CHILD_GRAPH, no_inline_flag);
  }
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

void MergeClonedLazyInlineOrderGraph(const FuncGraphPtr &top_graph) {
  auto manager = top_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  HashMap<int, FuncGraphPtr> order_fgs;
  auto all_func_graphs = top_graph->func_graphs_used_total();
  for (auto &fg : all_func_graphs) {
    MS_EXCEPTION_IF_NULL(fg);
    auto order_value = fg->get_attr(FUNC_GRAPH_FLAG_CELL_LAZY_INLINE_ORDER);
    if (order_value == nullptr) {
      continue;
    }
    if (!fg->has_flag(FUNC_GRAPH_FLAG_CELL_LAZY_INLINE_ORDER_CLONE)) {
      continue;
    }
    auto order = GetValue<int>(order_value);
    auto iter = order_fgs.find(order);
    if (iter == order_fgs.end()) {
      order_fgs.emplace(order, fg);
      continue;
    }
    auto cnodes_index = fg->func_graph_cnodes_index();
    for (auto &cnode_index : cnodes_index) {
      MS_EXCEPTION_IF_NULL(cnode_index.first);
      auto call_node = cnode_index.first->first;
      manager->SetEdge(call_node, cnode_index.first->second, NewValueNode(iter->second));
    }
  }
}
}  // namespace

// Make the reusable cell to be the reusable function graph.
bool GraphReusingAction(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  bool cell_reused = false;
  auto func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  MergeClonedLazyInlineOrderGraph(func_graph);
  std::multimap<int, FuncGraphPtr> order_fgs;
  for (auto &fg : func_graph->func_graphs_used_total()) {
    auto order_value = fg->get_attr(FUNC_GRAPH_FLAG_CELL_LAZY_INLINE_ORDER);
    if (order_value == nullptr) {
      continue;
    }
    fg->erase_flag(FUNC_GRAPH_FLAG_CELL_LAZY_INLINE_ORDER);
    if (!fg->has_attr(FUNC_GRAPH_FLAG_NO_INLINE)) {
      cell_reused = true;
    }
    order_fgs.insert(std::make_pair(GetValue<int>(order_value), fg));
  }
  for (auto it = order_fgs.rbegin(); it != order_fgs.rend(); ++it) {
    MS_LOG(INFO) << "Lazy_inline graph: " << it->second->ToString() << " , order: " << it->first;
    GeneralizeReusingGraph(it->second, func_graph);
  }
  if (!cell_reused) {
    return true;
  }

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  const bool force_no_inline = runtime::IsDisableRuntimeConfig(runtime::kRuntimeInline);
  context->SetCellReuseLevel(CellReuseLevel::kNoCellReuse);

  MS_LOG(INFO) << "Cell reuse(@lazy_inline) actually takes effect.";
  auto cell_reuse_level = CellReuseLevel::kLazyInline;
  if (force_no_inline) {
    cell_reuse_level = CellReuseLevel::kNoInline;
  }
  context->SetCellReuseLevel(cell_reuse_level);

  return true;
}

// Used for excluding the func graphs in VMap.
bool UsedByVmap(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
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
  static const bool enable_pre_lift = (common::GetCompileConfig("PRE_LIFT") == "1");
  if (!enable_pre_lift) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
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

bool EventMethodAction(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (resource->manager() == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Event Method failed, manager is null";
  }
  auto func_graph = resource->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Event Method failed, graph is null";
  }
  pipeline::EventMethod(func_graph);
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
  auto res = std::make_shared<abstract::AbstractRefTensor>(abs_tensor, ref_key);
  res->cast<abstract::AbstractRefPtr>()->set_is_parameter(true);
  return res;
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

bool IsCreatedByViewOp(const AnfNodePtr &node) {
  if (node->isa<CNode>()) {
    auto cnode = node->cast_ptr<CNode>();
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim != nullptr && prim->graph_view_prim()) {
      return true;
    }
  }
  auto abstract = node->abstract();
  if (abstract != nullptr && abstract->isa<abstract::AbstractRefTensor>()) {
    const auto ref = abstract->cast_ptr<abstract::AbstractRefTensor>();
    return ref->is_view();
  }
  return false;
}

bool IsViewInplaceNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &prim = GetCNodePrimitive(node);
  if (prim != nullptr && prim->inplace_prim()) {
    auto inplace_node = node->cast_ptr<CNode>();
    const AnfNodePtrList &input_nodes = inplace_node->inputs();
    const auto &rw_write_index = prim->rw_write_input_indexes();
    auto iter = std::find_if(rw_write_index.begin(), rw_write_index.end(), [&input_nodes](const size_t &index) {
      if (index + 1 >= input_nodes.size()) {
        MS_LOG(ERROR) << "Invalid rw_write index: " << index;
        return false;
      }
      return IsCreatedByViewOp(input_nodes[index + 1]);
    });
    if (iter != rw_write_index.end()) {
      MS_LOG(INFO) << "Found a view+inplace node. Inplace node: " << inplace_node->DebugString()
                   << ", and one of its input is created by view op: " << input_nodes[(*iter) + 1]->DebugString();
      return true;
    }
  }
  return false;
}

bool FindViewInplaceNode(const AnfNodePtrList &nodes) {
  bool has_view_inplace = false;
  std::vector<std::string> loc_str_list{};
  for (auto &node : nodes) {
    if (IsViewInplaceNode(node)) {
      has_view_inplace = true;
      auto location_str = trace::GetDebugInfoStr(node->debug_info());
      if (std::find(loc_str_list.begin(), loc_str_list.end(), location_str) != loc_str_list.end()) {
        continue;
      }
      MS_LOG(WARNING)
        << "There is an in-place modification to a Tensor view that requires gradients. However, computing gradients "
           "for in-place modified Tensor views is still an experimental feature, and the results may be inaccurate. It "
           "is recommended to replace the Tensor view here with a non-view Tensor. The code location is as follows:\n"
        << location_str;
      (void)loc_str_list.emplace_back(location_str);
    }
  }
  return has_view_inplace;
}

FuncGraphPtr GetFuncFromAbstract(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  auto func_graph_abstract = dyn_cast<abstract::FuncGraphAbstractClosure>(abs);
  MS_EXCEPTION_IF_NULL(func_graph_abstract);
  if (!func_graph_abstract->specialized()) {
    MS_LOG(DEBUG) << "Unspecialized func graph, partial abs: " << abs->ToString()
                  << ", partial fn abs: " << func_graph_abstract->ToString();
    return nullptr;
  } else {
    return func_graph_abstract->func_graph();
  }
}

FuncGraphPtr GetForwardGraphFromJNode(const AnfNodePtr &j_node) {
  MS_EXCEPTION_IF_NULL(j_node);
  if (j_node->cast<CNodePtr>() == nullptr) {
    return nullptr;
  }
  AnfNodePtr j_param = j_node->cast<CNodePtr>()->input(1);
  AbstractBasePtr abs = j_param->abstract();
  FuncGraphPtr forward_graph = nullptr;
  if (IsValueNode<FuncGraph>(j_param)) {
    forward_graph = GetValueNode<FuncGraphPtr>(j_param);
  } else if (abs != nullptr && abs->isa<abstract::FuncGraphAbstractClosure>()) {
    forward_graph = GetFuncFromAbstract(j_param->abstract());
  } else {
    MS_LOG(DEBUG) << "Unknown node: " << j_param->DebugString()
                  << ", abstract: " << (j_param != nullptr ? j_param->abstract()->ToString() : "NULL");
  }
  return forward_graph;
}

bool RequireJitGrad(const std::string &phase) {
  return phase.find("export") != 0 && pynative::GradState::Get().RequiresGrad();
}

void SetViewInplaceGradFlag(const ResourcePtr &resource) {
  // pynative+jit
  if (RequireJitGrad(PhaseManager::GetInstance().phase())) {
    const auto &all_nodes = resource->manager()->all_nodes();
    if (FindViewInplaceNode({all_nodes.begin(), all_nodes.end()})) {
      MS_LOG(DEBUG) << "Found a view+inplace node. Grad run new pass.";
      resource->set_is_pynative_grad_view_inplace(true);
    }
    return;
  }
  // Graph mode
  AnfNodePtrList j_nodes;
  for (const auto &node : resource->manager()->all_nodes()) {
    if (IsPrimitiveCNode(node, prim::kPrimJ)) {
      j_nodes.push_back(node);
    }
  }
  if (j_nodes.empty()) {
    MS_LOG(DEBUG) << "No J node is found, so no need to scan for view+inplace op";
    return;
  }
  for (const auto &j_node : j_nodes) {
    FuncGraphPtr forward_graph = GetForwardGraphFromJNode(j_node);
    if (forward_graph == nullptr) {
      MS_LOG(DEBUG) << "Cannot find forward graph from j node: " << j_node->DebugString();
      continue;
    }
    if (forward_graph->get_return() == nullptr) {
      MS_LOG(DEBUG) << "The return node of FuncGraph is NULL! fg: " << forward_graph->ToString();
      continue;
    }
    const AnfNodePtrList &all_nodes = mindspore::TopoSort(forward_graph->get_return(), SuccDeeperSimple);
    if (FindViewInplaceNode(all_nodes)) {
      j_node->set_user_data("has_view_inplace_grad", std::make_shared<bool>(true));
    }
  }
}
}  // namespace

AnfNodePtrList AllForwardNodes(const ResourcePtr &resource) {
  AnfNodePtrList all_forward_nodes;
  AnfNodePtrList j_nodes;
  for (const auto &node : resource->manager()->all_nodes()) {
    if (IsPrimitiveCNode(node, prim::kPrimJ)) {
      j_nodes.push_back(node);
    }
  }
  if (j_nodes.empty()) {
    MS_LOG(DEBUG) << "No J node is found";
    return all_forward_nodes;
  }
  for (const auto &j_node : j_nodes) {
    FuncGraphPtr forward_graph = GetForwardGraphFromJNode(j_node);
    if (forward_graph == nullptr) {
      MS_LOG(DEBUG) << "Cannot find forward graph from j node: " << j_node->DebugString();
      continue;
    }
    if (forward_graph->get_return() == nullptr) {
      MS_LOG(DEBUG) << "The return node of FuncGraph is NULL! fg: " << forward_graph->ToString();
      continue;
    }
    const AnfNodePtrList &all_nodes_from_one_jnode = mindspore::TopoSort(forward_graph->get_return(), SuccDeeperSimple);
    all_forward_nodes.insert(all_forward_nodes.end(), all_nodes_from_one_jnode.begin(), all_nodes_from_one_jnode.end());
  }
  return all_forward_nodes;
}

bool TypeInferenceAction(const ResourcePtr &resource) {
  EventMessage::PrintCompileStatusMessage("Start performing static analysis and type inference.");
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
  uint64_t start_time = profiler::GetClockSyscnt();
  AnalysisResult result;
  {
    MsProfileStatGuard stat_guard("type_inference.infer");
    result = AbstractAnalyze(engine, resource->func_graph(), GetArgsAbs(resource), resource->is_load());
  }
  (void)profiler::CollectHostInfo(kCompiler, kTypeInference, kAbstractAnalyze, start_time, profiler::GetClockSyscnt(),
                                  0);
  // Specialize
  start_time = profiler::GetClockSyscnt();
  FuncGraphPtr new_fg;
  {
    MsProfileStatGuard stat_guard("type_inference.specialize");
    new_fg = ProgramSpecialize(resource->engine(), result.context->func_graph(), result.context);
  }
  MS_EXCEPTION_IF_NULL(new_fg);
  (void)profiler::CollectHostInfo(kCompiler, kTypeInference, kProgramSpecialize, start_time, profiler::GetClockSyscnt(),
                                  0);
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

  UpdateFuncGraphParameter(new_fg, resource->arguments(), resource->real_arguments());
  CheckDuplicatedParameterName(new_fg->parameters());
  SetMindIRLoadFlag(resource);
  SetViewInplaceGradFlag(resource);
  abstract::ReOrderParameterForwardHookNodes(new_fg, resource);

  MS_LOG(DEBUG) << "End graph: " << new_fg->ToString() << ", return: " << new_fg->get_return()->DebugString(true);
  return true;
}

bool OptimizeAction(const ResourcePtr &resource, const std::vector<PassItem> &passes) {
  MS_EXCEPTION_IF_NULL(resource);
  size_t counter = 0;
  for (auto &pass : passes) {
    std::string pass_name = pass.first;
    MSTX_RANGE_GUARD(pass_name.c_str(), nullptr, profiler::MSTX_DOMAIN_MODEL_PREPARATION);
    MsProfileStatGuard stat_guard(std::move(pass_name), "compile_irpass", true);
    ProcessStatus::GetInstance().RecordStart(pass.first);
    uint64_t start_time = profiler::GetClockSyscnt();
    auto profile_context = MsProfile::GetProfile()->Step(pass.first);
    auto pass_func = [&pass, &resource, &counter]() {
      MS_LOG(INFO) << "Pass " << pass.first << " start ...";
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
      MS_LOG(INFO) << "Pass " << pass.first << " end.";
    };
    ProfileExecute(profile_context, pass_func);
    (void)profiler::CollectHostInfo(kCompiler, kOptimize, pass.first, start_time, profiler::GetClockSyscnt(), 0);
    ProcessStatus::GetInstance().RecordEnd();
  }

  return true;
}

bool OptInlineAction(const ResourcePtr &resource) {
  if (parallel::ParallelContext::GetInstance()->parallel_mode() == "semi_auto_parallel" ||
      parallel::ParallelContext::GetInstance()->parallel_mode() == "auto_parallel") {
    return OptimizeAction(resource, kInlinePasses);
  }
  return true;
}

bool OptAddAttrWithInlineAction(const ResourcePtr &resource) {
  return OptimizeAction(resource, kAddAttrWithInlinePass);
}

bool VmOptimizeAction(const ResourcePtr &resource) {
  EventMessage::PrintCompileStatusMessage("Start performing graph optimization.");
  bool ret;
  auto custom_passes = opt::PassConfigure::Instance().GetPasses();
  if (custom_passes.empty()) {
    ret = OptimizeAction(resource, kVmPasses);
  } else {
    MS_LOG(INFO) << "Use custom passes, size: " << custom_passes.size();
    ret = OptimizeAction(resource, custom_passes);
  }

  TraceManager::CloseParserDebugInfoFlag();
  return ret;
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
  if (!RequireJitGrad(PhaseManager::GetInstance().phase())) {
    return true;
  }
  return pynative::PyNativeAdapter::GetJitBpropGraph(resource, PhaseManager::GetInstance().phase());
}

bool RewriterAfterOptAPassAfterJitBprop(const ResourcePtr &resource) {
  // This function is only used to convert unsupported syntax into PyExecute nodes through Fallback,
  // when the forward graph is decorated with 'jit', and is derivative in pynative mode.
  const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() >= kCompatible);
  if (!allow_fallback_runtime) {
    return true;
  }
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

bool OptAfterJitGrad(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (resource->manager() == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "PynativeElimOpt error, manager is null.";
  }
  if (resource->func_graph() == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "PynativeElimOpt error, graph is null.";
  }
  return OptAfterJitGradPass(resource);
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
        MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, cnode)
          << "SwitchLayer input2 should be make_tuple, but got: " << make_tuple->DebugString();
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
    if (common::AnfAlgo::HasIncorporateCallNode(cnode)) {
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

bool SetModeForControlFlow(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &all_nodes) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  MS_EXCEPTION_IF_NULL(func_graph);
  auto set_ctx = [&context_ptr](bool task_sink, bool is_multi_graph_sink, bool enable_loop_sink) {
    context_ptr->set_param<bool>(MS_CTX_ENABLE_TASK_SINK, task_sink);
    context_ptr->set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, is_multi_graph_sink);
    context_ptr->set_param<bool>(MS_CTX_ENABLE_LOOP_SINK, enable_loop_sink);
  };
  // GRAPH | Closure\ENV\While scenario : KernelByKernel path in MindRT.
  auto graphs = func_graph->func_graphs_used_total();
  (void)graphs.insert(func_graph);
  bool exist_control_flow = ExistControlFlow(func_graph);
  bool exist_func = exist_control_flow && HasIncorporateCall(all_nodes);
  bool is_ge = AnfAlgo::IsBackendGe();
  if (exist_func) {
    if (is_ge) {
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
    if (is_ge) {
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
    if (exist_control_flow && !is_ge) {
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

bool IsCellReuse(const AnfNodePtr &input) {
  if (IsValueNode<FuncGraph>(input)) {
    auto fg = GetValueNode<FuncGraphPtr>(input);
    MS_EXCEPTION_IF_NULL(fg);
    if (fg->has_flag(FUNC_GRAPH_FLAG_CELL_REUSE)) {
      return true;
    }
  }
  return false;
}

bool CantInlineCellReuse(const FuncGraphPtr &fg) {
  MS_EXCEPTION_IF_NULL(fg);
  if (!fg->has_flag(FUNC_GRAPH_FLAG_CELL_REUSE)) {
    return false;
  }
  MS_LOG(INFO) << "Cell reuse graph: " << fg->ToString();
  // cell reuse func graph has switch
  if (!fg->switch_nodes().empty()) {
    MS_LOG(INFO) << "Set no inline because cell reuse graph has switch, " << fg->ToString();
    return true;
  }
  // cell reuse sub graph has switch or cell reuse
  for (auto &sub_graph : fg->func_graphs_used_total()) {
    if (!sub_graph->switch_nodes().empty() || sub_graph->has_flag(FUNC_GRAPH_FLAG_CELL_REUSE)) {
      MS_LOG(INFO) << "Set no inline because cell reuse sub graph has switch or nested cell reuse, "
                   << sub_graph->ToString();
      return true;
    }
  }
  MS_EXCEPTION_IF_NULL(fg->get_return());
  MS_LOG(INFO) << "Graph return node: " << fg->get_return()->DebugString();
  for (auto input : fg->get_return()->inputs()) {
    if (IsPrimitiveCNode(input, prim::kPrimMakeTuple)) {
      for (auto inner_input : input->cast<CNodePtr>()->inputs()) {
        if (IsPrimitiveCNode(inner_input, prim::kPrimMakeTuple)) {
          MS_LOG(INFO) << "Set no inline because cell reuse graph has nested make tuple.";
          return true;
        }
      }
    }
  }
  MS_LOG(INFO) << "Graph: " << fg->ToString() << " can be lazyinlined.";
  return false;
}

void ProcessCanNotInline(const FuncGraphPtr &func_graph, const std::shared_ptr<MsContext> &context_ptr) {
  auto graphs = func_graph->func_graphs_used_total();
  (void)graphs.insert(func_graph);
  bool exist_while =
    std::any_of(graphs.cbegin(), graphs.cend(), [](const FuncGraphPtr &fg) { return fg->recursive(); });
  if (exist_while && context_ptr->CellReuseLevel() == CellReuseLevel::kLazyInline) {
    MS_LOG(INFO) << "Set no inline because graph has while.";
    context_ptr->SetCellReuseLevel(CellReuseLevel::kNoInline);
  }
  if (std::any_of(graphs.cbegin(), graphs.cend(), CantInlineCellReuse)) {
    MS_LOG(INFO) << "Set no inline because cell reuse graph has switch or nested cell reuse.";
    context_ptr->SetCellReuseLevel(CellReuseLevel::kNoInline);
  }
  if (!runtime::IsEnableRuntimeConfig(runtime::kRuntimeInline)) {
    const auto &all_nodes = TopoSort(func_graph->return_node(), SuccDeeperSimple, AlwaysInclude);
    size_t micro_num = 0;
    for (auto &node : all_nodes) {
      if (!node->isa<CNode>()) {
        continue;
      }
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (IsCellReuse(cnode->input(0))) {
        micro_num++;
      }
    }
    auto parallel_context = parallel::ParallelContext::GetInstance();
    MS_EXCEPTION_IF_NULL(parallel_context);
    auto stages = parallel_context->pipeline_stage_split_num();
    if (stages <= 1) {
      return;
    }
    MS_LOG(INFO) << "Cell reuse micro num: " << micro_num;
    auto not_ge = !AnfAlgo::IsBackendGe();
    if (micro_num > kLazyInlineThershold && not_ge) {
      MS_LOG(INFO) << "Set no inline because cell reuse micro num is greater than " << kLazyInlineThershold
                   << ", micro num: " << micro_num;
      context_ptr->SetCellReuseLevel(CellReuseLevel::kNoInline);
    }
  }
}

void SetRunMode(const FuncGraphPtr &func_graph, std::string *kbk_reason) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  MS_EXCEPTION_IF_NULL(func_graph);
  auto set_ctx = [&context_ptr](bool task_sink, bool is_multi_graph_sink, bool enable_loop_sink) {
    context_ptr->set_param<bool>(MS_CTX_ENABLE_TASK_SINK, task_sink);
    context_ptr->set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, is_multi_graph_sink);
    context_ptr->set_param<bool>(MS_CTX_ENABLE_LOOP_SINK, enable_loop_sink);
  };
  ProcessCanNotInline(func_graph, context_ptr);
  auto jit_level = context_ptr->GetJitLevel();
  func_graph->set_attr(kAttrJitLevel, MakeValue<std::string>(jit_level));
  auto jit_config = PhaseManager::GetInstance().jit_config();
  jit_config[kAttrJitLevel] = jit_level;
  graphkernel::GraphKernelFlags::SaveJitConfig(jit_config);

  const bool is_jit = IsJit();
  const auto &device_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (!is_jit && device_target != kAscendDevice) {
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
  if (common::AnfAlgo::IsDynamicGraph(func_graph) && (context_ptr->backend_policy() != "ge")) {
    if (kbk_reason != nullptr) {
      *kbk_reason =
        "Run graph mode with kernel by kernel because graph exist dynamic shape. set env variable "
        "'MS_DEV_SAVE_GRAPHS=2' to check graph irs.";
      MS_LOG(INFO) << *kbk_reason;
    }
    set_ctx(false, false, false);
    return;
  }

  // GRAPH | Dynamic Scalar : Dynamic scalar ops in graph.
  if (IsNeedBackoffGraph(func_graph) && !common::AnfAlgo::IsDynamicGraph(func_graph)) {
    if (kbk_reason != nullptr) {
      *kbk_reason = "Run graph mode with kernel by kernel because graph exist dynamic scalar ops.";
      MS_LOG(INFO) << *kbk_reason;
    }
    set_ctx(false, false, false);
    return;
  }
  if (!SetModeForControlFlow(func_graph, all_nodes)) {
    return;
  }

  // GRAPH | normal network and if/for/switch scenario etc : MultiGraph path in MindRT.
  MS_LOG(INFO) << "Run graph mode with multi graph sink.";
  set_ctx(true, true, is_jit);
  return;
}

void OriginSetRunMode(const ResourcePtr &resource) {
  FuncGraphPtr func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::string backend = context_ptr->backend_policy();
  auto task_sink = context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
  if (func_graph->exist_multi_target() || !task_sink) {
    context_ptr->set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, false);
    context_ptr->set_param<bool>(MS_CTX_ENABLE_LOOP_SINK, false);
  } else if (IsJit()) {
    std::string device_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    auto manager = func_graph->manager();
    auto graphs = manager->func_graphs();
    if (graphs.size() > 1 && device_target == kAscendDevice) {
      MS_LOG(INFO) << "This func_graph has control flow nodes, owns " << graphs.size() << " subgraphs.";
    }
    bool exist_while =
      std::any_of(graphs.cbegin(), graphs.cend(), [](const FuncGraphPtr &fg) { return fg->recursive(); });
    if (device_target == kAscendDevice && backend != "vm" && !exist_while) {
      MS_LOG(INFO) << "Run graph mode with multigraph sink.";
      context_ptr->set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, true);
    } else {
      MS_LOG(INFO) << "Run graph mode with vm.";
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
    SetRunMode(resource->func_graph(), &kbk_reason);
  } else {
    OriginSetRunMode(resource);
  }
  auto is_task_sink = context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
  auto enable_hccl = context_ptr->get_param<bool>(MS_CTX_ENABLE_HCCL);
  if ((!is_task_sink ||
       (context_ptr->IsKByKExecutorMode() && common::AnfAlgo::IsDynamicGraph(resource->func_graph()))) &&
      enable_hccl && !common::UseHostCollective() && common::GetEnv(kSimulationLevel).empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Current execution mode is 'kernelbykernel', reason: " << kbk_reason
                               << ", but you're launching job using 'ranktable', which "
                                  "does not support 'kernelbykernel' mode.\n Please refer to link: "
                                  "https://www.mindspore.cn/tutorials/en/master/parallel/startup_method.html "
                                  "and use 'Dynamic cluster'(suggested) or 'mpirun' to launch your job.";
  }
}

bool TaskEmitAction(const ResourcePtr &resource) {
  EventMessage::PrintCompileStatusMessage("Start generating kernels.");
  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr func_graph = resource->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "TaskEmit args error";
  }
  if (func_graph->has_user_data("jit_config")) {
    PhaseManager::GetInstance().set_jit_config(
      *(func_graph->user_data<std::map<std::string, std::string>>("jit_config")));
  }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  context_ptr->Refresh();
  const auto &backend = context_ptr->backend_policy();
  auto is_jit = IsJit();
  if (is_jit && CheckGraphOutputConstOrParameter(func_graph)) {
    return true;
  }

  // In PyNative mode, multi target will generate in -1 shape in jit. But, jit in -1 shape will run as a call graph;
  // control flow not has flag kFlagJitCallGraph
  bool is_control_flow = !func_graph->func_graphs_used_total().empty();
  if (is_jit || (func_graph->has_flag(kFlagJitCallGraph) || is_control_flow)) {
    func_graph->SetMultiTarget();
  }

  if (backend != "ms" && backend != "ge") {
    resource->SetResult(kNoBackend, true);
    MS_LOG(INFO) << "No backend.";
    return true;
  }
  SetRunMode(resource);

  // In pyexecute kernel, the input data would be stored in user data which is a python object, this converter
  // is used to convert user data to device ptr in device address.
  pyexecute::set_pydata_converter(
    [](const py::object &obj, ValuePtr *value) { return parse::ConvertData(obj, value); });
  const auto &backend_jit_config = backend::BackendJitConfig::ParseBackendJitConfig();
  MS_LOG(INFO) << "Use the new backend.";
  auto backend_ret = backend::BackendManager::GetInstance().Build(resource->func_graph(), backend_jit_config,
                                                                  backend_jit_config.backend);
  resource->SetResult(kBuildBackendType, backend_ret.first);
  resource->SetResult(kBuildBackendOutput, backend_ret.second);
  return true;
}

bool ExecuteAction(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (IsJit() && CheckGraphOutputConstOrParameter(resource->func_graph())) {
    return true;
  }

  if (resource->HasResult(kNoBackend)) {
    MS_LOG(INFO) << "No backend.";
    return true;
  }
  if (!resource->HasResult(kBuildBackendType) || !resource->HasResult(kBuildBackendOutput)) {
    MS_LOG(INTERNAL_EXCEPTION) << "Execute args error";
  }
  auto backend_type = resource->GetResult(kBuildBackendType).cast<backend::BackendType>();
  auto backend_graph_id = resource->GetResult(kBuildBackendOutput).cast<backend::BackendGraphId>();
  // Construct the graph run function ptr.
  VmEvalPtr run = std::make_shared<VmEvalFunc>([backend_type, backend_graph_id](const VectorRef &args) -> BaseRef {
    MS_LOG(DEBUG) << "Execute args size " << args.size();
    VectorRef outputs;
    backend::BackendManager::GetInstance().Run(backend_type, backend_graph_id, args, &outputs);
    MS_LOG(DEBUG) << "out size " << outputs.size();
    if (outputs.empty()) {
      return VectorRef();
    } else {
      return outputs[0];
    }
  });
  resource->SetResult(kOutput, run);
  return true;
}

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

bool DatasetRepeatReaderOptAction(const ResourcePtr &resource) { return DatasetRepeatReaderOptPass(resource); }

bool PipelineSplitAction(const ResourcePtr &resource) { return PipelineSplitPass(resource); }

bool ParallelVirtualDatasetAction(const ResourcePtr &resource) { return ParallelVirtualDatasetPass(resource); }

bool AutoParallelSymbolWithReNormalizeAction(const ResourcePtr &resource) {
  return AutoParallelSymbolPassWithReNormalize(resource);
}

bool DetachBackwardAction(const ResourcePtr &resource) { return DetachBackward(resource); }

bool PipelineSchedulerAction(const ResourcePtr &resource) { return PipelineParallelScheduler(resource); }

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
    ModifyGraphs(resource->func_graph());
    resource->func_graph()->set_flag("generated_from_mindir_with_prim_func", true);
    (void)InferMindir(resource->func_graph(), args_abs_list, true);
  }
  return true;
}

static std::vector<ActionItem> CommonPipeline(bool trace_flag) {
  std::vector<ActionItem> actions;
  auto graph_executor = pipeline::GraphExecutorPy::GetInstance();
  MS_EXCEPTION_IF_NULL(graph_executor);
  const bool boost_infer = common::GetEnv("MS_DEV_BOOST_INFER") != "0";
  if (!trace_flag) {
    if (boost_infer) {
      // Bootstrap for JIT.
      (void)actions.emplace_back(std::make_pair(kBootstrap, BootstrapAction));
    } else {
      // Parse the python ast to ANF graph
      (void)actions.emplace_back(std::make_pair(kParse, ParseAction));

      // Resolve the python func
      (void)actions.emplace_back(std::make_pair(kSymbolResolve, SymbolResolveAction));

      // Notice: Temporary solution, to be implemented using Python Rewriter in the future.
      // Set mixed Precision flag in subgraph.
      static bool enable_set_mixed_precision_flag = (common::GetCompileConfig("AMP_ENABLE_ALL_FG") == "1");
      if (enable_set_mixed_precision_flag) {
        (void)actions.emplace_back(std::make_pair(kSetMixedPrecisionFlag, SetMixedPrecisionAction));
      }

      auto parallel_context = parallel::ParallelContext::GetInstance();
      MS_EXCEPTION_IF_NULL(parallel_context);
      auto parallel_mode = parallel_context->parallel_mode();
      const bool is_parallel_mode =
        parallel_mode == parallel::kSemiAutoParallel || parallel_mode == parallel::kAutoParallel;
      static const auto combine_like_graphs = (common::GetCompileConfig("COMBINE_LIKE_GRAPHS") == "1");
      static const auto force_disable_combine = (common::GetCompileConfig("COMBINE_LIKE_GRAPHS") == "0");
      if (!is_cluster_initialized && (!is_parallel_mode || combine_like_graphs) && !force_disable_combine) {
        (void)actions.emplace_back(std::make_pair(kCombineLikeGraphs, CombineLikeGraphs));
      }

      // Pre-Lift the func graphs.
      (void)actions.emplace_back(std::make_pair(kPreCConv, PreCConvAction));
    }
  }
  // Evaluate type and shape, and specialize.
  (void)actions.emplace_back(std::make_pair(kTypeInference, TypeInferenceAction));

  (void)actions.emplace_back(std::make_pair(kEventMethod, EventMethodAction));

  // Auto-monad for side-effects handling.
  (void)actions.emplace_back(std::make_pair(kAutoMonad, AutoMonadAction));

  (void)actions.emplace_back(std::make_pair(kGraphReusing, GraphReusingAction));

  // Do data structure simplifications and inline.
  (void)actions.emplace_back(std::make_pair(kInline, OptInlineAction));

  (void)actions.emplace_back(std::make_pair(kAddAttr, OptAddAttrWithInlineAction));

  (void)actions.emplace_back(std::make_pair("parallel-infer-symbol", AutoParallelSymbolWithReNormalizeAction));

  // Do prepositive auto parallel.
  (void)actions.emplace_back(std::make_pair(kPreAutoParallel, AutoParallelAction));
  // insert virtual dataset
  (void)actions.emplace_back(std::make_pair("insert-virtual-dataset", ParallelVirtualDatasetAction));
  (void)actions.emplace_back(std::make_pair("parallel-infer-symbol-second", AutoParallelSymbolWithReNormalizeAction));
  (void)actions.emplace_back(std::make_pair("dataset_repeat_opt", DatasetRepeatReaderOptAction));

  // Do PipelineSplit action.
  (void)actions.emplace_back(std::make_pair(kPipelineSplit, PipelineSplitAction));

  return actions;
}

std::vector<ActionItem> EraseParseActions(const std::vector<ActionItem> &actions) {
  std::vector<ActionItem> filtered_actions;
  for (const auto &item : actions) {
    if (item.first != "parse") {
      (void)filtered_actions.emplace_back(item);
    }
  }
  return filtered_actions;
}

std::vector<ActionItem> VmPipeline(const ResourcePtr &resource, bool trace_flag, bool erase_parse) {
  is_cluster_initialized = distributed::cluster::ClusterContext::instance()->initialized();
  std::vector<ActionItem> actions;
  // If enable compilation cache and the cache is read successfully, only do the backend actions.
  const std::string &phase = PhaseManager::GetInstance().phase();
  if (IsPhaseLoadFromMindIR(phase)) {
    actions = MindIRPipeline();
  } else if (!resource->EnableCompileCache() || resource->func_graph() == nullptr) {
    actions = CommonPipeline(trace_flag);

    // Optimize
    (void)actions.emplace_back(std::make_pair(kOptimize, VmOptimizeAction));

    (void)actions.emplace_back(std::make_pair(kDetachBackwardAction, DetachBackwardAction));

    (void)actions.emplace_back(std::make_pair(kPipelineParallelScheduler, PipelineSchedulerAction));

    opt::CustomPassExecutor::InsertFrontendCustomPassesToVmPipeline(actions);

    (void)actions.emplace_back(std::make_pair(kAutoMonadReorder, OrderEnforceAction));

    // Eliminate forward cnode for grad graph
    (void)actions.emplace_back(std::make_pair(kGetJitBpropGraph, GetJitBpropGraph));

    // Rewriter(dict convert pyexecute) after jit bprop.
    (void)actions.emplace_back(std::make_pair(kRewriterAfterJitBprop, RewriterAfterOptAPassAfterJitBprop));

    // Eliminate the virtual mirror node
    (void)actions.emplace_back(std::make_pair(kOptAfterJitGrad, OptAfterJitGrad));

    if ((common::GetEnv("MS_DEV_HYPER_OFFLOAD") == "1")) {
      (void)actions.emplace_back(std::make_pair(kOffloadAnnotation, ProcessOffloadAnnotationPass));
    }

    // Mind Compiler finish.
    (void)actions.emplace_back(std::make_pair(kValidate, ValidateAction));

    (void)actions.emplace_back(std::make_pair(kBackendPass, BackendPass));
  }

  if (erase_parse) {
    actions = EraseParseActions(actions);
  }

  auto is_precompile_only = MsContext::GetInstance()->get_param<bool>(MS_CTX_PRECOMPILE_ONLY) ||
                            common::GetEnv("MS_DEV_PRECOMPILE_ONLY") == "1";
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
    // Phase with "export" prefix need to skip backend compilation.
    if (pipeline::IsPhaseExport(phase)) {
      return actions;
    }
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
  std::vector<ActionItem> actions;
  // Set funcGraph loaded from MindIR to resource.
  (void)actions.emplace_back(std::make_pair(kLoadMindir, SetMindIRGraphAction));
  (void)actions.emplace_back(std::make_pair(kModifyMindirGraph, ModifyGraphGeneratedByMindIR));
  (void)actions.emplace_back(std::make_pair(kInferMindir, InferMindIR));
  (void)actions.emplace_back(std::make_pair(kValidate, ValidateAction));
  return actions;
}

std::vector<PassItem> JitPipeline(const ResourcePtr &resource, bool build_top_graph) {
  OrderedMap<std::string, std::function<bool(ResourcePtr)>> jit_passes;
  std::vector<PassItem> full_jit_passes;
  bool init_null = resource->func_graph() == nullptr;
  if (!resource->EnableCompileCache() || init_null) {
    // Compile the frontend graph.
    if (build_top_graph) {
      (void)jit_passes.emplace(kBootstrap, BootstrapAction);
    }
    (void)jit_passes.emplace(kTypeInference, TypeInferenceAction);
    (void)jit_passes.emplace(kEventMethod, EventMethodAction);
    (void)jit_passes.emplace(kAutoMonad, AutoMonadAction);
    (void)jit_passes.emplace(kGraphReusing, GraphReusingAction);
    (void)jit_passes.emplace(kAddAttr, OptAddAttrWithInlineAction);
    (void)jit_passes.emplace(kPreAutoParallel, SetTrainingFlagPass);
    (void)jit_passes.emplace(kPyInterpretToExecute, PyInterpretToExecutePass);
    (void)jit_passes.emplace(kRewriterBeforeOptA, RewriterBeforeOptAPass);
    (void)jit_passes.emplace(kExpandDumpFlag, ExpandDumpFlagPass);
    (void)jit_passes.emplace(kJitOptA, JitOptPassAGroup);
    (void)jit_passes.emplace(kPyInterpretToExecuteAfterOptA, PyInterpretToExecutePass);
    (void)jit_passes.emplace(kRewriterAfterOptA, RewriterAfterOptAPass);
    (void)jit_passes.emplace(kConvertAfterRewriter, ConvertAfterRewriterPass);
    (void)jit_passes.emplace(kOrderPyExecuteAfterRewriter, OrderPyExecuteAfterRewriterPass);
    (void)jit_passes.emplace(kMutableEliminate, MutableEliminatePass);
    (void)jit_passes.emplace(kJitOptB, JitOptPassBGroup);
    (void)jit_passes.emplace(kCconv, CconvPass);
    (void)jit_passes.emplace(kLoopUnroll, LoopUnrollPass);
    (void)jit_passes.emplace(kJitOptPassAfterCconv, JitOptPassAfterCconvGroup);
    (void)jit_passes.emplace(kRemoveDupValue, RemoveValueNodeDuplicationsPassForJit);
    (void)jit_passes.emplace(kPartialUnusedArgsEliminate, PartialUnusedArgsEliminatePass);
    if (common::GetCompileConfig("ENABLE_ELIMINATE_UNUSED_PARAMS") == "1") {
      (void)jit_passes.emplace(kUnusedParamsEliminate, EliminateUnusedParamsPass);
    }
    (void)jit_passes.emplace(kEnvironConv, EnvironConversionPass);
    (void)jit_passes.emplace(kAddRecomputation, AddRecomputationPass);
    (void)jit_passes.emplace(kCseAfterRecomputation, OptAfterRecomputeGroup);
    opt::CustomPassExecutor::InsertFrontendCustomPassesToJitPipeline(jit_passes);
    (void)jit_passes.emplace(kAutoMonadReorder, OrderEnforceAction);
    (void)jit_passes.emplace(kGetJitBpropGraph, GetJitBpropGraph);
    (void)jit_passes.emplace(kRewriterAfterJitBprop, RewriterAfterOptAPassAfterJitBprop);
    (void)jit_passes.emplace(kOptAfterJitGrad, OptAfterJitGrad);
    (void)jit_passes.emplace(kSymbolEngineOpt, SymbolEngineOptGroup);
    if ((common::GetEnv("MS_DEV_HYPER_OFFLOAD") == "1")) {
      (void)jit_passes.emplace(kOffloadAnnotation, ProcessOffloadAnnotationPass);
    }
    (void)jit_passes.emplace(kValidate, ValidatePass);
    (void)jit_passes.emplace(kBackendPass, BackendPass);
    for (const auto &action : jit_passes) {
      const auto &jit_action_positions = ActionConfigure::Instance().jit_action_positions();
      auto iter = jit_action_positions.find(action.first);
      if (iter != jit_action_positions.end()) {
        auto inserted_actions = iter->second;
        std::transform(inserted_actions.begin(), inserted_actions.end(), std::back_inserter(full_jit_passes),
                       [&](const auto &inserted_action) {
                         const auto &jit_actions = ActionConfigure::Instance().jit_actions();
                         auto iter_action = jit_actions.find(inserted_action);
                         return std::make_pair(inserted_action, iter_action->second);
                       });
      }
      (void)full_jit_passes.emplace_back(action);
    }
  }

  auto is_precompile_only = MsContext::GetInstance()->get_param<bool>(MS_CTX_PRECOMPILE_ONLY) ||
                            common::GetEnv("MS_DEV_PRECOMPILE_ONLY") == "1";
  if (is_precompile_only) {
    MS_LOG(INFO) << "PrecompileOnly, stop run graph";
    return full_jit_passes;
  }

  bool is_jit_grad = pynative::GradState::Get().RequiresGrad();
  if (resource->EnableCompileCache() && !init_null && is_jit_grad) {
    // Store forward graph for jit grad when using compile cache.
    (void)full_jit_passes.emplace_back(kGetJitBpropGraph, GetJitBpropGraph);
    return full_jit_passes;
  }

  if (common::GetEnv(kSimulationLevel) == kSimulationLevelCompileGraph) {
    return full_jit_passes;
  }

#ifndef WITH_BACKEND
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->backend_policy() != "ge") {
#endif
    // Phase with "export" prefix need to skip backend compilation.
    const std::string &phase = PhaseManager::GetInstance().phase();
    if (pipeline::IsPhaseExport(phase)) {
      return full_jit_passes;
    }
    // Compile the backend graph.
    (void)full_jit_passes.emplace_back(kTaskEmit, TaskEmitAction);

    // Execute the graph.
    (void)full_jit_passes.emplace_back(kExecute, ExecuteAction);
#ifndef WITH_BACKEND
  }
#endif

  return full_jit_passes;
}

ActionConfigure &ActionConfigure::Instance() {
  static ActionConfigure instance;
  return instance;
}
}  // namespace pipeline
}  // namespace mindspore
