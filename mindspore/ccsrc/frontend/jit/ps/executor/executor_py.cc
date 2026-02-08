/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#include "include/frontend/jit/ps/executor/executor_py.h"

#include <memory>
#include <map>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <unordered_map>
#include <functional>
#include <utility>
#include <vector>
#include <string>
#include "pybind_api/pybind_patch.h"
#include "pybind11/pybind11.h"

#include "tools/profiler/profiling.h"
#include "tools/profiler/profiler.h"

#include "utils/ms_context.h"

#include "mindspore/ccsrc/utils/ir_dump/dump_proto.h"
#include "include/utils/compile_cache_context.h"
#include "utils/config_manager.h"
#include "include/utils/python_utils.h"
#include "include/utils/tensor_py_wrapper.h"
#include "include/utils/ir_dump/onnx/onnx_exporter.h"
#include "include/utils/symbol_engine/symbol_engine_impl.h"

#include "frontend/jit/ps/compile_cache_manager.h"
#include "frontend/jit/ps/debug/trace.h"
#include "include/frontend/jit/ps/executor/graph_executor_py.h"
#include "include/frontend/jit/ps/executor/jit_executor_py.h"
#include "frontend/jit/ps/parse/data_converter.h"
#include "frontend/jit/ps/pipeline.h"

#include "utils/log_adapter.h"
#include "utils/phase.h"
#include "utils/shape_utils.h"

#include "frontend/parallel/dynamic_shape/dynamic_shape.h"
#include "mindspore/ccsrc/utils/symbol_engine/utils.h"

namespace mindspore {
// namespace to support intermediate representation definition
namespace pipeline {
using Tensor = mindspore::tensor::Tensor;
using mindspore::abstract::AbstractTensor;
using mindspore::abstract::AbstractTensorPtr;
using VmEvalPtr = std::shared_ptr<std::function<BaseRef(const VectorRef &)>>;
using MetaTensor = mindspore::tensor::MetaTensor;

const char IR_TYPE_ANF[] = "anf_ir";
const char IR_TYPE_ONNX[] = "onnx_ir";
const char IR_TYPE_MINDIR[] = "mind_ir";

std::unordered_map<abstract::AbstractBasePtrList, uint64_t, abstract::AbstractBasePtrListHasher,
                   abstract::AbstractBasePtrListEqual>
  kArgsCache;
std::unordered_map<PyObject *, abstract::AbstractBasePtrList> kCellArgsMap;

namespace {

void CheckShapeConsistency(const abstract::ShapePtr &compile_shape, const abstract::ShapePtr &args_shape,
                           const std::string &target_str, size_t index) {
  MS_EXCEPTION_IF_NULL(compile_shape);
  MS_EXCEPTION_IF_NULL(args_shape);
  if (*compile_shape == *args_shape) {
    return;
  }

  auto compile_shape_vec = compile_shape->shape();
  auto args_shape_vec = args_shape->shape();

  if (!IsDynamicRank(compile_shape_vec)) {
    if (!args_shape_vec.empty() && compile_shape_vec.size() != args_shape_vec.size()) {
      MS_EXCEPTION(ValueError) << "For " << target_str << " and tuple(list) in " << target_str << ", the dims of "
                               << (index + 1) << "th input must be the same as expected, "
                               << "but got expected: " << compile_shape_vec.size()
                               << ", and input: " << args_shape_vec.size() << "!";
    }

    for (size_t i = 0; i < compile_shape_vec.size(); ++i) {
      if (compile_shape_vec[i] == abstract::Shape::kShapeDimAny || compile_shape_vec[i] == args_shape_vec[i]) {
        continue;
      }
      MS_EXCEPTION(ValueError) << "For " << target_str << " and tuple(list) in " << target_str << ", the shape of "
                               << (index + 1) << "th input must be the same as expected, "
                               << "but got expected: " << compile_shape_vec[i] << ", and input: " << args_shape_vec[i]
                               << "!";
    }
  }
}

inline void CheckSizeConsistency(const AbstractBasePtrList &compile_abstracts,
                                 const AbstractBasePtrList &args_abstracts, const std::string &target_str,
                                 bool dynamic_len = false) {
  if (!dynamic_len && compile_abstracts.size() != args_abstracts.size()) {
    MS_EXCEPTION(ValueError) << "For " << target_str << " and tuple(list) in " << target_str
                             << ", the length of input must be equal to expected one, but got expected: "
                             << compile_abstracts.size() << " and input: " << args_abstracts.size() << "!";
  }
  if (dynamic_len && compile_abstracts.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "For " << target_str << ", the dynamic_len compile arguments should not be empty!";
  }
}

void CheckAbstractConsistency(const AbstractBasePtrList &compile_abstracts, const AbstractBasePtrList &args_abstracts,
                              const std::string &target_str, bool dynamic_len = false) {
  CheckSizeConsistency(compile_abstracts, args_abstracts, target_str, dynamic_len);
  for (size_t i = 0; i < args_abstracts.size(); ++i) {
    auto compile_abs = dynamic_len ? compile_abstracts[0] : compile_abstracts[i];
    auto args_abs = args_abstracts[i];
    auto is_compile_var = compile_abs->BuildValue()->ContainsValueAny();
    auto is_args_var = args_abs->BuildValue()->ContainsValueAny();
    if (is_compile_var != is_args_var) {
      MS_EXCEPTION(TypeError) << "For " << target_str << " or tuple(list) in " << target_str << ", the " << (i + 1)
                              << "th should be " << (is_compile_var ? "mutable" : "static") << " one, but got "
                              << (is_args_var ? "mutable" : "static") << "!";
    }

    if (is_compile_var) {
      if (compile_abs->isa<abstract::AbstractTensor>() && args_abs->isa<abstract::AbstractTensor>()) {
        auto compile_tensor = compile_abs->cast<abstract::AbstractTensorPtr>();
        auto args_tensor = args_abs->cast<abstract::AbstractTensorPtr>();

        // Check shape's consistency.
        auto compile_shape = compile_tensor->shape();
        auto args_shape = args_tensor->shape();
        CheckShapeConsistency(compile_shape, args_shape, target_str, i);

        auto compile_element = compile_tensor->element();
        auto args_element = args_tensor->element();
        if (!common::IsEqual(compile_element, args_element)) {
          MS_EXCEPTION(TypeError) << "For " << target_str << " or tuple(list) in " << target_str << ", the " << (i + 1)
                                  << "th type should be " << compile_tensor->BuildType()->ToString() << ", but got "
                                  << args_tensor->BuildType()->ToString() << "!";
        }
      } else if (compile_abs->isa<abstract::AbstractSequence>() && args_abs->isa<abstract::AbstractSequence>()) {
        auto compile_sequence = compile_abs->cast<abstract::AbstractSequencePtr>();
        auto args_sequence = args_abs->cast<abstract::AbstractSequencePtr>();
        CheckAbstractConsistency(compile_sequence->elements(), args_sequence->elements(), target_str,
                                 compile_sequence->dynamic_len());
      } else {
        if (!common::IsEqual(compile_abs, args_abs)) {
          MS_EXCEPTION(ValueError) << "For " << target_str << " or tuple(list) in " << target_str << ", the " << i + 1
                                   << "th should be" << compile_abs->ToString() << ", but got " << args_abs->ToString()
                                   << "!";
        }
      }
    } else if (compile_abs->isa<abstract::AbstractList>() && args_abs->isa<abstract::AbstractList>()) {
      auto compile_sequence = compile_abs->cast<abstract::AbstractSequencePtr>();
      auto args_sequence = args_abs->cast<abstract::AbstractSequencePtr>();
      CheckAbstractConsistency(compile_sequence->elements(), args_sequence->elements(), target_str);
    } else {
      if (!common::IsEqual(compile_abs, args_abs)) {
        MS_EXCEPTION(ValueError) << "For " << target_str << " or tuple(list) in " << target_str << ", the " << i + 1
                                 << "th should be" << compile_abs->ToString() << ", but got " << args_abs->ToString()
                                 << "!";
      }
    }
  }
}
}  // namespace

void ExecutorPy::CheckArgumentsConsistency(const py::tuple &compile_args, const py::tuple &args_list,
                                           const py::object &target) {
  if ((!py::isinstance<py::str>(target))) {
    MS_EXCEPTION(TypeError) << "The `target` must be string!";
  }
  std::string target_str = py::cast<std::string>(target);
  if (compile_args.size() != args_list.size()) {
    MS_EXCEPTION(ValueError) << "For " << target_str
                             << ", the length of input must be equal to expected one, but got expected: "
                             << compile_args.size() << " and input: " << args_list.size() << "!";
  }

  AbstractBasePtrList compile_abstracts;
  compile_abstracts.reserve(compile_args.size());
  AbstractBasePtrList args_abstracts;
  args_abstracts.reserve(compile_args.size());
  for (size_t i = 0; i < compile_args.size(); ++i) {
    ValuePtr compile_args_converted = nullptr;
    if (!parse::ConvertData(compile_args[i], &compile_args_converted)) {
      MS_LOG(INTERNAL_EXCEPTION) << "ConvertData for " << i << "th compiling argument failed, the argument type is "
                                 << compile_args[i].get_type() << ", value is '" << py::str(compile_args[i]) << "'.";
    }
    compile_abstracts.push_back(ArgsToAbstract(compile_args[i], compile_args_converted));

    ValuePtr args_converted = nullptr;
    if (!parse::ConvertData(args_list[i], &args_converted)) {
      MS_LOG(INTERNAL_EXCEPTION) << "ConvertData for " << i << "th input argument failed, the argument type is "
                                 << args_list[i].get_type() << ", value is '" << py::str(args_list[i]) << "'.";
    }
    args_abstracts.push_back(ArgsToAbstract(args_list[i], args_converted));
  }

  CheckAbstractConsistency(compile_abstracts, args_abstracts, target_str, false);
}

py::object ExecutorPy::GenerateArgumentsKey(const py::object &obj, const py::tuple &args, const py::dict &kwargs,
                                            bool enable_tuple_broaden) {
  MS_LOG(DEBUG) << "GenerateArgumentsKey, args size: " << args.size()
                << ", enable_tuple_broaden: " << enable_tuple_broaden;
  abstract::AbstractBasePtrList args_abs;
  ClearCurConvertInput();
  for (std::size_t i = 0; i < args.size(); i++) {
    ValuePtr converted = nullptr;
    if (!parse::ConvertData(args[i], &converted)) {
      MS_LOG(INTERNAL_EXCEPTION) << "ConvertData for " << i << "th argument failed, the argument type is "
                                 << args[i].get_type() << ", value is '" << py::str(args[i]) << "'.";
    }
    AbstractBasePtr abs = ArgsToAbstract(args[i], converted, enable_tuple_broaden);

    (void)args_abs.emplace_back(abs);
    // The 'converted' maybe a Parameter, we need connect it to the Parameter of func graph,
    // so we keep all inputs for subsequent procedure.
    (void)cur_convert_input_.emplace(args[i].ptr(), std::make_pair(converted, abs));
  }
  for (const auto &item : kwargs) {
    ValuePtr key = nullptr;
    ValuePtr value = nullptr;
    bool success = parse::ConvertData(py::cast<py::object>(item.first), &key) &&
                   parse::ConvertData(py::cast<py::object>(item.second), &value);
    if (!success) {
      MS_LOG(INTERNAL_EXCEPTION) << "ConvertData for argument (" << py::str(item.first) << ": " << py::str(item.second)
                                 << ") failed.";
    }
    AbstractBasePtr value_abs = ArgsToAbstract(py::cast<py::object>(item.second), value, enable_tuple_broaden);
    auto keyword_arg_abs = std::make_shared<abstract::AbstractKeywordArg>(GetValue<std::string>(key), value_abs);

    (void)args_abs.emplace_back(keyword_arg_abs);
    (void)cur_convert_input_.emplace(item.first.ptr(), std::make_pair(value, keyword_arg_abs));
  }

  // If cache matched no need CheckArgsValid
  auto iter = kArgsCache.find(args_abs);
  if (iter != kArgsCache.end()) {
    return py::int_(iter->second);
  }

  // If the input Tensor is modified by inplace or view operators,
  // the abstract will be converted from Tensor to RefTensor,
  // causing the abstract in the cache to also be converted from Tensor to RefTensor.
  // will cause cache misses and result in repeated compilation.
  // So, clone an abstract copy in the cache first to avoid conversion to RefTensor causing lookup failures.
  abstract::AbstractBasePtrList new_args_abs;
  std::transform(args_abs.begin(), args_abs.end(), std::back_inserter(new_args_abs),
                 [](AbstractBasePtr abs) { return abs->Clone(); });
  static uint64_t key_counter = 0;
  kArgsCache[new_args_abs] = key_counter;
  if (!py::isinstance<py::none>(obj)) {
    kCellArgsMap[obj.ptr()] = args_abs;
  }
  MS_LOG(INFO) << "Generate a new compile key for new args, key: " << key_counter;
  if (IS_OUTPUT_ON(mindspore::kInfo)) {
    std::ostringstream buffer;
    buffer << "New cached args:"
           << "\n";
    for (size_t i = 0; i < args_abs.size(); ++i) {
      buffer << "Arg[" << i << "]: " << args_abs[i]->ToString() << "\n";
    }
    MS_LOG(INFO) << buffer.str();
  }
  return py::int_(key_counter++);
}

void ExecutorPy::ClearCompileArgumentsResource() {
  // Clear global converted args saved in GenerateArgumentsKey.
  ClearCurConvertInput();
  // Clear real arguments to avoid memory usage.
  real_arguments_.clear();
}

void ExecutorPy::ClearCurConvertInput() { cur_convert_input_.clear(); }

ResourcePtr ExecutorPy::GetResource(const std::string &phase) {
  MS_LOG(DEBUG) << "Phase size:" << info_.size();
  if (info_.count(phase) == 0) {
    return nullptr;
  }
  return info_[phase]->resource;
}

FuncGraphPtr ExecutorPy::GetFuncGraph(const std::string &phase) {
  const auto it = info_.find(phase);
  if (it == info_.end()) {
    MS_LOG(INFO) << "No executor info. found for phase: " << phase;
    return nullptr;
  }
  return it->second->func_graph;
}

std::vector<bool> ExecutorPy::CheckFuncGraphSequenceParamAbstract(const std::string &phase) {
  MS_LOG(DEBUG) << "phase: " << phase;
  std::vector<bool> sequence_used_by_inplace;
  const auto it = info_.find(phase);
  if (it == info_.end()) {
    MS_LOG(INFO) << "No executor info. found for phase: " << phase;
    return sequence_used_by_inplace;
  }
  const auto &func = it->second->func_graph;
  MS_EXCEPTION_IF_NULL(func);
  const auto &params = func->parameters();
  MS_LOG(DEBUG) << "params.size(): " << params.size();
  for (const auto &param_node : params) {
    const auto &param_abs = param_node->abstract();
    MS_EXCEPTION_IF_NULL(param_abs);
    MS_LOG(DEBUG) << "param_abs: " << param_abs->ToString();
    if (param_abs->isa<abstract::AbstractSequence>()) {
      const auto &seq = param_abs->cast<abstract::AbstractSequencePtr>();
      const auto &elements = seq->elements();
      bool exist_ref_tensor = false;
      for (const auto &ele : elements) {
        const auto &ref_abs = ele->cast<abstract::AbstractRefPtr>();
        if (ref_abs != nullptr && ref_abs->is_inplace()) {
          MS_LOG(DEBUG) << "The tuple or list need append.";
          exist_ref_tensor = true;
          break;
        }
      }
      (void)sequence_used_by_inplace.emplace_back(exist_ref_tensor);
    }
  }
  return sequence_used_by_inplace;
}

void ExecutorPy::SetJitPrimalFuncGraph(const FuncGraphPtr &primal_func_graph, const std::string &phase) {
  MS_EXCEPTION_IF_NULL(primal_func_graph);
  const auto it = info_.find(phase);
  if (it == info_.end()) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, primal_func_graph->return_node())
      << "No executor info. found for phase: " << phase;
    return;
  }
  it->second->jit_primal_func_graph = primal_func_graph;
}

FuncGraphPtr ExecutorPy::GetJitPrimalFuncGraph(const std::string &phase) {
  const auto it = info_.find(phase);
  if (it == info_.end()) {
    MS_LOG(INFO) << "No executor info. found for phase: " << phase;
    return nullptr;
  }
  return it->second->jit_primal_func_graph;
}

FuncGraphPtr ExecutorPy::GetJitGradGraph(const std::string &phase) {
  const auto it = info_.find(phase);
  if (it == info_.end()) {
    MS_LOG(INFO) << "No executor info. found for phase: " << phase;
    return nullptr;
  }
  return it->second->jit_grad_graph;
}

void ExecutorPy::SetJitGradGraph(const FuncGraphPtr &grad_graph, const std::string &phase) {
  MS_EXCEPTION_IF_NULL(grad_graph);
  const auto it = info_.find(phase);
  if (it == info_.end()) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, grad_graph->return_node()) << "No executor info. found for phase: " << phase;
    return;
  }
  if (it->second->jit_grad_graph != nullptr) {
    MS_LOG(DEBUG) << "The grad graph has existed, phase is: " << phase;
  }
  it->second->jit_grad_graph = grad_graph;
}

VmEvalPtr ExecutorPy::GetVmEvalFunc(const std::string &phase, const std::string &kind) {
  ResourcePtr res = GetResource(phase);
  MS_EXCEPTION_IF_NULL(res);
  if (res->HasResult(kind) && res->GetResult(kind).is<VmEvalPtr>()) {
    return res->GetResult(kind).cast<VmEvalPtr>();
  }
  MS_LOG(ERROR) << "GetVmEvalFunc vm model can't find kind:" << kind;
  return nullptr;
}

bool ExecutorPy::HasCompiled(const std::string &phase) const { return info_.count(phase) != 0; }

py::bytes ExecutorPy::GetFuncGraphProto(const std::string &phase, const std::string &ir_type, const bool &incremental) {
  FuncGraphPtr fg_ptr = GetFuncGraph(phase);
  if (fg_ptr == nullptr) {
    for (const auto &item : info_) {
      MS_LOG(DEBUG) << "Phase key is: " << item.first;
    }
    MS_LOG(EXCEPTION) << "Can not find func graph " << phase;
  }

  if (ir_type == IR_TYPE_ANF) {
    std::string proto_str = GetFuncGraphProtoString(fg_ptr);
    if (proto_str.empty()) {
      MS_LOG(EXCEPTION) << "Export ANF format model failed.";
    }
    return proto_str;
  }

  if (ir_type == IR_TYPE_ONNX) {
    std::string proto_str = GetOnnxProtoString(fg_ptr);
    if (proto_str.empty()) {
      MS_LOG(EXCEPTION) << "Export ONNX format model failed.";
    }
    return proto_str;
  }

  if (ir_type == IR_TYPE_MINDIR) {
    std::string proto_str = GetBinaryProtoString(fg_ptr, incremental);
    if (proto_str.empty()) {
      MS_LOG(EXCEPTION) << "Export MINDIR format model failed.";
    }
    return proto_str;
  }

  MS_LOG(INTERNAL_EXCEPTION) << "Unknown ir type: " << ir_type;
}

py::bytes ExecutorPy::GetOnnxFuncGraphProto(const std::string &phase, const std::vector<std::string> &input_names,
                                            const std::vector<std::string> &outputs_names, const int &opset_version,
                                            const bool &export_params, const bool &keep_initializers_as_inputs,
                                            const py::dict &dynamic_axes, const bool &extra_save_params,
                                            const std::string &save_file_dir) {
  FuncGraphPtr fg_ptr = GetFuncGraph(phase);
  std::map<std::string, std::map<int, std::string>> dynamic_axes_map;
  if (!dynamic_axes.empty()) {
    for (auto item : dynamic_axes) {
      std::string input_name = py::cast<std::string>(item.first);
      py::dict inner_dict = py::cast<py::dict>(item.second);
      std::map<int, std::string> dim_name_map;
      for (auto inner_item : inner_dict) {
        int dim = py::cast<int>(inner_item.first);
        std::string shape_name = py::cast<std::string>(inner_item.second);
        dim_name_map[dim] = shape_name;
      }
      dynamic_axes_map[input_name] = dim_name_map;
    }
  }
  std::string proto_str =
    GetOnnxProtoString(fg_ptr, input_names, outputs_names, opset_version, export_params, keep_initializers_as_inputs,
                       dynamic_axes_map, extra_save_params, save_file_dir);

  if (proto_str.empty()) {
    MS_LOG(EXCEPTION) << "Export ONNX format model failed.";
  }
  return proto_str;
}

namespace {
std::map<string, string> GenerateJitConfigMap(const py::dict &jit_config) {
  std::map<string, string> ret{};
  for (auto jit_param = jit_config.begin(); jit_param != jit_config.end(); ++jit_param) {
    auto param_name = py::cast<std::string>(jit_param->first);
    auto param_value = py::cast<std::string>(jit_param->second);
    ret[param_name] = param_value;
  }
  return ret;
}
}  // namespace

void ExecutorPy::SetJitConfig(const py::dict &config) {
  auto jit_config = GenerateJitConfigMap(config);
  PhaseManager::GetInstance().set_jit_config(jit_config);

  auto infer_boost_iter = jit_config.find("infer_boost");
  if (infer_boost_iter != jit_config.end()) {
    if (!MsContext::GetInstance()->IsEnableInferBoost()) {
      // Model Phase only enable infer boost
      // when set_context enable infer boost, do not disable by executor
      MsContext::GetInstance()->SetJitInferBoost(infer_boost_iter->second);
    }
  }
}

std::map<std::string, std::string> ExecutorPy::GetJitConfig() { return PhaseManager::GetInstance().jit_config(); }

namespace {
void ClearArgCache(const py::object &obj) {
  if (py::isinstance<py::none>(obj)) {
    return;
  }
  auto iter = kCellArgsMap.find(obj.ptr());
  if (iter != kCellArgsMap.end()) {
    (void)kArgsCache.erase(iter->second);
    (void)kCellArgsMap.erase(iter);
  }
}

inline pid_t GetCurrentPID() {
#if defined(_WIN32) || defined(_WIN64)
  return GetCurrentProcessId();
#else
  return getpid();
#endif
}

}  // namespace

// Not support multi thread, not support nested call too.
// Here using nested_called flg to avoid nested call.
void ExecutorPy::DelNetRes(const py::object &source, const py::set &id) {
  // no need to del net res by gc in independent dataset process which is a subprocess forked by main process
  if (process_id_ != GetCurrentPID()) {
    return;
  }
  ClearArgCache(source);
  // Del all graphs by different phase
  for (auto item : id) {
    DelOneNetRes(item);
  }
}

void ExecutorPy::set_process_id() { process_id_ = GetCurrentPID(); }

std::string ExecutorPy::get_queue_name(const std::string &dataset_phase) {
  return CompileCacheManager::GetCachedDataQueueName(dataset_phase);
}

void ExecutorPy::InitCompileCacheInfo(const ResourcePtr &resource, const std::string &phase) {
  // The compilation cache only support for training cell or functions decorated with 'jit' currently.
  // If enable compilation cache, it will get a non-empty dependent files list from python.
  if (!CompileCacheEnable()) {
    return;
  }
  bool has_python_script = true;
  if (compile_cache_dep_files_.empty()) {
    has_python_script = false;
  }

  {
    MsProfileStatGuard stat_guard("LoadCachedFuncGraph");
    static size_t idx = 0;
    MS_EXCEPTION_IF_NULL(resource);
    resource->GetCompileCacheResource(compile_cache_dep_files_, weights_, queue_name_, idx++,
                                      &compile_cache_consistent_, has_python_script);
    if (resource->func_graph() != nullptr) {
      ResetId(resource);
    }
  }
}

void ExecutorPy::InitCompileCacheResource(const ResourcePtr &resource, const std::string &phase) {
  InitCompileCacheInfo(resource, phase);
  bool enable_compile_cache = resource->EnableCompileCache();
  bool use_compile_cache = enable_compile_cache && resource->func_graph();
  if (use_compile_cache) {
    auto jit_config = PhaseManager::GetInstance().jit_config();
    resource->func_graph()->set_user_data<std::map<std::string, std::string>>(
      "jit_config", std::make_shared<std::map<std::string, std::string>>(jit_config));
  }
  auto &compile_cache_context = CompileCacheContext::GetInstance();
  compile_cache_context.SetUseCompileCache(use_compile_cache);
  ConfigManager::GetInstance().ResetQueue(queue_name_);
}

void ExecutorPy::ReleaseResourceOnException(const py::object &phase) {
  bool clear = false;
  // Be sure the pointer res destroyed before do DelOneNetRes.
  {
    ResourcePtr res = GetResource(py::cast<std::string>(phase));
    if (res != nullptr) {
      clear = true;
      CleanCompileRes(res);
    }
  }
  ProcessStatus::GetInstance().Clear();
  if (clear) {
    DelOneNetRes(phase);
  }
}

namespace {
std::string GetCompileExceptionInfo() {
  std::ostringstream oss;
  trace::GetTraceStackInfo(oss);
  return oss.str();
}
}  // namespace

bool ExecutorPy::Compile(const py::object &source, const py::tuple &args, const py::dict &kwargs,
                         const py::object &phase, const py::dict &config) {
  bool res = false;
  HandleExceptionRethrow(
    [this, &res, &source, &args, &kwargs, &phase, &config]() {
      bool executor_running = false;
      std::string running_obj_desc;
      if (GraphExecutorPy::GetInstance()->executor_running()) {
        executor_running = true;
        running_obj_desc = GraphExecutorPy::GetInstance()->obj_desc();
      } else if (JitExecutorPy::GetInstance()->executor_running()) {
        executor_running = true;
        running_obj_desc = JitExecutorPy::GetInstance()->obj_desc();
      }
      if (executor_running) {
        MS_LOG(EXCEPTION) << "Nested execution during JIT execution for " << GetObjDesc(source) << " is not supported "
                          << "when " << running_obj_desc << " compile and execute. For more details, please refer to "
                          << "https://www.mindspore.cn/search?inputValue=Nested%20execution";
      }
      ProcessStatus::GetInstance().RecordStart(kCompiler);
      std::map<std::string, std::string> custom_info;
      custom_info["phase"] = py::cast<std::string>(phase);
      uint64_t start_time = profiler::GetClockSyscnt();
      SetJitConfig(config);
      res = CompileInner(source, args, kwargs, phase);
      (void)profiler::CollectHostInfo(kCompiler, kCompiler, kCompiler, start_time, profiler::GetClockSyscnt(), 1,
                                      custom_info);
      ProcessStatus::GetInstance().RecordEnd();
      ProcessStatus::GetInstance().Print();
    },
    [this, &phase]() {
      if (!StaticAnalysisException::Instance().HasException()) {
        // print function call stack info before release
        std::string compile_exception_info = GetCompileExceptionInfo();
        if (!compile_exception_info.empty()) {
          MS_LOG(ERROR) << compile_exception_info;
        }
      }
      ReleaseResourceOnException(phase);
    },
    [this, &phase]() { ReleaseResourceOnException(phase); }, [this, &phase]() { ReleaseResourceOnException(phase); });

  // Set need recompile to false after compile finished.
  return res;
}

namespace {
void ProcessVmArgInner(const py::tuple &args, const ResourcePtr &res, VectorRef *const arg_list) {
  MS_EXCEPTION_IF_NULL(arg_list);
  bool arg_list_inited = !arg_list->empty();
  for (std::size_t i = 0; i < args.size(); i++) {
    py::object arg = args[i];
    ValuePtr converted = nullptr;
    bool succ = parse::ConvertData(arg, &converted);
    if (!succ) {
      MS_LOG(INTERNAL_EXCEPTION) << "The " << i << "th arg convert failed.";
    }
    if (!arg_list_inited) {
      arg_list->push_back(converted);
      continue;
    }
    if (i >= arg_list->size()) {
      MS_LOG(INTERNAL_EXCEPTION) << "i:" << i << " output of range:" << arg_list->size();
    }
    (*arg_list)[i] = converted;
  }

  MS_EXCEPTION_IF_NULL(res);
  auto graph = res->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  const std::vector<AnfNodePtr> &graph_params = graph->parameters();
  std::size_t graph_params_size = graph_params.size();
  if ((*arg_list).size() != graph_params_size) {
    // Maybe some default parameter
    for (std::size_t i = (*arg_list).size(); i < graph_params_size; i++) {
      MS_EXCEPTION_IF_NULL(graph_params[i]);
      auto param_ptr = (graph_params[i])->cast_ptr<Parameter>();
      MS_EXCEPTION_IF_NULL(param_ptr);
      if (!param_ptr->has_default()) {
        MS_LOG_WITH_NODE(EXCEPTION, graph_params[i]) << "Parameter[" << i << "] has no default param";
      }
      if (!param_ptr->default_param()->isa<Tensor>()) {
        MS_LOG_WITH_NODE(EXCEPTION, graph_params[i])
          << "Parameter[" << param_ptr->ToString() << "] is not initialized, need to call `.init_data()`";
      }
      arg_list->push_back(param_ptr->default_param());
    }
  }
}
}  // namespace

void ExecutorPy::ProcessVmArg(const py::tuple &args, const std::string &phase, VectorRef *const arg_list) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kGraphExecutorPy, runtime::ProfilerEvent::kInputProcess,
                                     phase);
  ProcessVmArgInner(args, GetResource(phase), arg_list);
}

py::object ExecutorPy::Run(const py::tuple &args, const py::object &phase) {
  py::object res;
  HandleExceptionRethrow(
    [this, &res, &args, &phase]() {
      executor_running_ = true;

      uint64_t start_time = 0;
      PROFILER_START(start_time);
      res = RunInner(args, phase);
      PROFILER_STAGE_END(start_time, runtime::ProfilerStage::kRunGraph);

      executor_running_ = false;
    },
    [this]() { executor_running_ = false; }, [this]() { executor_running_ = false; },
    [this]() { executor_running_ = false; }, nullptr, true);
  return res;
}

void ExecutorPy::ClearRunArgumentsResource(size_t input_arg_size, VectorRef *arg_list) {
  for (std::size_t i = 0; i < input_arg_size; ++i) {
    (*arg_list)[i] = nullptr;
  }
}

py::dict ExecutorPy::GetParams(const std::string &phase) {
  FuncGraphPtr func_graph = info_[phase]->resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  py::dict parameter_dict;
  std::vector<AnfNodePtr> graph_params = func_graph->parameters();
  for (auto &param : graph_params) {
    MS_EXCEPTION_IF_NULL(param);
    auto param_ptr = std::static_pointer_cast<Parameter>(param);
    std::string name = param_ptr->name();
    auto tensorpy = tensor::GetTensorPyFromValue(param_ptr->default_param_raw());
    if (tensorpy != py::none()) {
      parameter_dict[py::str(name)] = tensorpy;
    }
  }
  return parameter_dict;
}

void ExecutorPy::SetRealArguments(const py::tuple &args, const py::dict &kwargs) {
  ValuePtrList arguments;
  for (std::size_t i = 0; i < args.size(); ++i) {
    ValuePtr converted = nullptr;
    bool success = parse::ConvertData(args[i], &converted);
    if (!success) {
      MS_LOG(INTERNAL_EXCEPTION) << "Fail to convert the " << i << "th argument, args[" << i
                                 << "]: " << py::str(args[i]);
    }
    (void)arguments.emplace_back(converted);
  }
  for (const auto &item : kwargs) {
    ValuePtr value = nullptr;
    bool success = parse::ConvertData(py::cast<py::object>(item.second), &value);
    if (!success) {
      MS_LOG(INTERNAL_EXCEPTION) << "Fail to convert the argument (" << py::str(item.first) << ": "
                                 << py::str(item.second) << ").";
    }
    (void)arguments.emplace_back(value);
  }
  real_arguments_ = arguments;
}

void ExecutorPy::ConvertSymbolicShape(const py::tuple &args, AbstractBasePtrList *args_abs) {
  std::vector<symshape::SymbolInfoList> symbol_infos;
  symbol_infos.reserve(args_abs->size());
  bool has_dyn_shape = false;
  bool is_parallel = parallel::IsSemiOrAutoParallelMode();

  for (size_t i = 0; i < args.size(); i++) {
    auto iter = cur_convert_input_.find(args[i].ptr());
    if (iter == cur_convert_input_.end()) {
      continue;
    }
    auto &info_list = symbol_infos.emplace_back(symshape::SymbolInfoList{});
    if (!iter->second.first->isa<MetaTensor>()) {
      continue;
    }
    auto digital_shape = iter->second.second->GetShape();
    MS_EXCEPTION_IF_NULL(digital_shape);
    if (digital_shape->IsDynamic()) {
      has_dyn_shape = true;
    }
    constexpr char symbolic_shape_attr[] = "symbolic_shape";
    if (!py::hasattr(args[i], symbolic_shape_attr) ||
        !py::isinstance<py::list>(py::getattr(args[i], symbolic_shape_attr))) {
      if (is_parallel && digital_shape->isa<abstract::TensorShape>()) {
        info_list.resize(digital_shape->GetShapeVector().size());
      }
      continue;
    }
    auto symbolic_shape_obj = py::getattr(args[i], symbolic_shape_attr);
    MS_EXCEPTION_IF_CHECK_FAIL(py::isinstance<py::list>(symbolic_shape_obj), "tensor.symbolic_shape should be a list");
    auto obj_list = py::cast<py::list>(symbolic_shape_obj);
    info_list.resize(obj_list.size());
    for (size_t j = 0; j < obj_list.size(); j++) {
      if (!py::isinstance<py::dict>(obj_list[j])) {
        continue;
      }
      auto dict_obj = py::cast<py::dict>(obj_list[j]);
      for (auto cfg_iter = dict_obj.begin(); cfg_iter != dict_obj.end(); ++cfg_iter) {
        auto cfg_key = py::cast<std::string>(cfg_iter->first);
        if (cfg_key == "max") {
          info_list[j].max = py::cast<int64_t>(cfg_iter->second);
        } else if (cfg_key == "min") {
          info_list[j].min = py::cast<int64_t>(cfg_iter->second);
        } else if (cfg_key == "divisor") {
          info_list[j].divisor = py::cast<int64_t>(cfg_iter->second);
        } else if (cfg_key == "remainder") {
          info_list[j].remainder = py::cast<int64_t>(cfg_iter->second);
        } else if (cfg_key == "id") {
          info_list[j].id = py::cast<int64_t>(cfg_iter->second);
        } else if (cfg_key == "name") {
          info_list[j].name = py::cast<std::string>(cfg_iter->second);
        }
      }
    }
  }

  MS_LOG(DEBUG) << "before parallel symbol";
  parallel::PrintSymbolInfo(symbol_infos);
  symbol_infos = parallel::ParallelSymbolInfo(symbol_infos, has_dyn_shape);
  MS_LOG(DEBUG) << "after parallel symbol";
  parallel::PrintSymbolInfo(symbol_infos);

  auto symbolic_shape_list = symshape::BuildSymbolicShapeBySymbolInfo(*args_abs, symbol_infos);
  for (size_t i = 0; i < symbolic_shape_list.size(); i++) {
    // when the same tensor object is used in set_inputs interface, the inputs may shared a same Abstract object.
    // but for dynamic shape, the same "-1" in abstract can be different symbolic shape.
    auto abs = symshape::CloneAbstractIfSymbolExists((*args_abs)[i]);
    MS_EXCEPTION_IF_NULL(abs);
    abs->SetSymbolicShape(symbolic_shape_list[i]);
    (*args_abs)[i] = abs;
  }
}

pipeline::ExecutorPyPtr GetExecutor(const std::string &phase) {
  if (common::GetEnv("MS_DEV_JIT_PIPELINE") == "0") {
    return pipeline::GraphExecutorPy::GetInstance();
  }
  if (phase.empty() || pipeline::JitExecutorPy::GetInstance()->HasCompiled(phase)) {
    return pipeline::JitExecutorPy::GetInstance();
  }
  return pipeline::GraphExecutorPy::GetInstance();
}

void CleanCache() {
  kArgsCache.clear();
  kCellArgsMap.clear();
}

}  // namespace pipeline
}  // namespace mindspore
