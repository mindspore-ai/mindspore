/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#include "abstract/dshape.h"

#include "tools/profiler/profiling.h"
#include "tools/profiler/profiler.h"

#include "utils/ms_context.h"

#include "include/frontend/jit/ps/executor/graph_executor_py.h"
#include "include/frontend/jit/ps/executor/jit_executor_py.h"
#include "include/utils/ir_dump/dump_proto_interface.h"
#include "include/utils/compile_cache_context.h"
#include "include/utils/python_utils.h"
#include "include/utils/tensor_py_wrapper.h"
#include "include/utils/tensor_py.h"
#include "include/utils/ir_dump/onnx/onnx_exporter.h"
#include "include/utils/symbol_engine/symbol_engine_impl.h"

#include "frontend/jit/ps/compile_cache_manager.h"
#include "frontend/jit/ps/debug/trace.h"
#include "frontend/jit/ps/parse/data_converter.h"
#include "frontend/jit/ps/pipeline.h"

#include "frontend/parallel/dynamic_shape/dynamic_shape.h"

#include "mindspore/ccsrc/utils/symbol_engine/utils.h"
#include "mindspore/core/include/ir/dtype.h"

#include "utils/config_manager.h"
#include "utils/device_manager_conf.h"
#include "utils/log_adapter.h"
#include "utils/phase.h"
#include "utils/shape_utils.h"

#include "pybind_api/pybind_patch.h"
#include "pybind_api/runtime/event_py.h"
#include "pybind11/pybind11.h"

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

namespace {
constexpr size_t kDynamicShapePairSize = 2;

bool IsDimUnknown(const ShapeVector &shape) {
  if (shape.size() == 1 && shape[0] == abstract::Shape::kShapeRankAny) {
    return true;
  }
  if (std::count(shape.begin(), shape.end(), abstract::Shape::kShapeRankAny) > 0) {
    MS_EXCEPTION(ValueError) << "'shape' should have only one " << abstract::Shape::kShapeRankAny << " or no "
                             << abstract::Shape::kShapeRankAny << " at all but got (" << tensor::ShapeToString(shape)
                             << ").";
  }
  return false;
}

void CheckArgTypeShape(const py::object &arg, const py::object &dyn_arg, const std::string &name) {
  if (tensor::IsPyObjectTensorPy(arg.ptr()) && tensor::IsPyObjectTensorPy(dyn_arg.ptr())) {
    auto arg_tensor = tensor::ConvertPyObjectToTensor(arg.ptr());
    auto dyn_arg_tensor = tensor::ConvertPyObjectToTensor(dyn_arg.ptr());
    auto arg_dtype = arg_tensor->data_type();
    auto dyn_arg_dtype = dyn_arg_tensor->data_type();
    if (arg_dtype != dyn_arg_dtype) {
      MS_EXCEPTION(TypeError) << "When using decorator enable_dynamic, input tensor dtype = "
                              << TypeIdToString(arg_dtype)
                              << ", dynamic tensor dtype = " << TypeIdToString(dyn_arg_dtype)
                              << ", tensor dtypes are not the same.";
    }

    auto &arg_shape = arg_tensor->shape_c();
    auto &dyn_arg_shape = dyn_arg_tensor->shape_c();

    // Check if dynamic rank (kShapeRankAny = -2), skip further validation
    if (IsDimUnknown(dyn_arg_shape)) {
      return;
    }

    if (arg_shape.size() != dyn_arg_shape.size()) {
      MS_EXCEPTION(ValueError) << "When using decorator enable_dynamic, input tensor shape = ("
                               << tensor::ShapeToString(arg_shape) << "), dynamic tensor shape = ("
                               << tensor::ShapeToString(dyn_arg_shape) << "), tensor shapes are not the same.";
    }

    for (size_t i = 0; i < arg_shape.size(); ++i) {
      auto dim = arg_shape[i];
      if (dim <= 0) {
        MS_EXCEPTION(ValueError) << "When using decorator enable_dynamic, the corresponding shape of inputs should be "
                                 << "a tuple/list of positive integers";
      }
      auto y = dyn_arg_shape[i];
      if (y != abstract::Shape::kShapeDimAny && dim != y) {
        MS_EXCEPTION(ValueError) << "When using decorator enable_dynamic, input tensor shape = "
                                 << tensor::ShapeToString(arg_shape)
                                 << ", dynamic tensor shape = " << tensor::ShapeToString(dyn_arg_shape)
                                 << ", tensor shapes are not the same.";
      }
    }
  } else if ((py::isinstance<py::tuple>(arg) || py::isinstance<py::list>(arg)) &&
             (py::isinstance<py::tuple>(dyn_arg) || py::isinstance<py::list>(dyn_arg))) {
    if (py::len(arg) != py::len(dyn_arg)) {
      MS_EXCEPTION(ValueError) << "Input sequences must have the same structure and length.";
    }
    for (size_t i = 0; i < py::len(arg); ++i) {
      CheckArgTypeShape(arg[py::cast(i)], dyn_arg[py::cast(i)], name);
    }
  } else {
    MS_EXCEPTION(TypeError) << "When using decorator enable_dynamic, the corresponding inputs only supports Tensor "
                               "or a tuple/list of Tensors, Additionally, the type between argument '"
                            << name << "' and corresponding input should be the same.";
  }
}

py::object GenerateDynamicSequenceArgs(const py::object &args, const py::object &dyn_args) {
  if (tensor::IsPyObjectTensorPy(args.ptr()) && tensor::IsPyObjectTensorPy(dyn_args.ptr())) {
    auto args_tensor = tensor::ConvertPyObjectToTensor(args.ptr());
    auto dyn_args_tensor = tensor::ConvertPyObjectToTensor(dyn_args.ptr());
    if (args_tensor->shape_c() == dyn_args_tensor->shape_c()) {
      return args;
    }
    return dyn_args;
  }
  // Handle tuple/list
  py::list result;
  for (size_t i = 0; i < py::len(args); ++i) {
    result.append(GenerateDynamicSequenceArgs(args[py::cast(i)], dyn_args[py::cast(i)]));
  }
  if (py::isinstance<py::tuple>(args)) {
    return py::tuple(result);
  }
  return result;
}

py::object CheckAllTensor(const py::object &sequence) {
  // Only check tuple (consistent with Python implementation)
  if (!py::isinstance<py::tuple>(sequence)) {
    return py::bool_(true);
  }
  auto tuple_seq = py::cast<py::tuple>(sequence);
  for (const auto &element : tuple_seq) {
    // Check if element is a tensor or stub tensor
    if (tensor::IsPyObjectTensorPy(element.ptr())) {
      continue;
    }
    if (py::hasattr(element, "stub")) {
      continue;
    }
    // Check if element is a tuple that needs recursive check
    if (py::isinstance<py::tuple>(element) && CheckAllTensor(py::reinterpret_borrow<py::object>(element.ptr()))) {
      continue;
    }
    return py::bool_(false);
  }
  return py::bool_(true);
}

py::object HandleArg(const py::object &arg, bool has_mutable_arg, bool is_predict, bool enable_tuple_broaden,
                     bool grad_for_scalar) {
  // Check for Python Tensor (mindspore.common.tensor.Tensor)
  bool is_python_tensor = tensor::IsPyObjectTensorPy(arg.ptr());
  if (is_python_tensor) {
    // Call init_data() if has_init is True
    auto has_init = py::getattr(arg, "has_init", py::bool_(false));
    if (py::cast<bool>(has_init)) {
      py::object init_data = py::getattr(arg, "init_data", py::none());
      if (!init_data.is_none()) {
        init_data();
      }
    }
    // Check const_arg
    auto const_arg = py::getattr(arg, "const_arg", py::bool_(true));
    if (!py::cast<bool>(const_arg)) {
      return arg;
    }
    return py::none();
  }

  if (py::isinstance<mindspore::tensor::Tensor>(arg) || py::isinstance<mindspore::tensor::CSRTensor>(arg) ||
      py::isinstance<mindspore::tensor::COOTensor>(arg)) {
    return arg;
  }

  // Handle mutable args
  if (has_mutable_arg) {
    // Empty list is not supported by backend
    if (py::isinstance<py::list>(arg) && (py::len(arg) == 0)) {
      return py::none();
    }
    return arg;
  }

  // Handle scalar args for grad
  if (!is_predict && grad_for_scalar && (py::isinstance<py::int_>(arg) || py::isinstance<py::float_>(arg))) {
    return arg;
  }

  // Handle tuple broaden
  if (enable_tuple_broaden && py::isinstance<py::tuple>(arg) && py::cast<bool>(CheckAllTensor(arg))) {
    return arg;
  }

  // Handle Event type
  if (py::isinstance<mindspore::hal::EventPy>(arg)) {
    return arg;
  }

  return py::none();
}
}  // namespace

py::tuple ExecutorPy::GetArgsForRun(const py::object &obj, const py::tuple &args, const py::dict &kwargs,
                                    const py::list &has_mutable_args_list, const py::list &sequence_modified,
                                    bool is_predict, bool is_in_phase_cache) {
  // Cache obj attributes to avoid repeated getattr in loop
  bool enable_tuple_broaden = false;
  if (!obj.is_none()) {
    auto attr = py::getattr(obj, "enable_tuple_broaden", py::bool_(false));
    enable_tuple_broaden = py::cast<bool>(attr);
  }

  // Check GRAD_FOR_SCALAR from compile_config
  static const bool grad_for_scalar = (common::GetCompileConfig("GRAD_FOR_SCALAR") == "1");

  py::list new_args;
  size_t sequence_index = 0;

  auto sequence_modified_len = py::len(sequence_modified);
  size_t min_len = std::min(py::len(args), py::len(has_mutable_args_list));
  for (size_t i = 0; i < min_len; ++i) {
    py::object arg = args[i];
    bool has_mutable_arg = py::cast<bool>(has_mutable_args_list[i]);

    py::object new_arg = HandleArg(arg, has_mutable_arg, is_predict, enable_tuple_broaden, grad_for_scalar);
    if (!new_arg.is_none()) {
      new_args.append(new_arg);
    } else if (!is_in_phase_cache && !sequence_modified.is_none() && sequence_modified_len > 0 &&
               (py::isinstance<py::list>(arg) || py::isinstance<py::tuple>(arg))) {
      if (sequence_index < sequence_modified_len && py::cast<bool>(sequence_modified[py::cast(sequence_index)])) {
        new_args.append(arg);
      }
      sequence_index++;
    }
  }

  // Process kwargs
  for (const auto &kv : kwargs) {
    py::object new_value = HandleArg(py::reinterpret_borrow<py::object>(kv.second.ptr()), false, is_predict,
                                     enable_tuple_broaden, grad_for_scalar);
    if (!new_value.is_none()) {
      new_args.append(new_value);
    }
  }

  return py::tuple(new_args);
}

py::list ExecutorPy::GenerateDynamicTensorArgs(const py::tuple &args_list, const py::object &dynamic_shapes) {
  // If dynamic_shapes is None, return original args_list
  if (dynamic_shapes.is_none()) {
    return py::list(args_list);
  }

  // Assume dynamic_shapes is a dict
  auto dynamic_dict = py::cast<py::dict>(dynamic_shapes);
  py::list new_compile_args(args_list);

  for (size_t index = 0; index < py::len(args_list); ++index) {
    py::object arg = args_list[index];
    py::object index_key = py::int_(index);
    // Check if index exists in dynamic_shapes dict
    if (!dynamic_dict.contains(index_key)) {
      continue;
    }

    py::object name_dyn_arg_pair = dynamic_dict[index_key];

    // Check if tuple/list has __ms_mutable__ attribute
    if ((py::isinstance<py::tuple>(arg) || py::isinstance<py::list>(arg)) && !py::hasattr(arg, "__ms_mutable__")) {
      MS_EXCEPTION(ValueError) << "When using decorator enable_dynamic, the corresponding attribute of input should be "
                               << "mutable(tuple/list)";
    }

    auto pair = py::cast<py::tuple>(name_dyn_arg_pair);
    if (py::len(pair) != kDynamicShapePairSize) {
      MS_EXCEPTION(ValueError) << "Invalid dynamic shape pair for index " << index;
    }

    const std::string &name = py::cast<std::string>(pair[0]);
    py::object dyn_arg = pair[1];

    CheckArgTypeShape(arg, dyn_arg, name);
    new_compile_args[index] = GenerateDynamicSequenceArgs(arg, dyn_arg);
  }
  return new_compile_args;
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

void ExecutorPy::SyncTensorForInferValue(const AbstractBasePtr &arg, bool broaden) {
  // If argument is broadened, keep abstract unchanged.
  if (broaden || arg == nullptr) {
    return;
  }

  // For tensor abstract, convert underlying value to a cpu tensor and set it back to the same abstract.
  if (arg->isa<abstract::AbstractTensor>()) {
    ValuePtr value = arg->BuildValue();
    auto tensor = dyn_cast<tensor::Tensor>(value);
    if (tensor != nullptr) {
      auto cpu_tensor = tensor->cpu();
      arg->set_value(cpu_tensor);
    }
    return;
  }

  // For sequence abstract, recursively sync elements in-place.
  if (arg->isa<abstract::AbstractSequence>()) {
    auto seq = arg->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(seq);
    const auto &elements = seq->elements();
    for (const auto &elem : elements) {
      SyncTensorForInferValue(elem);
    }
    return;
  }

  // For dictionary abstract, recursively sync key and value abstracts in-place.
  if (arg->isa<abstract::AbstractDictionary>()) {
    auto dict = arg->cast<abstract::AbstractDictionaryPtr>();
    MS_EXCEPTION_IF_NULL(dict);
    const auto &elements = dict->elements();
    for (const auto &element : elements) {
      SyncTensorForInferValue(element.first);
      SyncTensorForInferValue(element.second);
    }
    return;
  }

  // Other abstract types keep unchanged.
  return;
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

void CheckInMetaDevice() {
  auto device_type = DeviceManagerConf::GetInstance()->device_type();
  if (device_type == device::DeviceType::kMeta) {
    MS_LOG(EXCEPTION) << "Meta device is not supported in graph mode or JIT compilation.";
  }
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
      CheckInMetaDevice();
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

namespace {

void RunSymbolicShapeCheck(const py::object &dyn_input, const py::object &actual_input, size_t index,
                           std::unordered_map<int64_t, int64_t> *unique_id_value_map) {
  auto symbolic_shape = py::getattr(dyn_input, "symbolic_shape", py::none());
  if (symbolic_shape.is_none()) {
    return;
  }
  if (!py::isinstance<py::list>(symbolic_shape)) {
    return;
  }
  auto shape_list = py::cast<py::list>(symbolic_shape);

  if (!tensor::IsPyObjectTensorPy(actual_input.ptr())) {
    return;
  }

  auto actual_tensor = tensor::ConvertPyObjectToTensor(actual_input.ptr());
  const auto &shape_vec = actual_tensor->shape_c();
  const auto actual_size = shape_vec.size();
  const auto symbolic_size = shape_list.size();
  const auto check_count = actual_size < symbolic_size ? actual_size : symbolic_size;

  for (size_t i = 0; i < check_count; ++i) {
    auto sym = shape_list[i];
    if (!py::isinstance<py::dict>(sym)) {
      continue;
    }
    auto sym_dict = py::cast<py::dict>(sym);
    const int64_t actual_value = shape_vec[i];

    if (sym_dict.contains("id")) {
      int64_t sym_id = py::cast<int64_t>(sym_dict["id"]);
      auto it = unique_id_value_map->find(sym_id);
      if (it == unique_id_value_map->end()) {
        (*unique_id_value_map)[sym_id] = actual_value;
      } else if (it->second != actual_value) {
        MS_EXCEPTION(ValueError) << "The " << (i + 1) << "th shape value of " << (index + 1)
                                 << "th actual input args is a unique symbol, all values must be the same. "
                                 << "The previous value is " << it->second << ", but the current value is "
                                 << actual_value << ". Actual shape: " << tensor::ShapeToString(shape_vec)
                                 << ", axis: " << i << ".";
      }
    }

    if (sym_dict.contains("min")) {
      int64_t min_val = py::cast<int64_t>(sym_dict["min"]);
      if (actual_value < min_val) {
        MS_EXCEPTION(ValueError) << "The " << (i + 1) << "th shape value of " << (index + 1)
                                 << "th actual input args must be greater than or equal to the 'min' value '" << min_val
                                 << "' of `Symbol`, but got '" << actual_value
                                 << "'. Actual shape: " << tensor::ShapeToString(shape_vec) << ", axis: " << i << ".";
      }
    }

    if (sym_dict.contains("max")) {
      int64_t max_val = py::cast<int64_t>(sym_dict["max"]);
      if (actual_value > max_val) {
        MS_EXCEPTION(ValueError) << "The " << (i + 1) << "th shape value of " << (index + 1)
                                 << "th actual input args must be less than or equal to the 'max' value '" << max_val
                                 << "' of `Symbol`, but got '" << actual_value
                                 << "'. Actual shape: " << tensor::ShapeToString(shape_vec) << ", axis: " << i << ".";
      }
    }

    const int64_t divisor = sym_dict.contains("divisor") ? py::cast<int64_t>(sym_dict["divisor"]) : 1;
    const int64_t remainder = sym_dict.contains("remainder") ? py::cast<int64_t>(sym_dict["remainder"]) : 0;
    if (actual_value < divisor || actual_value % divisor != remainder) {
      MS_EXCEPTION(ValueError) << "The " << (i + 1) << "th shape value of " << (index + 1)
                               << "th actual input args must match the 'divisor'(d) and 'remainder'(d) of `Symbol`. "
                                  "The value should be 'd * N + r' for 'N > 0', got d="
                               << divisor << " and r=" << remainder << ", but actual shape value is '" << actual_value
                               << "'. Actual shape: " << tensor::ShapeToString(shape_vec) << ", axis: " << i << ".";
    }
  }
}

template <typename DynSeq, typename ActualSeq>
void RunCheckRecursiveImpl(const DynSeq &dyn_inputs, const ActualSeq &actual_inputs,
                           std::unordered_map<int64_t, int64_t> *unique_id_value_map) {
  for (size_t i = 0; i < std::min(dyn_inputs.size(), actual_inputs.size()); ++i) {
    auto dyn_input = dyn_inputs[i];
    auto actual_input = actual_inputs[i];

    // Check if dynamic input is a tuple/list (nested)
    if (py::isinstance<py::tuple>(dyn_input) && py::isinstance<py::tuple>(actual_input)) {
      RunCheckRecursiveImpl<py::tuple, py::tuple>(dyn_input, actual_input, unique_id_value_map);
    } else if (py::isinstance<py::list>(dyn_input) && py::isinstance<py::list>(actual_input)) {
      RunCheckRecursiveImpl<py::list, py::list>(dyn_input, actual_input, unique_id_value_map);
    } else if (py::isinstance<py::tuple>(dyn_input) && py::isinstance<py::list>(actual_input)) {
      RunCheckRecursiveImpl<py::tuple, py::list>(dyn_input, actual_input, unique_id_value_map);
    } else if (py::isinstance<py::list>(dyn_input) && py::isinstance<py::tuple>(actual_input)) {
      RunCheckRecursiveImpl<py::list, py::tuple>(dyn_input, actual_input, unique_id_value_map);
    } else {
      // Check symbolic shape for this input
      RunSymbolicShapeCheck(dyn_input, actual_input, i, unique_id_value_map);
    }
  }
}

}  // namespace

void ExecutorPy::CheckSymbolicShape(const py::list &dynamic_inputs, const py::tuple &actual_inputs) {
  std::unordered_map<int64_t, int64_t> unique_id_value_map;
  RunCheckRecursiveImpl(dynamic_inputs, actual_inputs, &unique_id_value_map);
}

void ExecutorPy::CheckSymbolicShape(const py::tuple &dynamic_inputs, const py::tuple &actual_inputs) {
  std::unordered_map<int64_t, int64_t> unique_id_value_map;
  RunCheckRecursiveImpl(dynamic_inputs, actual_inputs, &unique_id_value_map);
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
