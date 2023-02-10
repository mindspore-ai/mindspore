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

#include "pybind_api/ir/primitive_py.h"

#include <map>
#include "ir/signature.h"
#include "pipeline/jit/parse/data_converter.h"
#include "include/common/utils/python_adapter.h"
#include "pybind11/pytypes.h"
#include "include/common/pybind_api/api_register.h"
#include "pybind_api/export_flags.h"
#include "pybind_api/ir/base_ref_py.h"
#include "utils/convert_utils_base.h"
#include "include/common/utils/convert_utils_py.h"
#include "utils/ms_context.h"
#include "include/common/utils/primitive_utils.h"
#include "utils/check_convert_utils.h"
#include "pipeline/pynative/pynative_execute.h"

namespace mindspore {
namespace {
constexpr auto kBpropAttrName = "bprop";
constexpr auto kCellHookAttrName = "cell_hook";
constexpr auto kCellIDAttrName = "cell_id";
constexpr auto kCustomOpBpropAttrName = "custom_op_bprop";
std::map<std::string, std::string> kOpAttrNameReplaceMap = {
  {"data_format", "format"},
};

void SyncData(const py::object &arg) {
  if (py::isinstance<py::tuple>(arg)) {
    py::tuple arg_list = py::cast<py::tuple>(arg);
    for (size_t i = 0; i < arg_list.size(); i++) {
      SyncData(arg_list[i]);
    }
  }
  if (py::isinstance<tensor::Tensor>(arg)) {
    auto tensor = py::cast<tensor::TensorPtr>(arg);
    tensor->data_sync();
  }
}

void ConvertCTensorToPyTensor(const py::tuple &input_args, py::tuple *convert_args) {
  MS_EXCEPTION_IF_NULL(convert_args);
  if (input_args.size() != (*convert_args).size()) {
    MS_LOG(EXCEPTION) << "The size of input_args: " << input_args.size()
                      << " should be equal to the size of convert_args: " << (*convert_args).size();
  }
  for (size_t i = 0; i < input_args.size(); ++i) {
    if (py::isinstance<tensor::Tensor>(input_args[i])) {
      (*convert_args)[i] =
        python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE, parse::PYTHON_MOD_CONVERT_TO_MS_TENSOR, input_args[i]);
    } else if (py::isinstance<tensor::CSRTensor>(input_args[i])) {
      (*convert_args)[i] = python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE,
                                                    parse::PYTHON_MOD_CONVERT_TO_MS_CSRTENSOR, input_args[i]);
    } else if (py::isinstance<tensor::COOTensor>(input_args[i])) {
      (*convert_args)[i] = python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE,
                                                    parse::PYTHON_MOD_CONVERT_TO_MS_COOTENSOR, input_args[i]);
    } else if (py::isinstance<py::tuple>(input_args[i])) {
      auto tuple_inp_arg = py::cast<py::tuple>(input_args[i]);
      py::tuple convert_tuple_arg(tuple_inp_arg.size());
      ConvertCTensorToPyTensor(tuple_inp_arg, &convert_tuple_arg);
      (*convert_args)[i] = convert_tuple_arg;
    } else {
      (*convert_args)[i] = input_args[i];
    }
  }
}

py::tuple ConstructCellHookFnArgs(const std::string &cell_id, const py::object &grad_input,
                                  const py::object &grad_output) {
  constexpr size_t grad_input_index = 1;
  constexpr size_t grad_output_index = 2;
  constexpr size_t input_args_nums = 3;
  // Convert c++ object to python object.
  py::tuple c_grad_args(input_args_nums - 1);
  c_grad_args[0] = grad_input;
  c_grad_args[1] = grad_output;
  py::tuple py_grad_args(input_args_nums - 1);
  ConvertCTensorToPyTensor(c_grad_args, &py_grad_args);
  // Get tuple args of cell hook function.
  py::tuple hook_fn_args(input_args_nums);
  hook_fn_args[0] = cell_id;
  if (!py::isinstance<py::tuple>(py_grad_args[0])) {
    hook_fn_args[grad_input_index] = py::make_tuple(py_grad_args[0]);
  } else {
    hook_fn_args[grad_input_index] = py_grad_args[0];
  }
  if (!py::isinstance<py::tuple>(py_grad_args[1])) {
    hook_fn_args[grad_output_index] = py::make_tuple(py_grad_args[1]);
  } else {
    hook_fn_args[grad_output_index] = py_grad_args[1];
  }
  return hook_fn_args;
}

struct RunPrimitivePyHookFunctionRegister {
  RunPrimitivePyHookFunctionRegister() {
    python_adapter::PyAdapterCallback::SetRunPrimitivePyHookFunctionHandler(
      [](const PrimitivePtr &prim, const VectorRef &args) -> BaseRef {
        auto py_prim = prim->cast<PrimitivePyPtr>();
        MS_EXCEPTION_IF_NULL(py_prim);
        return py_prim->RunHookFunction(args);
      });
  }
} callback_register;
}  // namespace
std::map<std::string, py::object> PrimitivePy::hook_grad_;

PrimitivePy::PrimitivePy(const std::string &name) : Primitive(name, false), python_obj_(py::none()) {}

PrimitivePy::PrimitivePy(const PrimitivePy &prim_py)
    : Primitive(prim_py),
      python_obj_(prim_py.python_obj_),
      bprop_cls_name_(prim_py.bprop_cls_name_),
      adapter_(prim_py.adapter_),
      signatures_(prim_py.signatures_),
      bprop_cut_prims_(prim_py.bprop_cut_prims_),
      backward_hook_fn_(prim_py.backward_hook_fn_) {}

PrimitivePy &PrimitivePy::operator=(const PrimitivePy &other) {
  if (this == &other) {
    return *this;
  }
  Primitive::operator=(other);
  python_obj_ = other.python_obj_;
  bprop_cls_name_ = other.bprop_cls_name_;
  adapter_ = other.adapter_;
  signatures_ = other.signatures_;
  bprop_cut_prims_ = other.bprop_cut_prims_;
  backward_hook_fn_ = other.backward_hook_fn_;
  return *this;
}

PrimitivePy::PrimitivePy(const py::object &python_obj, const PrimitivePyAdapterPtr &adapter)
    : Primitive(adapter->name_, false), python_obj_(python_obj), adapter_(adapter) {
  MS_LOG(DEBUG) << "New primitive:" << adapter->name_;
  set_signatures(adapter->signatures_);
  (void)Primitive::SetAttrs(adapter->attrs_);
  Primitive::set_prim_type(adapter->prim_type_);
  Primitive::set_const_prim(adapter->is_const_prim_);
  Primitive::set_const_input_indexes(adapter->const_input_indexes_);
  for (const auto &elem : adapter->backward_hook_fn_) {
    AddBackwardHookFn(elem.first, elem.second);
  }
  set_instance_name(adapter->instance_name_);
}
PrimitivePy::~PrimitivePy() {}

void PrimitivePy::set_signatures(const std::vector<Signature> &signatures) {
  signatures_ = signatures;
  set_has_signature(!signatures.empty());
}

py::function PrimitivePy::GetVmapRuleFunction(const bool, int axis_size) {
  constexpr char get_vmap_rule_func_name[] = "get_vmap_rule";
  if (py::hasattr(python_obj_, get_vmap_rule_func_name)) {
    return python_obj_.attr(get_vmap_rule_func_name)().cast<py::function>();
  }
  return GetVmapRuleFunctionByObj(python_obj_, axis_size);
}

py::function PrimitivePy::GetBpropFunction() {
  static const char *const get_bprop_func_name = "get_bprop";
  if (py::hasattr(python_obj_, get_bprop_func_name)) {
    py::function fn = python_obj_.attr(get_bprop_func_name)().cast<py::function>();
    return fn;
  }

  auto fn = GetBpropFunctionByObj(python_obj_);
  return fn;
}

py::function PrimitivePy::GetTaylorRuleFunction() {
  static const char *const get_taylor_rule_func_name = "get_taylor_rule";
  if (py::hasattr(python_obj_, get_taylor_rule_func_name)) {
    py::function fn = python_obj_.attr(get_taylor_rule_func_name)().cast<py::function>();
    return fn;
  }
  auto fn = GetTaylorRuleFunctionByObj(python_obj_);
  return fn;
}

py::tuple check_bprop_out(const py::object &grads_obj, const py::tuple &py_args, const std::string &bprop_cls_name) {
  py::tuple grads;
  if (py::isinstance<py::none>(grads_obj)) {
    MS_EXCEPTION(TypeError) << "The 'grads_obj' is none.";
  } else if (!py::isinstance<py::tuple>(grads_obj)) {
    grads = py::make_tuple(grads_obj);
  } else {
    grads = py::cast<py::tuple>(grads_obj);
  }
  if (!MsContext::GetInstance()->get_param<bool>(MS_CTX_CHECK_BPROP_FLAG)) {
    return grads;
  }
  constexpr int filter_args_size = 2;
  if (grads.size() != py_args.size() - filter_args_size) {
    MS_EXCEPTION(TypeError) << "For user defined method 'bprop' of net '" << bprop_cls_name
                            << "', the number of return values(gradients) should be equal to the number of input "
                               "arguments except 'out' and 'dout', which is: "
                            << (py_args.size() - filter_args_size) << ", but got:" << grads.size() << ".";
  }
  for (size_t i = 0; i < grads.size(); i++) {
    if (py::isinstance<tensor::Tensor>(py_args[i])) {
      if (!py::isinstance<tensor::Tensor>(grads[i])) {
        MS_EXCEPTION(ValueError) << "For user defined method 'bprop' of net '" << bprop_cls_name << "', the " << i
                                 << "th return value(gradient of the " << i << "th argument) should be Tensor, but got "
                                 << py::cast<std::string>(grads[i].attr("__class__").attr("__name__"))
                                 << ", and the value is " << py::cast<py::str>(grads[i]) << ".";
      }

      py::object arg_dtype = py_args[i].attr("dtype");
      py::object grad_dtype = grads[i].attr("dtype");
      py::tuple arg_shape = py_args[i].attr("shape");
      py::tuple grad_shape = grads[i].attr("shape");
      if (!grad_dtype.equal(arg_dtype)) {
        MS_EXCEPTION(TypeError) << "For user defined method 'bprop' of net '" << bprop_cls_name << "', the " << i
                                << "th return value(gradient of the " << i
                                << "th argument) should have the same dtype as the " << i
                                << "th argument, which is:" << py::cast<py::str>(arg_dtype)
                                << ", but got: " << py::cast<py::str>(grad_dtype) << ".";
      }
      if (!grad_shape.equal(arg_shape)) {
        MS_EXCEPTION(ValueError) << "For user defined method 'bprop' of net '" << bprop_cls_name << "', the " << i
                                 << "th return value(gradient of the " << i
                                 << "th argument) should have the same shape as the " << i
                                 << "th argument, which is:" << py::cast<py::str>(arg_shape)
                                 << ", but got: " << py::cast<py::str>(grad_shape) << ".";
      }
    }
  }
  return grads;
}

void PrimitivePy::AddBpropCutPrim(const PrimitivePyPtr &bprop_cut_prim) {
  MS_EXCEPTION_IF_NULL(bprop_cut_prim);
  (void)bprop_cut_prims_.emplace_back(bprop_cut_prim);
}

void PrimitivePy::AddBackwardHookFn(const int &key, const py::function &backward_hook_fn) {
  backward_hook_fn_[key] = backward_hook_fn;
  for (const auto &elem : bprop_cut_prims_) {
    PrimitivePyPtr bprop_cut_prim = elem.lock();
    if (bprop_cut_prim != nullptr) {
      bprop_cut_prim->AddBackwardHookFn(key, backward_hook_fn);
    }
  }
}

void PrimitivePy::RemoveBackwardHookFn(const int &key) {
  auto iter = backward_hook_fn_.find(key);
  if (iter != backward_hook_fn_.end()) {
    (void)backward_hook_fn_.erase(key);
  }
  // Remove hook_fn for bprop cut prim on grad graph.
  for (const auto &elem : bprop_cut_prims_) {
    PrimitivePyPtr bprop_cut_prim = elem.lock();
    if (bprop_cut_prim != nullptr) {
      bprop_cut_prim->RemoveBackwardHookFn(key);
    }
  }
}

py::object PrimitivePy::UnpackRetValueOfCellHook(const py::object &grad_out) const {
  if (!py::isinstance<py::tuple>(grad_out)) {
    hook_grad_.clear();
    MS_EXCEPTION(TypeError) << "The return gradient of cell backward hook function should be a tuple!";
  }
  auto out_tuple = py::cast<py::tuple>(grad_out);
  if (out_tuple.size() == 1) {
    // The input number of current cell is 1.
    return out_tuple[0];
  }
  return grad_out;
}

void PrimitivePy::CheckHookConsistency(const py::object &grad_out, const py::object &expected_grad_out,
                                       const py::object &code_obj, const py::object &co_name) const {
  if (py::isinstance<py::tuple>(expected_grad_out)) {
    if (!py::isinstance<py::tuple>(grad_out)) {
      hook_grad_.clear();
      MS_EXCEPTION(TypeError) << "The output gradient should be a tuple!";
    }
    auto actual_out_tuple = py::cast<py::tuple>(grad_out);
    auto expected_out_tuple = py::cast<py::tuple>(expected_grad_out);
    if (actual_out_tuple.size() != expected_out_tuple.size()) {
      hook_grad_.clear();
      MS_EXCEPTION(ValueError) << "The tuple size of output gradient should be " << expected_out_tuple.size()
                               << ", but it is " << actual_out_tuple.size();
    }
    for (size_t i = 0; i < expected_out_tuple.size(); ++i) {
      CheckHookConsistency(actual_out_tuple[i], expected_out_tuple[i], code_obj, co_name);
    }
  }

  if (py::isinstance<tensor::Tensor>(expected_grad_out)) {
    if (!py::isinstance<tensor::Tensor>(grad_out)) {
      hook_grad_.clear();
      MS_EXCEPTION(TypeError) << "The output type of:" << py::str(co_name) << " should be a tensor but got "
                              << py::cast<std::string>(grad_out.attr("__class__").attr("__name__")) << ".";
    }
    auto actual_out_tensor = PyTensorCast(grad_out);
    auto expected_out_tensor = PyTensorCast(expected_grad_out);
    MS_EXCEPTION_IF_NULL(actual_out_tensor);
    MS_EXCEPTION_IF_NULL(expected_out_tensor);
    if (actual_out_tensor->GetShapeAndDataTypeInfo() != expected_out_tensor->GetShapeAndDataTypeInfo()) {
      hook_grad_.clear();
      MS_EXCEPTION(ValueError) << "The output type of " << py::str(co_name)
                               << " is not consistent with the expected, it should be "
                               << expected_out_tensor->GetShapeAndDataTypeInfo() << ", but got "
                               << actual_out_tensor->GetShapeAndDataTypeInfo();
    }
  }
}

BaseRef PrimitivePy::RunCellBpropFunction(const py::tuple &py_args) const {
  if (backward_hook_fn_.size() > 1) {
    MS_LOG(EXCEPTION) << "Multiple registration of bprop function is not supported.";
  }
  SyncData(py_args);
  py::tuple converted_args(py_args.size());
  ConvertCTensorToPyTensor(py_args, &converted_args);
  constexpr size_t non_inp_args_size = 2;  // out and dout.
  auto inp_args_size = py_args.size() - non_inp_args_size;
  py::tuple input_args(inp_args_size);
  for (size_t i = 0; i < inp_args_size; ++i) {
    input_args[i] = py_args[i];
  }
  // Run bprop function.
  auto inst = pynative::PyNativeExecutor::GetInstance();
  MS_EXCEPTION_IF_NULL(inst);
  try {
    MS_LOG(DEBUG) << "Run bprop function start.";
    py::tuple grads;
    for (const auto &elem : backward_hook_fn_) {
      inst->NewGraph(elem.second, input_args.cast<py::args>());
      py::object grads_obj = elem.second(*converted_args);
      grads = check_bprop_out(grads_obj, py_args, bprop_cls_name_);
      inst->EndGraph(elem.second, grads_obj, input_args.cast<py::args>());
    }
    MS_LOG(DEBUG) << "Run bprop function end.";
    return std::make_shared<PyObjectRef>(grads);
  } catch (std::exception &bt) {
    inst->ClearRes();
    std::rethrow_exception(std::current_exception());
  }
}

BaseRef PrimitivePy::RunOpBpropFunction(const py::tuple &py_args) const {
  if (backward_hook_fn_.size() > 1) {
    MS_LOG(EXCEPTION) << "Multiple registration of bprop function is not supported.";
  }
  py::tuple grads;
  SyncData(py_args);
  py::tuple converted_args(py_args.size());
  ConvertCTensorToPyTensor(py_args, &converted_args);
  try {
    MS_LOG(DEBUG) << "start execute custom op bprop";
    for (const auto &elem : backward_hook_fn_) {
      py::object grads_obj = elem.second(*converted_args);
      grads = check_bprop_out(grads_obj, py_args, bprop_cls_name_);
    }
    return std::make_shared<PyObjectRef>(grads);
  } catch (std::exception &bt) {
    std::rethrow_exception(std::current_exception());
  }
}

BaseRef PrimitivePy::RunCellHookFunction(const py::tuple &py_args) const {
  // Get the gradient passed to current bprop cut op.
  const auto args_size = py_args.size();
  py::object grad_output = py_args[args_size - 1];
  // Get the cell id.
  auto cell_id = GetValue<std::string>(this->GetAttr(kCellIDAttrName));
  auto iter = hook_grad_.find(cell_id);
  if (iter != hook_grad_.end()) {
    // The second bprop_cut used to hook output gradient of cell.
    for (const auto &elem : backward_hook_fn_) {
      py::object code_obj = py::getattr(elem.second, "__code__");
      py::object co_name = py::getattr(code_obj, "co_name");
      if (std::string(py::str(co_name)) == "staging_specialize") {
        py::object name_obj = py::getattr(elem.second, "__name__");
        MS_LOG(EXCEPTION) << "Decorating hook function " << py::str(name_obj) << " with '@jit' is not supported.";
      }
      SyncData(grad_output);
      py::tuple hook_fn_args = ConstructCellHookFnArgs(cell_id, iter->second, grad_output);
      py::object ret = elem.second(*hook_fn_args);
      if (!py::isinstance<py::none>(ret)) {
        grad_output = UnpackRetValueOfCellHook(ret);
      }
      CheckHookConsistency(grad_output, py_args[args_size - 1], code_obj, co_name);
    }
    (void)hook_grad_.erase(cell_id);
  } else {
    // The first bprop_cut used to hook input gradient of cell.
    SyncData(grad_output);
    hook_grad_[cell_id] = grad_output;
  }
  if (!py::isinstance<py::tuple>(grad_output)) {
    grad_output = py::make_tuple(grad_output);
  }
  return std::make_shared<PyObjectRef>(grad_output);
}

BaseRef PrimitivePy::RunVariableHookFunction(const py::tuple &py_args) const {
  constexpr size_t grad_output_index = 2;
  py::object grad_output = py_args[grad_output_index];
  for (const auto &elem : backward_hook_fn_) {
    py::object code_obj = py::getattr(elem.second, "__code__");
    py::object co_name = py::getattr(code_obj, "co_name");
    if (std::string(py::str(co_name)) == "staging_specialize") {
      py::object name_obj = py::getattr(elem.second, "__name__");
      MS_LOG(EXCEPTION) << "Decorating hook function " << py::str(name_obj) << " with '@jit' is not supported.";
    }
    SyncData(grad_output);
    py::object ret = elem.second(py::make_tuple(grad_output));
    if (!py::isinstance<py::none>(ret)) {
      grad_output = ret;
    }
    CheckHookConsistency(grad_output, py_args[grad_output_index], code_obj, co_name);
  }
  grad_output = py::make_tuple(grad_output);
  return std::make_shared<PyObjectRef>(grad_output);
}

BaseRef PrimitivePy::RunHookFunction(const VectorRef &args) const {
  py::tuple py_args = ConvertDatatoPyTuple(args);
  bool is_bprop = this->HasAttr(kBpropAttrName);
  if (is_bprop) {
    return RunCellBpropFunction(py_args);
  }
  bool is_cell = this->HasAttr(kCellHookAttrName);
  if (is_cell) {
    return RunCellHookFunction(py_args);
  }
  bool is_custom_op_bprop = this->HasAttr(kCustomOpBpropAttrName);
  if (is_custom_op_bprop) {
    return RunOpBpropFunction(py_args);
  }
  return RunVariableHookFunction(py_args);
}

py::function PrimitivePy::GetComputeFunction() const {
  static const char *const compute_func_name = "vm_impl";

  if (py::hasattr(python_obj_, compute_func_name)) {
    MS_LOG(DEBUG) << name() << " compute_func_name";
    py::function fn = python_obj_.attr(compute_func_name).cast<py::function>();
    return fn;
  }

  static const std::string vm_module = "mindspore.ops.vm_impl_registry";
  static const std::string get_vm_impl_fn = "get_vm_impl_fn";
  MS_LOG(DEBUG) << name() << ": get_vm_impl_fn";
  py::function get_fn = python_adapter::GetPyFn(vm_module, get_vm_impl_fn);
  py::function vm_fn = get_fn(python_obj_);
  if (py::isinstance<py::none>(vm_fn)) {
    MS_LOG(DEBUG) << "Cannot find " << python_obj_.attr("__class__").attr("__name__").cast<std::string>();
    vm_fn = mindspore::GetComputeFunction(Primitive::name());
  }
  return vm_fn;
}

py::dict PrimitivePy::GetAttrDict() {
  py::dict attr_dict;
  for (auto &attr : attrs_) {
    attr_dict[py::str(attr.first)] = ValueToPyData(attr.second);
  }
  return attr_dict;
}

void PrimitivePy::CopyHookFunction(const PrimitivePyPtr &primitive_py) {
  MS_EXCEPTION_IF_NULL(primitive_py);
  const auto &backward_hook_fn = primitive_py->backward_hook_fn();
  for (const auto &elem : backward_hook_fn) {
    AddBackwardHookFn(elem.first, elem.second);
  }
  if (primitive_py->HasAttr(kBpropAttrName)) {
    set_bprop_cls_name(primitive_py->bprop_cls_name_);
    (void)this->AddAttr(kBpropAttrName, primitive_py->GetAttr(kBpropAttrName));
  }
}

BaseRef PrimitivePy::RunComputeFunction(const VectorRef &args) const {
  auto py_args = ConvertDatatoPyTuple(args);
  auto result = this->RunPyComputeFunction(py_args);
  if (py::isinstance<py::none>(result)) {
    return std::make_shared<BaseRef>(nullptr);
  }
  return std::make_shared<PyObjectRef>(result);
}

py::object PrimitivePy::RunPyComputeFunction(const py::tuple &py_args) const {
  auto func = this->GetComputeFunction();
  if (py::isinstance<py::none>(func)) {
    return py::none();
  }
  auto result = func(*py_args);
  return result;
}

bool PrimitivePy::HasComputeFunction() const {
  auto func = GetComputeFunction();
  return !py::isinstance<py::none>(func);
}

PrimitivePtr PrimitivePy::Clone() {
  auto clone_fn = python_obj_.attr("_clone");
  py::object obj_adapter = clone_fn();
  auto prim_adapter = obj_adapter.cast<PrimitivePyAdapterPtr>();
  auto prim = std::make_shared<PrimitivePy>(obj_adapter, prim_adapter);
  prim_adapter->set_attached_primitive(prim);
  return prim;
}

py::dict PrimitivePy::RunInfer(const py::tuple &args) {
  if (!HasPyObj()) {
    MS_LOG(EXCEPTION) << "[" << this->ToString() << "]: pyobj is empty";
  }
  // Python obj could be replaced as None, so it will losed the original info when throw exception in python.
  if (!py::hasattr(python_obj_, PY_PRIM_METHOD_INFER)) {
    MS_LOG(EXCEPTION) << "prim:" << ToString() << " has no attr:" << PY_PRIM_METHOD_INFER;
  }
  auto infer_fuc = python_obj_.attr(PY_PRIM_METHOD_INFER);
  return infer_fuc(*args);
}

void PrimitivePy::RunCheck(const py::tuple &args) {
  if (!HasPyObj()) {
    MS_LOG(EXCEPTION) << "[" << this->ToString() << "]: pyobj is empty";
  }
  // Python obj could be replaced as None, so it will losed the original info when throw exception in python.
  if (!py::hasattr(python_obj_, PY_PRIM_METHOD_CHECK)) {
    MS_LOG(EXCEPTION) << "prim:" << ToString() << " has no attr:" << PY_PRIM_METHOD_CHECK;
  }
  auto check_func = python_obj_.attr(PY_PRIM_METHOD_CHECK);
  (void)check_func(*args);
}

py::object PrimitivePy::RunInferValue(const py::tuple &args) {
  if (!HasPyObj()) {
    MS_LOG(EXCEPTION) << "[" << this->ToString() << "]: pyobj is empty";
  }
  // Python obj could be replaced as None, so it will losed the original info when throw exception in python.
  if (!py::hasattr(python_obj_, PY_PRIM_METHOD_INFER_VALUE)) {
    MS_LOG(EXCEPTION) << "prim:" << ToString() << " has no attr:" << PY_PRIM_METHOD_INFER_VALUE;
  }
  auto infer_value = python_obj_.attr(PY_PRIM_METHOD_INFER_VALUE);
  return infer_value(*args);
}

void PrimitivePy::ClearHookRes() { hook_grad_.clear(); }

PrimitivePyAdapter::PrimitivePyAdapter(const py::str &name) : name_(name) {}

void PrimitivePyAdapter::AddPyAttr(const py::str &name, const py::object &obj) {
  std::string attr_name = name;
  ValuePtr converted_ret = nullptr;
  if (py::isinstance<py::module>(obj)) {
    MS_LOG(EXCEPTION) << "Call 'add_attr' to add attribute to primitive failed,"
                      << " not support py::module to be attribute value; primitive name: " << this->name_
                      << ", attribute name: " << attr_name << " attribute value: " << py::str(obj);
  }
  bool converted = parse::ConvertData(obj, &converted_ret);
  if (!converted) {
    MS_LOG(EXCEPTION) << "Call 'add_attr' to add attribute to primitive failed,"
                      << " convert python obj to MindSpore obj failed; primitive name: " << this->name_
                      << ", attribute name:" << attr_name << ", attribute value:" << py::str(obj)
                      << ", attribute type:" << py::cast<std::string>(obj.attr("__class__").attr("__name__"));
  }
  if (kOpAttrNameReplaceMap.find(attr_name) != kOpAttrNameReplaceMap.end()) {
    attr_name = kOpAttrNameReplaceMap[attr_name];
  }
  (void)CheckAndConvertUtils::ConvertAttrValueToInt(this->name_, name, &converted_ret);
  if (attr_name == "primitive_target") {
    MS_EXCEPTION_IF_NULL(converted_ret);
    if (!converted_ret->isa<StringImm>()) {
      MS_LOG(EXCEPTION) << "Call 'add_attr' to add attribute to primitive '" << this->name_
                        << "' failed, value of attribute 'primitive_target' must be CPU|GPU|Ascend but got "
                        << py::str(obj);
    }
    auto target = GetValue<std::string>(converted_ret);
    if (!target.empty() && target != kCPUDevice && target != kGPUDevice && target != kAscendDevice &&
        target != "Device") {
      MS_LOG(EXCEPTION) << "Call 'add_attr' to add attribute to primitive '" << this->name_
                        << "' failed, value of attribute 'primitive_target' must be CPU|GPU|Ascend but got "
                        << py::str(obj);
    }
  }

  attrs_[attr_name] = converted_ret;
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    (void)prim->AddAttr(attr_name, converted_ret);
  }
}

void PrimitivePyAdapter::DelPyAttr(const py::str &name) {
  (void)attrs_.erase(name);
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    (void)prim->DelAttr(name);
  }
}

py::dict PrimitivePyAdapter::GetAttrDict() {
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    return prim->GetAttrDict();
  }

  py::dict attr_dict;
  for (auto &attr : attrs_) {
    attr_dict[py::str(attr.first)] = ValueToPyData(attr.second);
  }
  return attr_dict;
}

void PrimitivePyAdapter::set_prim_type(const PrimType t) {
  prim_type_ = t;
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    prim->set_prim_type(t);
  }
}
void PrimitivePyAdapter::set_const_prim(bool is_const_prim) {
  is_const_prim_ = is_const_prim;
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    prim->set_const_prim(is_const_prim);
  }
}
void PrimitivePyAdapter::set_const_input_indexes(const std::vector<size_t> &const_input_indexes) {
  const_input_indexes_ = const_input_indexes;
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    prim->set_const_input_indexes(const_input_indexes);
  }
}

void PrimitivePyAdapter::set_signatures(const std::vector<Signature> &signatures) {
  signatures_ = signatures;
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    prim->set_signatures(signatures);
  }
}

int PrimitivePyAdapter::AddBackwardHookFn(const py::function &backward_hook_fn) {
  ++backward_hook_fn_key_;
  backward_hook_fn_[backward_hook_fn_key_] = backward_hook_fn;
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    prim->AddBackwardHookFn(backward_hook_fn_key_, backward_hook_fn);
  }
  return backward_hook_fn_key_;
}

void PrimitivePyAdapter::RemoveBackwardHookFn(int key) {
  const auto iter = backward_hook_fn_.find(key);
  if (iter != backward_hook_fn_.end()) {
    (void)backward_hook_fn_.erase(iter);
  }
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    prim->RemoveBackwardHookFn(key);
  }
}

void PrimitivePyAdapter::set_instance_name(const std::string &s) {
  instance_name_ = s;
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    prim->set_instance_name(s);
  }
}

void PrimitivePyAdapter::set_attached_primitive(const PrimitivePyPtr &prim) {
  if (attached_primitive_.lock() != nullptr) {
    MS_LOG(EXCEPTION) << "PrimitivePyAdapter can't attach to multi Primitive.";
  }
  MS_EXCEPTION_IF_NULL(prim);
  attached_primitive_ = prim;
}

void RegPrimitive(const py::module *m) {
  (void)py::enum_<PrimType>(*m, "prim_type", py::arithmetic())
    .value("unknown", PrimType::kPrimTypeUnknown)
    .value("builtin", PrimType::kPrimTypeBuiltIn)
    .value("py_infer_shape", PrimType::kPrimTypePyInfer)
    .value("user_custom", PrimType::kPrimTypeUserCustom)
    .value("py_infer_check", PrimType::kPrimTypePyCheck);
  (void)py::class_<PrimitivePyAdapter, std::shared_ptr<PrimitivePyAdapter>>(*m, "Primitive_")
    .def_readonly(PYTHON_PRIMITIVE_FLAG, &PrimitivePyAdapter::parse_info_)
    .def(py::init<py::str &>())
    .def("add_attr", &PrimitivePyAdapter::AddPyAttr, "add primitive attr")
    .def("del_attr", &PrimitivePyAdapter::DelPyAttr, "del primitive attr")
    .def("get_attr_dict", &PrimitivePyAdapter::GetAttrDict, "get primitive attr")
    .def("set_prim_type", &PrimitivePyAdapter::set_prim_type, "Set primitive type.")
    .def("set_const_prim", &PrimitivePyAdapter::set_const_prim, "Set primitive is const.")
    .def("set_const_input_indexes", &PrimitivePyAdapter::set_const_input_indexes, "Set primitive const input indexes.")
    .def("set_signatures", &PrimitivePyAdapter::set_signatures, "Set primitive inputs signature.")
    .def("add_backward_hook_fn", &PrimitivePyAdapter::AddBackwardHookFn, "Add primitive backward hook function.")
    .def("remove_backward_hook_fn", &PrimitivePyAdapter::RemoveBackwardHookFn,
         "Remove primitive backward hook function.")
    .def("set_instance_name", &PrimitivePyAdapter::set_instance_name, "Set primitive instance name.");
}
}  // namespace mindspore
