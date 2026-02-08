/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "pynative/backward/hook/custom_function.h"
#include <memory>
#include "pynative/backward/hook/function_py.h"
#include "include/runtime/pipeline/pipeline.h"
#include "pynative/backward/op_grad/func_builder.h"
#include "include/pynative/utils/pynative_execute.h"
#include "include/utils/python_adapter.h"
#include "include/utils/convert_utils_py.h"
#include "include/utils/tensor_py.h"
#include "include/pynative/utils/pynative_utils.h"
#include "include/utils/pynative/common_utils.h"
#include "mindspore/ccsrc/pynative/backward/op_grad/func_grad.h"

namespace mindspore {
namespace pynative {
namespace autograd {
namespace {
// The arguments of backward function are ctx and gradients correspongding to outputs of forward function.
PyObject *ConstructBackwardArgs(PyObject *ctx, PyObject *py_tensor_grad) {
  auto num_args = PyTuple_Check(py_tensor_grad) ? 1 + PyTuple_GET_SIZE(py_tensor_grad) : 2;
  PyObject *res = PyTuple_New(num_args);
  Py_INCREF(ctx);
  PyTuple_SetItem(res, 0, ctx);
  if (PyTuple_Check(py_tensor_grad)) {
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(py_tensor_grad); i++) {
      auto grad = PyTuple_GET_ITEM(py_tensor_grad, i);
      Py_INCREF(grad);
      PyTuple_SetItem(res, i + 1, grad);
    }
  } else {
    Py_INCREF(py_tensor_grad);
    PyTuple_SetItem(res, 1, py_tensor_grad);
  }
  return res;
}

ValuePtr ValueListToValue(const ValuePtrList &list) {
  if (list.size() == kSizeZero) {
    MS_LOG(EXCEPTION) << "Value ptr list should not be empty";
  }
  if (list.size() == kSizeOne) {
    return list[kIndex0];
  }
  return std::make_shared<ValueTuple>(list);
}
}  // namespace

CustomBackward::~CustomBackward() {
  py::gil_scoped_acquire gil_acquire;
  bprop_fn_ = py::object();
}

ValuePtrList CustomBackward::CallBackward(const ValuePtrList &grads) {
  runtime::Pipeline::Get().WaitFrontend();
  MS_LOG(DEBUG) << "Begin CustomBackwardNode CallBackward ";
  auto gradient = PyNativeAlgo::DataConvert::ValueListToValue(grads, out_abstract_);
  const auto &device_target = DeviceManagerConf::GetInstance()->device_type();
  // Python grad func can not process None, we need to convert None to zero tensor.
  auto func_builder = FuncBuilder(name_, device_target, nullptr);
  auto filled_zeros_grad = func_builder.FillZeros(gradient, out_abstract_);

  auto bprop_inputs = SavedValueListToValueList(saved_values_, shared_from_this());
  bprop_inputs.emplace_back(filled_zeros_grad);

  // Run bprop function.
  py::gil_scoped_acquire gil_acquire;
  auto bprop_fn_args = py::reinterpret_steal<py::tuple>(tensor::Wrap(bprop_inputs));
  py::object grads_obj = bprop_fn_(*bprop_fn_args);
  py::tuple input_grads = CheckBpropOut(grads_obj, bprop_fn_args, name());
  MS_LOG(DEBUG) << "Run cell custom bprop function end.";
  ValuePtrList gradient_values;
  ConvertPyObjectToCTensor(input_grads, &gradient_values, true);
  if (gradient_values.empty()) {
    MS_LOG(EXCEPTION) << "Hook fn grad output is empty!";
  }
  auto gradient_tensors = PostProcess(gradient_values);
  MS_LOG(DEBUG) << "End HookBackwardNode CallBackward";
  runtime::Pipeline::Get().WaitFrontend();
  return gradient_tensors;
}

ValuePtrList CustomBackward::PostProcess(const ValuePtrList &gradient_value) {
  auto flatten_gradients = CommonUtils::FlattenTensorSeqInValueSeq(gradient_value, false);
  // Zero3 algorithm may split input tensor, which causes gradients shape not same as input tensor.
  // It will split gradient tensor by tensor hook after recompute. So here input tensor shape
  // may different from gradient tensor.
  if (is_recompute_) {
    return flatten_gradients;
  }
  return AutoGradUtil::AutoCastAndReduce(flatten_gradients, AutoGradUtil::GenerateInputsMeta(next_edges()));
}

void CustomBackward::Release() {
  saved_values_.clear();
  py::gil_scoped_acquire gil_acquire;
  bprop_fn_ = py::object();
}

ValuePtrList PyBackwardNode::CallBackward(const ValuePtrList &grads) {
  runtime::Pipeline::Get().WaitFrontend();
  MS_LOG(DEBUG) << "Begin PyBackwardNode CallBackward";
  // Construct input for backward function.
  py::gil_scoped_acquire gil_acquire;
  auto gradients = ValueListToValue(grads);
  FunctionBase *ctx = reinterpret_cast<FunctionBase *>(obj_.ptr());
  MS_EXCEPTION_IF_NULL(ctx);
  PyObject *py_tensor_grad;
  if (ctx->materialize_grads) {
    const auto &device_target = DeviceManagerConf::GetInstance()->device_type();
    // Python grad func can not process None, we need to convert None to zero tensor.
    auto func_builder = FuncBuilder(name_, device_target, nullptr);
    auto filled_zeros_grad = func_builder.FillZeros(gradients, out_abstract_);
    py_tensor_grad = tensor::Wrap(filled_zeros_grad);
  } else {
    py_tensor_grad = tensor::Wrap(gradients);
  }
  MS_LOG(DEBUG) << "Args info, grad is tuple " << PyTuple_Check(py_tensor_grad) << ", is tensor input size "
                << ctx->is_tensor_input.size() << "materialize_grads " << ctx->materialize_grads;
  auto fn_args = py::reinterpret_steal<py::object>(ConstructBackwardArgs(obj_.ptr(), py_tensor_grad));
  Py_DECREF(py_tensor_grad);
  // Call python backward function.
  auto grads_obj = py::reinterpret_steal<py::object>(PyObject_CallObject(backward_fn_.ptr(), fn_args.ptr()));
  if (!grads_obj) {
    throw py::error_already_set();
  }
  (void)ensure_obj_tuple(&grads_obj);
  auto grad_tuple = py::cast<py::tuple>(grads_obj);
  size_t num_backward_out = grad_tuple.size();
  size_t num_forward_in = ctx->is_tensor_input.size();
  if (num_backward_out < num_forward_in) {
    MS_LOG(EXCEPTION) << "Function backward return a wrong number of gradients, expect: " << num_forward_in
                      << "but: " << num_backward_out;
  }

  for (size_t i = 0; i < num_backward_out; i++) {
    bool is_tensor = ctx->is_tensor_input[i];
    py::object output = grad_tuple[i];
    // The gradient of Input that is not tensor should be none.
    if (!is_tensor && !py::isinstance<py::none>(output)) {
      MS_LOG(EXCEPTION) << "Input is not tensor, but gradient is not none, position: " << i
                        << " type: " << output.get_type();
    }
    // The gradient should be either none or tensor.
    if (!py::isinstance<py::none>(output) && !tensor::IsTensorPy(output)) {
      MS_LOG(EXCEPTION) << "Gradient should be none or tensor, position: " << i << " type: " << output.get_type();
    }
  }

  // Convert python object to tensor.
  ValuePtrList gradient_values;
  ConvertPybindTupleGradToCValue(grad_tuple, &gradient_values);
  if (gradient_values.empty()) {
    MS_LOG(EXCEPTION) << "Custom backward function output is empty!";
  }
  runtime::Pipeline::Get().WaitFrontend();
  auto gradient_tensors = PostProcess(gradient_values);
  MS_LOG(DEBUG) << "End PyBackwardNode CallBackward";
  return gradient_tensors;
}

ValuePtrList PyBackwardNode::PostProcess(const ValuePtrList &gradient_value) {
  return AutoGradUtil::AutoCastAndReduce(gradient_value, AutoGradUtil::GenerateInputsMeta(next_edges()));
}

void PyBackwardNode::Release() {
  py::gil_scoped_acquire gil_acquire;
  backward_fn_ = py::object();
  obj_ = py::object();
  saved_tensors_.clear();
}

PyBackwardNode::~PyBackwardNode() {
  py::gil_scoped_acquire gil_acquire;
  backward_fn_ = py::object();
  obj_ = py::object();
}
}  // namespace autograd
}  // namespace pynative
}  // namespace mindspore
