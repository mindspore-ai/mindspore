/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/py_func.h"

#include <pybind11/pybind11.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "include/utils/convert_utils_py.h"
#include "include/utils/tensor_py.h"
#include "ir/tensor.h"
#include "ops/op_def.h"
#include "ops_utils/callback.h"

namespace mindspore::ops {
using mindspore::tensor::Tensor;
using mindspore::tensor::TensorPtr;
namespace {
constexpr auto kAttrGroupedInfo = "grouped_info";

// Convert a sequence of InferInfo to a ValueTuple of Tensors.
ValuePtr ConvertTensorListToValue(const InferInfoPtr &info) {
  MS_EXCEPTION_IF_NULL(info);
  if (!info->IsSequence()) {
    MS_LOG(EXCEPTION) << "Expected sequence, got: " << info->DebugInfo();
  }

  const auto &elems = info->GetSequenceElements();
  std::vector<ValuePtr> tensors;
  tensors.reserve(elems.size());

  std::transform(elems.begin(), elems.end(), std::back_inserter(tensors),
                 [](const auto &elem) { return std::make_shared<Tensor>(elem->GetType(), elem->GetShape()); });

  return std::make_shared<ValueTuple>(std::move(tensors));
}

// Pack a sequence of primitive values into a Value.
template <typename T>
ValuePtr PackSequence(const InferInfoPtr &info) {
  auto opt = info->GetArrayValue<T>();
  if (!opt.has_value()) {
    MS_LOG(EXCEPTION) << "PackSequence<" << typeid(T).name() << "> failed: " << info->DebugInfo();
  }
  return MakeValue(opt.value().ToVector());
}

// Wrap a scalar value into a Value.
template <typename T>
ValuePtr MakeScalar(const InferInfoPtr &info) {
  return MakeValue(info->GetScalarValueWithCheck<T>());
}

// Convert a numeric InferInfo to a Value.
ValuePtr ConvertNumberToValue(const InferInfoPtr &info) {
  MS_EXCEPTION_IF_NULL(info);
  const auto type = info->GetType();
  switch (type) {
    case kNumberTypeInt64:
      return MakeScalar<int64_t>(info);
    case kNumberTypeBool:
      return MakeScalar<bool>(info);
    case kNumberTypeFloat32:
      return MakeScalar<float>(info);
    default:
      MS_LOG(EXCEPTION) << "Unsupported number type: " << TypeIdToString(type);
  }
}

using Handler = std::function<ValuePtr(const InferInfoPtr &)>;

// Mapping from OP_DTYPE to converter function.
static const std::unordered_map<OP_DTYPE, Handler> kInferInfo2Value{
  {DT_TENSOR, [](const InferInfoPtr &i) { return std::make_shared<Tensor>(i->GetType(), i->GetShape()); }},
  {DT_BOOL, &MakeScalar<bool>},
  {DT_INT, &MakeScalar<int64_t>},
  {DT_FLOAT, &MakeScalar<float>},
  {DT_STR, &MakeScalar<std::string>},
  {DT_NUMBER, &ConvertNumberToValue},
  {DT_LIST_BOOL, &PackSequence<bool>},
  {DT_TUPLE_BOOL, &PackSequence<bool>},
  {DT_LIST_INT, &PackSequence<int64_t>},
  {DT_TUPLE_INT, &PackSequence<int64_t>},
  {DT_LIST_FLOAT, &PackSequence<float>},
  {DT_TUPLE_FLOAT, &PackSequence<float>},
  {DT_LIST_STR, &PackSequence<std::string>},
  {DT_TUPLE_STR, &PackSequence<std::string>},
  {DT_LIST_TENSOR, &ConvertTensorListToValue},
  {DT_TUPLE_TENSOR, &ConvertTensorListToValue},
};

// Convert InferInfo to Value according to dtype.
ValuePtr GetValueByInferInfo(const InferInfoPtr &info, OP_DTYPE dtype) {
  MS_EXCEPTION_IF_NULL(info);
  const auto it = kInferInfo2Value.find(dtype);
  if (it == kInferInfo2Value.end()) {
    MS_LOG(EXCEPTION) << "Unsupported OP_DTYPE: " << static_cast<int>(dtype);
  }
  return it->second(info);
}

// Retrieve the Python callable object by its registration id.
pybind11::function GetPythonFunc(int64_t func_id) {
  pybind11::gil_scoped_acquire gil_acquire;
  static constexpr auto kModuleName = "mindspore.ops.operations._pyfunc_registry";
  static constexpr auto kEntrance = "get_pyfunc";

  pybind11::module mod = pybind11::module::import(kModuleName);
  pybind11::object get_pyfunc_obj = mod.attr(kEntrance);
  if (get_pyfunc_obj.is_none()) {
    MS_LOG(EXCEPTION) << "Cannot find `" << kEntrance << "` in module " << kModuleName;
  }

  pybind11::function get_pyfunc = get_pyfunc_obj.cast<pybind11::function>();
  pybind11::object py_func_obj = get_pyfunc(pybind11::int_(func_id));
  if (py_func_obj.is_none()) {
    MS_LOG(EXCEPTION) << "Cannot find python func with id: " << func_id;
  }
  return py_func_obj.cast<pybind11::function>();
}

// Record dynamic input sizes for grouped inputs.
void SetDynInputSizes(const PrimitivePtr &primitive, const InferInfoPtrList &inputs,
                      const std::vector<OpInputArg> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  if (primitive->HasAttr(kAttrGroupedInfo)) {
    return;
  }

  std::vector<int64_t> dyn_input_sizes;
  dyn_input_sizes.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    MS_EXCEPTION_IF_NULL(inputs[i]);
    const auto arg_dtype = input_args[i].arg_dtype_;
    if (arg_dtype == OP_DTYPE::DT_TUPLE_TENSOR || arg_dtype == OP_DTYPE::DT_LIST_TENSOR) {
      dyn_input_sizes.emplace_back(SizeToLong(inputs[i]->GetSequenceElements().size()));
    } else if (inputs[i]->IsNone()) {
      dyn_input_sizes.emplace_back(0);
    } else {
      dyn_input_sizes.emplace_back(1);
    }
  }
  primitive->set_attr(kAttrGroupedInfo, MakeValue(dyn_input_sizes));
}

// Convert Python inference result to vector of TensorPtr.
std::vector<TensorPtr> PyInferResultToTensor(const pybind11::object &py_ret) {
  std::vector<TensorPtr> outs;
  static auto func_is_tensor_py = GET_OPS_CALLBACK(IsTensorPy, bool, const pybind11::object &);
  MS_EXCEPTION_IF_NULL(func_is_tensor_py);
  static auto func_convert_py_to_tensor = GET_OPS_CALLBACK(ConvertPyObjToTensor, TensorPtr, const pybind11::object &);
  MS_EXCEPTION_IF_NULL(func_convert_py_to_tensor);
  try {
    if (func_is_tensor_py(py_ret)) {
      outs.emplace_back(func_convert_py_to_tensor(py_ret));
      return outs;
    }

    if (pybind11::isinstance<pybind11::tuple>(py_ret) || pybind11::isinstance<pybind11::list>(py_ret)) {
      auto seq = pybind11::cast<pybind11::sequence>(py_ret);
      outs.reserve(seq.size());

      std::transform(seq.begin(), seq.end(), std::back_inserter(outs), [](const auto &item) {
        if (!func_is_tensor_py(item)) {
          MS_LOG(EXCEPTION) << "Element inside tuple/list is not a Tensor.";
        }
        return func_convert_py_to_tensor(item);
      });

      return outs;
    }
  } catch (const std::exception &e) {
    MS_LOG(EXCEPTION) << "Failed to convert Python inference result to Tensor: " << e.what();
  }

  MS_LOG(EXCEPTION) << "Infer function must return Tensor/tuple/list of Tensor.";
}

// Invoke the Python inference function and return resulting tensors.
std::vector<TensorPtr> CallPyInfer(const PrimitivePtr &primitive, const InferInfoPtrList &inputs) {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &name = primitive->name();
  const auto op_def = mindspore::ops::GetOpDef(name);
  MS_EXCEPTION_IF_NULL(op_def);
  auto &input_args = op_def->args_;

  if (input_args.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Custom node `" << name << "` op_def.inputs.size()=" << input_args.size()
                      << " != inputs.size()=" << inputs.size();
  }

  SetDynInputSizes(primitive, inputs, input_args);

  if (Py_IsInitialized() == 0) {
    MS_LOG(EXCEPTION) << "Python interpreter not initialized.";
  }

  const auto func_id = GetValue<int64_t>(primitive->GetAttr("infer_func_id"));
  const auto py_func = GetPythonFunc(func_id);

  pybind11::gil_scoped_acquire gil_acquire;
  pybind11::tuple args(inputs.size());
  static auto func_value_to_py_data =
    GET_OPS_CALLBACK(ValueToPyData, py::object, const ValuePtr &, const AbstractBasePtr &);
  MS_EXCEPTION_IF_NULL(func_value_to_py_data);
  for (size_t i = 0; i < inputs.size(); ++i) {
    args[i] = func_value_to_py_data(GetValueByInferInfo(inputs[i], input_args[i].arg_dtype_), nullptr);
  }

  pybind11::object result = inputs.empty() ? py_func() : py_func(*args);
  return PyInferResultToTensor(result);
}

}  // namespace

// Infer output shapes by calling the Python inference function.
ShapeArray PyFuncInferImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &inputs) const {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_LOG(DEBUG) << "Start infer shape for " << primitive->name();
  const auto tensors = CallPyInfer(primitive, inputs);

  ShapeArray shapes;
  shapes.reserve(tensors.size());

  std::transform(tensors.begin(), tensors.end(), std::back_inserter(shapes), [](const auto &t) { return t->shape(); });

  MS_LOG(DEBUG) << "End.";
  return shapes;
}

// Infer output types by calling the Python inference function.
std::vector<TypeId> PyFuncInferImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &inputs) const {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_LOG(DEBUG) << "Start infer type for " << primitive->name();
  const auto tensors = CallPyInfer(primitive, inputs);

  std::vector<TypeId> types;
  types.reserve(tensors.size());

  std::transform(tensors.begin(), tensors.end(), std::back_inserter(types),
                 [](const auto &t) { return t->data_type(); });

  MS_LOG(DEBUG) << "End";
  return types;
}

}  // namespace mindspore::ops
