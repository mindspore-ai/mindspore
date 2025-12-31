/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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
#include "frontend/operator/py_execute_py.h"  // Only include one-time in the whole project.

#include <vector>
#include <string>
#include <memory>
#include <utility>

#include "ir/tensor_new.h"
#include "pybind11/pybind11.h"
#include "pybind_api/pybind_patch.h"

#include "include/utils/pipeline/fallback.h"
#include "mindspore/ops/infer/py_execute.h"
#include "include/utils/anfalgo.h"
#include "include/utils/convert_utils_py.h"
#include "include/utils/python_utils.h"
#include "include/utils/tensor_py.h"
#include "include/utils/python_adapter.h"
#include "include/utils/pipeline/python_fallback_running.h"
#include "mindspore/ccsrc/include/backend/common/pass_manager/helper.h"
#include "mindspore/ccsrc/frontend/jit/ps/parse/data_converter.h"
#include "mindspore/ccsrc/frontend/jit/ps/parse/resolve.h"

namespace py = pybind11;
namespace mindspore {
ValuePtr PyExecuteInitializer::GetValueByAbstract(const abstract::AbstractBase *abstract) {
  MS_EXCEPTION_IF_NULL(abstract);
  if (!abstract->isa<kernel::KernelTensor>()) {
    MS_LOG(EXCEPTION) << "Invalid kernel tensor:" << abstract->ToString();
  }
  const auto &kernel_tensor = dynamic_cast<const kernel::KernelTensor *>(abstract);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  if (kernel_tensor->user_data() != nullptr) {
    const auto &user_data =
      kernel_tensor->user_data()->get<kernel::PyExecuteOutputUserData>(kernel::PyExecuteOutputUserData::key);
    MS_EXCEPTION_IF_NULL(user_data);
    return std::make_shared<parse::PyObjectWrapper>(user_data->obj, "graph python obj");
  }

  if (kernel_tensor->GetValueTrack() != nullptr && !kernel_tensor->GetValueTrack()->isa<ValueAny>()) {
    return kernel_tensor->GetValueTrack();
  } else if (IsShapeEmpty(kernel_tensor->GetShapeVector())) {
    auto type_id =
      (kernel_tensor->dtype_id() == TypeId::kTypeUnknown ? TypeId::kNumberTypeInt64 : kernel_tensor->dtype_id());
    return tensor::from_spec(type_id, kernel_tensor->GetShapeVector(), device::DeviceType::kCPU);
  }

  MS_LOG(DEBUG) << "Type:" << kernel_tensor->dtype_id() << " shape:" << kernel_tensor->GetShapeVector()
                << " size:" << kernel_tensor->size();
  auto real_value = kernel_tensor->GetValue();
  MS_EXCEPTION_IF_NULL(real_value);
  if (!real_value->isa<KernelTensorValue>()) {
    MS_LOG(EXCEPTION) << "Invalid kernel tensor value:" << real_value->ToString();
  }

  auto kernel_tensor_value = real_value->cast<KernelTensorValuePtr>();
  MS_EXCEPTION_IF_NULL(kernel_tensor_value);
  if (kernel_tensor->GetType() != nullptr && kernel_tensor->GetType()->isa<Number>()) {
    return common::AnfAlgo::ValueToScalar(kernel_tensor_value, kernel_tensor->GetType()->type_id());
  }

  tensor::TensorPtr tensor =
    tensor::from_spec(kernel_tensor->dtype_id(), kernel_tensor->GetShapeVector(), device::DeviceType::kCPU);
  MS_EXCEPTION_IF_NULL(tensor);
  if (LongToSize(tensor->DataNBytes()) != kernel_tensor_value->GetDataSize()) {
    MS_LOG(EXCEPTION) << "Invalid host tensor size:" << tensor->DataNBytes()
                      << " and kernel tensor size:" << kernel_tensor_value->GetDataSize() << " for pyexecute.";
  }
  if (kernel_tensor_value->GetDataSize() == 0) {
    return tensor;
  }
  auto data_ptr = tensor->data_c();
  MS_EXCEPTION_IF_NULL(data_ptr);
  const auto &res = memcpy_s(data_ptr, kernel_tensor_value->GetDataSize(), kernel_tensor_value->GetDataPtr(),
                             kernel_tensor_value->GetDataSize());
  if (res != EOK) {
    MS_LOG(EXCEPTION) << "memcpy failed. res: " << res << ", for tensor:" << tensor->ToString()
                      << " size:" << kernel_tensor_value->GetDataSize();
  }
  return tensor;
}

ValuePtr PyExecuteInitializer::ConstructEmptyTupleValue(const ValuePtr &structural) {
  MS_EXCEPTION_IF_NULL(structural);
  if (!structural->isa<ValueTuple>()) {
    MS_LOG(EXCEPTION) << "input abstract is out of range.";
  }
  auto value_tuple = structural->cast_ptr<ValueTuple>();
  MS_EXCEPTION_IF_NULL(value_tuple);

  std::vector<ValuePtr> values;
  for (size_t i = 0; i < value_tuple->size(); ++i) {
    auto item = (*value_tuple)[i];
    (void)values.emplace_back(ConstructEmptyTupleValue(item));
  }

  return std::make_shared<ValueTuple>(values);
}

std::pair<ValuePtr, size_t> PyExecuteInitializer::ConstructInputValue(
  const ValuePtr &value, const std::vector<abstract::AbstractBase *> &input_abstract, size_t input_index) {
  MS_EXCEPTION_IF_NULL(value);
  auto begin_iter = input_abstract.begin() + input_index;
  if (value->isa<ValueSequence>()) {
    size_t offset = 0;
    std::vector<ValuePtr> values;
    auto seq_value = value->cast_ptr<ValueSequence>();
    MS_EXCEPTION_IF_NULL(seq_value);
    for (size_t i = 0; i < seq_value->size(); ++i) {
      auto [value, offset_inner] = ConstructInputValue((*seq_value)[i], input_abstract, input_index + offset);
      MS_EXCEPTION_IF_NULL(value);
      (void)values.emplace_back(value);
      offset += offset_inner;
    }
    (void)std::for_each(begin_iter, begin_iter + offset,
                        [](const auto &abs) -> void { MS_LOG(DEBUG) << "The convert abs is :" << abs->ToString(); });
    return std::make_pair(std::make_shared<ValueTuple>(values), offset);
  }

  const auto num_value = GetValue<int64_t>(value);

  constexpr auto kNotDynamicFlag = -1;
  if (num_value == kNotDynamicFlag) {
    return std::make_pair(GetValueByAbstract(*begin_iter), 1);
  } else {
    MS_LOG(EXCEPTION) << "The attr of structural must all value -1 but got " << num_value;
  }
}

ValuePtr PyExecuteInitializer::ConstructInputValues(const PrimitivePtr &prim,
                                                    const std::vector<abstract::AbstractBase *> &input_abstract) {
  MS_EXCEPTION_IF_NULL(prim);
  auto input_structural = prim->GetAttr(kAttrTupleInputStructural);
  if (input_structural == nullptr) {
    MS_LOG(EXCEPTION) << "Invalid primitive:" << prim->ToString();
  }
  auto tuple_structural_value = input_structural->cast_ptr<ValueSequence>();
  MS_EXCEPTION_IF_NULL(tuple_structural_value);

  std::vector<ValuePtr> values;
  size_t input_index = 0;

  for (size_t i = 0; i < tuple_structural_value->size(); ++i) {
    auto item = (*tuple_structural_value)[i];
    MS_EXCEPTION_IF_NULL(item);
    if (input_abstract.size() <= input_index) {
      // The Ori  Node : Oper(a, b, ())  ==> Oper(a, b)  with structural --> (-1, -1 , ())
      // The abstract size will be smaller than the attr of tuple input structural.
      (void)values.emplace_back(ConstructEmptyTupleValue(item));
    }
    auto [value, offset] = ConstructInputValue(item, input_abstract, input_index);
    input_index += offset;
    (void)values.emplace_back(value);
    MS_LOG(DEBUG) << "Rectify abs :" << item->ToString() << ", from structural " << value->ToString();
  }

  return std::make_shared<ValueTuple>(values);
}

abstract::AbstractBasePtr PyExecuteInitializer::GenerateAbstract(const py::object &output) {
  if (tensor::IsTensorPy(output)) {
    const auto &tensor = tensor::ConvertToTensor(output);
    MS_EXCEPTION_IF_NULL(tensor);
    const auto &infer_shape = std::make_shared<abstract::Shape>(tensor->shape());
    return tensor->ToAbstract();
  } else if (py::isinstance<py::bool_>(output)) {
    return tensor::from_scalar(py::cast<bool>(output))->ToAbstract();
  } else if (py::isinstance<py::int_>(output)) {
    return tensor::from_scalar(py::cast<int64_t>(output))->ToAbstract();
  } else if (py::isinstance<py::float_>(output)) {
    return tensor::from_scalar(py::cast<float>(output))->ToAbstract();
  } else if (py::isinstance<py::list>(output) || py::isinstance<py::tuple>(output)) {
    ValuePtr converted_res = nullptr;
    if (parse::ConvertData(output, &converted_res)) {
      auto ret_list = converted_res->ToAbstract();
      return fallback::GenerateAbstractSequence(ret_list->BuildShape(), ret_list->BuildType(), false);
    }
  }
  return nullptr;
}

abstract::AbstractBasePtr PyExecuteInitializer::PyExecuteInferPy(const PrimitivePtr &primitive,
                                                                 const ValuePtr &input_value) {
  MS_EXCEPTION_IF_NULL(input_value);
  if (!input_value->isa<ValueSequence>()) {
    MS_LOG(EXCEPTION) << "Invalid pyexecute input value:" << input_value->ToString();
  }
  const auto &tuple_values = input_value->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(tuple_values);
  const auto &inputs = tuple_values->value();
  constexpr auto number_two = 2;
  if (inputs.size() <= number_two) {
    MS_LOG(EXCEPTION) << "Invalid pyexecute input value:" << input_value->ToString();
  }

  if (!inputs[0]->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << "Invalid script value:" << inputs[0]->ToString();
  }
  const auto &script = dyn_cast<StringImm>(inputs[0]);
  MS_EXCEPTION_IF_NULL(script);

  if (!inputs[1]->isa<ValueSequence>()) {
    MS_LOG(WARNING) << "The keys is not tuple value, but got " << inputs[1]->ToString();
    return abstract::MakeAbstract(std::make_shared<abstract::Shape>(ShapeVector({1})), kFloat64);
  }
  const auto &keys = dyn_cast<ValueSequence>(inputs[1]);
  MS_EXCEPTION_IF_NULL(keys);

  if (!inputs[number_two]->isa<ValueSequence>()) {
    MS_LOG(DEBUG) << "The values is not tuple value, but got " << inputs[number_two]->ToString();
    return abstract::MakeAbstract(std::make_shared<abstract::Shape>(ShapeVector({1})), kFloat64);
  }
  const auto &values = dyn_cast<ValueSequence>(inputs[number_two]);
  MS_EXCEPTION_IF_NULL(values);

  MS_LOG(DEBUG) << "The script is: " << script->ToString() << ", keys: " << keys->ToString()
                << ", values: " << values->ToString();
  if (keys->size() != values->size()) {
    MS_LOG(EXCEPTION) << "The length of keys(" << keys->size() << ") is not equal of the length of values("
                      << values->size() << ").";
  }

  py::gil_scoped_acquire gil_acquire;
  py::dict local_dict;
  for (size_t i = 0; i < keys->size(); ++i) {
    const auto &key = (*keys)[i];
    const auto &key_str = dyn_cast<StringImm>(key);
    MS_EXCEPTION_IF_NULL(key_str);

    const auto &value = (*values)[i];
    MS_EXCEPTION_IF_NULL(value);
    auto obj = ValueToPyData(value);
    local_dict[py::str(key_str->value())] = obj;
  }

  const auto &py_script = py::str(script->value());
  auto params = py::tuple(number_two);
  params[0] = py::dict();
  params[1] = local_dict;
  MS_LOG(DEBUG) << "Python script: " << py_script << ", local_dict: " << local_dict;
  try {
    mindspore::ScopedFallbackRunning fallback_running;
    const auto &output = parse::data_converter::CallPythonScript(py_script, params);
    MS_LOG(DEBUG) << "Python output type: " << py::str(output.get_type()) << ", output: " << output;
    primitive->set_attr(kAttrPyExecuteOutput, std::make_shared<parse::PyObjectWrapper>(output, "graph python obj"));
    const auto &abstract = GenerateAbstract(output);
    if (abstract != nullptr) {
      return abstract;
    }
  } catch (const py::error_already_set &e) {
    auto error_type_name = py::cast<std::string>(python_adapter::GetPyObjAttr(e.type(), "__name__"));
    auto error_iter = exception_types_map.find(error_type_name);
    if (error_iter != exception_types_map.end()) {
      auto &handler = LogWriter::GetExceptionHandler();
      if (handler != nullptr) {
        handler(error_iter->second, py::str(e.value()));
      }
    }
    throw std::runtime_error(py::str(e.value()));
  }

  const auto &infer_shape = std::make_shared<abstract::Shape>(ShapeVector({1}));
  return abstract::MakeAbstract(infer_shape, kFloat64);
}

abstract::AbstractBasePtr PyExecuteInitializer::CppInferShapeAndTypePy(
  const PrimitivePtr &primitive, const std::vector<abstract::AbstractBase *> &args_abs_list) {
  // We can't catch the pybind11 exception by py::builtin_exception or its base class,
  // so we have to list all pybind11 exceptions and catch one by one here.
  AbstractBasePtr res;
  std::function<void(void)> already_set_error_handler;
  std::function<void(void)> other_error_handler;
  std::function<void(void)> default_error_handler;
  HandleExceptionRethrow(
    [&res, &primitive, &args_abs_list]() {
      res = PyExecuteInferPy(primitive, ConstructInputValues(primitive, args_abs_list));
      MS_LOG(DEBUG) << "The abstract:" << res;
      return res;
    },
    already_set_error_handler, other_error_handler, default_error_handler);
  return res;
}
}  // namespace mindspore
