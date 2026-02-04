/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "include/backend/common/kernel_graph/py_execute_utils.h"

#include "include/utils/anfalgo.h"
#include "include/utils/pipeline/fallback.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "runtime/hardware_abstract/utils.h"
#include "utils/convert_utils.h"
#include "include/utils/convert_utils_py.h"
#include "include/utils/tensor_py.h"

namespace mindspore::pyexecute {
PyDataConverter py_data_convert_handler{nullptr};

void set_pydata_converter(const PyDataConverter &pydata_converter) { py_data_convert_handler = pydata_converter; }

namespace {

void TensorToRawMemory(const tensor::TensorPtr &tensor, KernelTensor *const kernel_tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  const auto &device_address = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_address);
  MS_LOG(DEBUG) << "tensor:" << tensor->ToString();
  if (tensor->Size() != device_address->GetSize()) {
    MS_LOG(EXCEPTION) << "Invalid tensor size:" << tensor->Size() << " device tensor size:" << device_address->GetSize()
                      << " for device tensor:" << device_address;
  }
  if (tensor->Size() == 0) {
    return;
  }
  if (device_address->GetDeviceType() == device::DeviceType::kCPU) {
    MS_EXCEPTION_IF_NULL(device_address->GetMutablePtr());
    auto cpu_tensor = tensor->cpu();
    MS_EXCEPTION_IF_NULL(cpu_tensor->data_c());
    const auto &res = memcpy_s(reinterpret_cast<char *>(device_address->GetMutablePtr()), device_address->GetSize(),
                               cpu_tensor->data_c(), device_address->GetSize());
    if (res != EOK) {
      MS_LOG(EXCEPTION) << "memcpy failed. res: " << res << ", for tensor:" << tensor->ToString()
                        << " size:" << device_address->GetSize();
    }
  } else {
    MS_LOG(DEBUG) << "Tensor:" << tensor->ToString() << " shape:" << tensor->shape() << " type:" << tensor->data_type()
                  << " size:" << tensor->Size();
    device::DeviceContextKey host_key = {device_address->GetDeviceType(), device_address->device_id()};
    device::DeviceContext *host_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
    host_context->device_res_manager_->SyncAllStreams();
    MS_EXCEPTION_IF_NULL(tensor->device_address());
    SyncCopy(kernel_tensor, tensor.get(), device_address->stream_id());
  }
}

tensor::TensorPtr ScalarToValue(const py::object &obj) {
  ValuePtr value = nullptr;
  if (py::isinstance<py::bool_>(obj)) {
    value = MakeValue(py::cast<bool>(obj));
  } else if (py::isinstance<py::int_>(obj)) {
    value = MakeValue(py::cast<int64_t>(obj));
  } else if (py::isinstance<py::float_>(obj)) {
    value = MakeValue(py::cast<float>(obj));
  } else {
    MS_LOG(EXCEPTION) << "Invalid scalar py obj.";
  }
  if (value == nullptr || (!value->isa<Scalar>())) {
    MS_LOG(EXCEPTION) << "Invalid value for obj.";
  }
  return ScalarToTensor(value->cast<ScalarPtr>());
}

template <typename T>
bool CheckSequenceElementSame(const py::sequence &obj) {
  // Check from second element, the type of first element is determined by T.
  for (size_t i = 1; i < py::len(obj); ++i) {
    if (!py::isinstance<T>(obj[i])) {
      return false;
    }
  }
  return true;
}

bool CheckSequenceElementSameTensor(const py::sequence &obj) {
  // Check from second element, the type of first element is determined by T.
  for (size_t i = 1; i < py::len(obj); ++i) {
    if (!tensor::IsTensorPy(obj[i])) {
      return false;
    }
  }
  return true;
}

bool CheckSequenceToMemory(const py::sequence &obj) {
  // A sequence object can be passed to raw memory and used by other operator if:
  //   1. The length of sequence is not empty.
  //   2. The sequence is not nested.
  //   3. The sequence only contains Scalar or Tensor elements.
  //   4. All the elements in sequence should be the same.
  if (py::len(obj) == 0) {
    return false;
  }
  auto first_obj = obj[0];
  if (py::isinstance<py::bool_>(first_obj)) {
    return CheckSequenceElementSame<py::bool_>(obj);
  } else if (py::isinstance<py::int_>(first_obj)) {
    return CheckSequenceElementSame<py::int_>(obj);
  } else if (py::isinstance<py::float_>(first_obj)) {
    return CheckSequenceElementSame<py::float_>(obj);
  } else if (tensor::IsTensorPy(first_obj)) {
    return CheckSequenceElementSameTensor(obj);
  }
  return false;
}

tensor::TensorPtr SequenceToValue(const py::sequence &obj) {
  if (!CheckSequenceToMemory(obj)) {
    MS_LOG(EXCEPTION) << "Invalid py object.";
  }

  size_t obj_len = py::len(obj);
  std::vector<ValuePtr> values;
  for (size_t i = 0; i < obj_len; ++i) {
    auto element_obj = obj[i];
    if (tensor::IsTensorPy(element_obj)) {
      values.emplace_back(tensor::ConvertToTensor(element_obj));
    } else {
      values.emplace_back(ScalarToValue(element_obj));
    }
  }
  return common::AnfAlgo::SequenceToTensor(std::make_shared<ValueTuple>(values));
}

bool IsValidObj(const py::object &obj) {
  py::gil_scoped_acquire gil_acquire;
  return tensor::IsTensorPy(obj) ||
         ((py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) &&
          CheckSequenceToMemory(py::sequence(obj))) ||
         py::isinstance<py::bool_>(obj) || py::isinstance<py::int_>(obj) || py::isinstance<py::float_>(obj);
}

TypeId GetTypeIdByAbstract(const AbstractBasePtr &abstract) {
  MS_EXCEPTION_IF_NULL(abstract);
  if (abstract->isa<abstract::AbstractScalar>()) {
    const auto &type = abstract->BuildType();
    MS_EXCEPTION_IF_NULL(type);
    return type->type_id();
  } else if (abstract->isa<abstract::AbstractTensor>()) {
    const auto &tensor_abstract = abstract->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor_abstract);
    MS_EXCEPTION_IF_NULL(tensor_abstract->element());
    const auto &type = tensor_abstract->element()->BuildType();
    MS_EXCEPTION_IF_NULL(type);
    return type->type_id();
  } else {
    MS_LOG(EXCEPTION) << "Invalid abstract:" << abstract->ToString();
  }
}

bool IsValidAbstract(const abstract::AbstractBasePtr &abstract) {
  MS_EXCEPTION_IF_NULL(abstract);
  if (abstract->isa<abstract::AbstractScalar>() || abstract->isa<abstract::AbstractTensor>()) {
    return true;
  }
  if (!abstract->isa<abstract::AbstractSequence>()) {
    return false;
  }
  const auto &seq_abstract = abstract->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(seq_abstract);
  const auto &sub_abstracts = seq_abstract->elements();
  if (sub_abstracts.size() <= 1) {
    return true;
  }
  if (sub_abstracts[0] == nullptr ||
      ((!sub_abstracts[0]->isa<abstract::AbstractScalar>()) && (!sub_abstracts[0]->isa<abstract::AbstractTensor>()))) {
    return false;
  }

  auto get_shape_vector_by_abstract = [](const AbstractBasePtr &abstract) -> ShapeVector {
    MS_EXCEPTION_IF_NULL(abstract);
    if (abstract->isa<abstract::AbstractScalar>()) {
      return {};
    } else if (abstract->isa<abstract::AbstractTensor>()) {
      const auto &base_shape = abstract->BuildShape();
      MS_EXCEPTION_IF_NULL(base_shape);
      if (!base_shape->isa<abstract::Shape>()) {
        MS_LOG(EXCEPTION) << "Invalid shape:" << base_shape->ToString() << " in abstract:" << abstract->ToString();
      }
      const auto &shape = base_shape->cast<abstract::ShapePtr>();
      MS_EXCEPTION_IF_NULL(shape);
      return shape->shape();
    } else {
      MS_LOG(EXCEPTION) << "Invalid abstract:" << abstract->ToString();
    }
  };

  const auto &base_type_id = GetTypeIdByAbstract(sub_abstracts[0]);
  const auto &base_shape_vector = get_shape_vector_by_abstract(sub_abstracts[0]);
  for (size_t i = 1; i < sub_abstracts.size(); ++i) {
    MS_EXCEPTION_IF_NULL(sub_abstracts[i]);
    if (sub_abstracts[i] == nullptr ||
        ((!sub_abstracts[i]->isa<abstract::AbstractScalar>()) &&
         (!sub_abstracts[i]->isa<abstract::AbstractTensor>())) ||
        base_type_id != GetTypeIdByAbstract(sub_abstracts[i]) ||
        base_shape_vector != get_shape_vector_by_abstract(sub_abstracts[i])) {
      return false;
    }
  }
  return true;
}

size_t GetSizeForAbstract(const abstract::AbstractBasePtr &abstract) {
  MS_EXCEPTION_IF_NULL(abstract);
  if (abstract->isa<abstract::AbstractScalar>()) {
    return GetTypeByte(abstract->BuildType());
  } else if (abstract->isa<abstract::AbstractTensor>()) {
    const auto &tensor_abstract = abstract->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor_abstract);
    const auto &base_shape = tensor_abstract->BuildShape();
    MS_EXCEPTION_IF_NULL(base_shape);
    const auto &shape = base_shape->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape);
    const auto &shape_vector = shape->shape();
    MS_EXCEPTION_IF_NULL(tensor_abstract->element());
    const auto &type = tensor_abstract->element()->BuildType();
    return std::accumulate(shape_vector.begin(), shape_vector.end(), GetTypeByte(type), std::multiplies<size_t>());
  }

  const auto &seq_abstract = abstract->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(seq_abstract);
  const auto &sub_abstracts = seq_abstract->elements();
  if (sub_abstracts.empty()) {
    return 0;
  }
  return sub_abstracts.size() * GetSizeForAbstract(sub_abstracts[0]);
}

ValuePtr ConvertPyObjectToValue(const py::object &obj) {
  if (tensor::IsTensorPy(obj)) {
    return tensor::ConvertToTensor(obj);
  } else if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
    if (!CheckSequenceToMemory(py::sequence(obj))) {
      return nullptr;
    }
    size_t obj_len = py::len(obj);
    std::vector<ValuePtr> values;
    for (size_t i = 0; i < obj_len; ++i) {
      auto sub_value = ConvertPyObjectToValue((py::sequence(obj))[i]);
      if (sub_value == nullptr) {
        return nullptr;
      }
      values.emplace_back(sub_value);
    }
    return std::make_shared<ValueTuple>(values);
  } else if (py::isinstance<py::bool_>(obj)) {
    return MakeValue(py::cast<bool>(obj));
  } else if (py::isinstance<py::int_>(obj)) {
    return MakeValue(py::cast<int64_t>(obj));
  } else if (py::isinstance<py::float_>(obj)) {
    return MakeValue(py::cast<float>(obj));
  }
  return nullptr;
}
}  // namespace

tensor::TensorPtr GetValueByPyObj(const py::object &obj) {
  py::gil_scoped_acquire gil_acquire;
  if (tensor::IsTensorPy(obj)) {
    return tensor::ConvertToTensor(obj);
  } else if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
    return SequenceToValue(py::sequence(obj));
  } else if (py::isinstance<py::bool_>(obj) || py::isinstance<py::int_>(obj) || py::isinstance<py::float_>(obj)) {
    return ScalarToValue(obj);
  }
  MS_LOG(EXCEPTION) << "Invalid object:" << obj;
}

abstract::AbstractBasePtr GenerateAbstractFromPyObject(const py::object &obj) {
  // This function will be moved to runtime compile pass later.
  py::gil_scoped_acquire gil_acquire;
  if (tensor::IsTensorPy(obj)) {
    const auto &tensor = tensor::ConvertToTensor(obj);
    MS_EXCEPTION_IF_NULL(tensor);
    MS_LOG(DEBUG) << "tensor:" << tensor->ToString();
    return tensor->ToAbstract();
  }

  if (py::isinstance<py::bool_>(obj)) {
    return MakeValue(py::cast<bool>(obj))->ToAbstract();
  } else if (py::isinstance<py::int_>(obj)) {
    return MakeValue(py::cast<int64_t>(obj))->ToAbstract();
  } else if (py::isinstance<py::float_>(obj)) {
    return MakeValue(py::cast<float>(obj))->ToAbstract();
  }

  // obj is tuple will add later.
  if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
    ValuePtr converted_res = nullptr;
    MS_EXCEPTION_IF_NULL(py_data_convert_handler);
    if (py_data_convert_handler(obj, &converted_res)) {
      auto ret_list = converted_res->ToAbstract();
      return fallback::GenerateAbstractSequence(ret_list->BuildShape(), ret_list->BuildType(), false);
    }
  }
  ShapeVector shape = {1};
  return std::make_shared<abstract::AbstractTensor>(TypeIdToType(TypeId::kNumberTypeFloat64), shape);
}

void UserDataToRawMemory(KernelTensor *const kernel_tensor) {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_address = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_address);
  MS_EXCEPTION_IF_NULL(kernel_tensor->user_data());
  MS_LOG(DEBUG) << "Start sync data from device address:" << device_address
                << " user data:" << kernel_tensor->user_data();
  const auto &user_data_obj =
    kernel_tensor->user_data()->get<kernel::PyExecuteOutputUserData>(kernel::PyExecuteOutputUserData::key);
  MS_EXCEPTION_IF_NULL(user_data_obj);
  const auto &obj = user_data_obj->obj;
  if (!IsValidObj(obj)) {
    return;
  }
  const auto abstract = GenerateAbstractFromPyObject(obj);
  MS_EXCEPTION_IF_NULL(abstract);
  if (!IsValidAbstract(abstract)) {
    MS_LOG(DEBUG) << "Invalid abstract:" << abstract->ToString();
    return;
  }
  device_address->SetSize(GetSizeForAbstract(abstract));

  MS_LOG(DEBUG) << "Infer abstract:" << abstract->ToString() << " size:" << device_address->GetSize()
                << " device name:" << device::GetDeviceNameByType(device_address->GetDeviceType())
                << " device id:" << device_address->device_id();

  const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device_address->GetDeviceType(), device_address->device_id()});
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  if (device_address->GetPtr() != nullptr) {
    device_context->device_res_manager_->FreeMemory(device_address.get());
  }
  device_address->set_ptr(nullptr);
  if (!device_context->device_res_manager_->AllocateMemory(device_address.get())) {
    MS_LOG(ERROR) << "Device(id:" << std::to_string(device_context->device_context_key().device_id_)
                  << ") memory isn't enough and alloc failed, alloc size: " + std::to_string(device_address->GetSize());
    return;
  } else {
    static std::string name = "Alloc memory";
    kernel_tensor->IncreaseNewRefCount(name);
  }
  kernel_tensor->DecreaseNewRefCount("Pyexecute sync");
  tensor::TensorPtr tensor = GetValueByPyObj(obj);
  TensorToRawMemory(tensor, kernel_tensor);
}

ValuePtr GetValueFromUserData(const UserDataPtr &user_data) {
  if (user_data == nullptr) {
    return nullptr;
  }
  if (!user_data->has(kernel::PyExecuteOutputUserData::key)) {
    return nullptr;
  }
  auto py_obj = user_data->get<kernel::PyExecuteOutputUserData>(kernel::PyExecuteOutputUserData::key);
  if (py_obj == nullptr) {
    return nullptr;
  }
  py::gil_scoped_acquire gil_acquire;
  auto value = ConvertPyObjectToValue(py_obj->obj);
  MS_LOG(DEBUG) << "Set value:" << (value == nullptr ? "nullptr" : value->ToString());
  return value;
}
}  // namespace mindspore::pyexecute
