/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "include/api/types.h"
#include "include/api/data_type.h"
#include "include/api/format.h"
#include "src/common/log_adapter.h"
#include "src/litert/cxx_api/tensor_utils.h"
#include "third_party/securec/include/securec.h"
#include "mindspore/lite/src/common/mutable_tensor_impl.h"
#include "mindspore/lite/python/src/tensor_numpy_impl.h"
#include "mindspore/core/ir/api_tensor_impl.h"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "pybind11/stl.h"
#ifdef ENABLE_CLOUD_INFERENCE
#include "extendrt/kernel/ascend/plugin/ascend_allocator_plugin.h"
#endif
namespace mindspore::lite {
namespace {
bool IsCContiguous(const py::array &input) {
  auto flags = static_cast<unsigned int>(input.flags());
  return (flags & pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_) != 0;
}
}  // namespace

namespace py = pybind11;
using MSTensorPtr = std::shared_ptr<MSTensor>;

py::buffer_info GetPyBufferInfo(const MSTensorPtr &tensor);
bool SetTensorNumpyData(const MSTensorPtr &tensor, const py::array &input);

void TensorPyBind(const py::module &m) {
  (void)py::enum_<DataType>(m, "DataType")
    .value("kTypeUnknown", DataType::kTypeUnknown)
    .value("kObjectTypeString", DataType::kObjectTypeString)
    .value("kObjectTypeList", DataType::kObjectTypeList)
    .value("kObjectTypeTuple", DataType::kObjectTypeTuple)
    .value("kObjectTypeTensorType", DataType::kObjectTypeTensorType)
    .value("kNumberTypeBool", DataType::kNumberTypeBool)
    .value("kNumberTypeInt8", DataType::kNumberTypeInt8)
    .value("kNumberTypeInt16", DataType::kNumberTypeInt16)
    .value("kNumberTypeInt32", DataType::kNumberTypeInt32)
    .value("kNumberTypeInt64", DataType::kNumberTypeInt64)
    .value("kNumberTypeUInt8", DataType::kNumberTypeUInt8)
    .value("kNumberTypeUInt16", DataType::kNumberTypeUInt16)
    .value("kNumberTypeUInt32", DataType::kNumberTypeUInt32)
    .value("kNumberTypeUInt64", DataType::kNumberTypeUInt64)
    .value("kNumberTypeFloat16", DataType::kNumberTypeFloat16)
    .value("kNumberTypeFloat32", DataType::kNumberTypeFloat32)
    .value("kNumberTypeFloat64", DataType::kNumberTypeFloat64)
    .value("kInvalidType", DataType::kInvalidType);

  (void)py::enum_<Format>(m, "Format")
    .value("DEFAULT_FORMAT", Format::DEFAULT_FORMAT)
    .value("NCHW", Format::NCHW)
    .value("NHWC", Format::NHWC)
    .value("NHWC4", Format::NHWC4)
    .value("HWKC", Format::HWKC)
    .value("HWCK", Format::HWCK)
    .value("KCHW", Format::KCHW)
    .value("CKHW", Format::CKHW)
    .value("KHWC", Format::KHWC)
    .value("CHWK", Format::CHWK)
    .value("HW", Format::HW)
    .value("HW4", Format::HW4)
    .value("NC", Format::NC)
    .value("NC4", Format::NC4)
    .value("NC4HW4", Format::NC4HW4)
    .value("NCDHW", Format::NCDHW)
    .value("NWC", Format::NWC)
    .value("NCW", Format::NCW)
    .value("NDHWC", Format::NDHWC)
    .value("NC8HW8", Format::NC8HW8);

  (void)py::class_<MSTensor::Impl, std::shared_ptr<MSTensor::Impl>>(m, "TensorImpl_");
  (void)py::class_<MSTensor, std::shared_ptr<MSTensor>>(m, "TensorBind")
    .def(py::init<>())
    .def("set_tensor_name", [](MSTensor &tensor, const std::string &name) { tensor.SetTensorName(name); })
    .def("get_tensor_name", &MSTensor::Name)
    .def("set_data_type", &MSTensor::SetDataType)
    .def("get_data_type", &MSTensor::DataType)
    .def("set_shape", &MSTensor::SetShape)
    .def("get_shape", &MSTensor::Shape)
    .def("set_format", &MSTensor::SetFormat)
    .def("get_format", &MSTensor::format)
    .def("get_element_num", &MSTensor::ElementNum)
    .def("get_data_size", &MSTensor::DataSize)
    .def("set_data", &MSTensor::SetData)
    .def("get_data", &MSTensor::MutableData)
    .def("is_null", [](const MSTensorPtr &tensor) { return tensor == nullptr; })
    .def("get_tensor_device_type",
         [](const MSTensorPtr &tensor) {
           std::string device = "None";
           if (!tensor->GetDevice().empty()) {
             device = tensor->GetDevice();
           }
           return device + ":" + std::to_string(tensor->GetDeviceId());
         })
    .def("set_data_from_numpy",
         [](const MSTensorPtr &tensor, const py::array &input) { return SetTensorNumpyData(tensor, input); })
    .def("get_data_to_numpy", [](const MSTensorPtr &tensor) -> py::array {
      if (tensor == nullptr) {
        MS_LOG(ERROR) << "Tensor object cannot be nullptr";
        return py::array();
      }
      auto shape = tensor->Shape();
      if (std::any_of(shape.begin(), shape.end(), [](auto item) { return item <= 0; })) {
        MS_LOG(ERROR) << "Tensor shape " << shape << " is invalid";
        return py::array();
      }
      auto elem_num = tensor->ElementNum();
      if (elem_num <= 0) {
        MS_LOG(ERROR) << "Tensor element num " << elem_num << " cannot <= 0";
        return py::array();
      }
      auto info = GetPyBufferInfo(tensor);
      py::object self = py::cast(tensor->impl());
      return py::array(py::dtype(info), info.shape, info.strides, info.ptr, self);
    });
}

MSTensorPtr create_tensor(DataType data_type, const std::vector<int64_t> &shape, const std::string &device_type,
                          int device_id) {
  auto tensor = mindspore::MSTensor::CreateTensor("", data_type, shape, nullptr, 0, device_type, device_id);
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "create tensor failed.";
    return nullptr;
  }
  mindspore::Format data_format = NCHW;
  tensor->SetFormat(data_format);
  return MSTensorPtr(tensor);
}

MSTensorPtr create_tensor_by_tensor(const MSTensor &tensor, const std::string &device_type, int device_id) {
  auto new_tensor = mindspore::MSTensor::CreateTensor("", tensor, device_type, device_id);
  if (new_tensor == nullptr) {
    MS_LOG(ERROR) << "create tensor failed.";
    return nullptr;
  }
  new_tensor->SetFormat(tensor.format());
  return MSTensorPtr(new_tensor);
}

MSTensorPtr create_tensor_by_numpy(const py::array &input, const std::string &device_type, int32_t device_id) {
  // Check format.
  if (!IsCContiguous(input)) {
    MS_LOG(ERROR) << "Numpy array is not C Contiguous";
    return nullptr;
  }
  auto py_buffer_info = input.request();
  auto py_data_type = TensorNumpyImpl::GetDataType(py_buffer_info);
  auto py_data_size = py_buffer_info.size * py_buffer_info.itemsize;
  auto py_shape = py_buffer_info.shape;
  auto data_size = mindspore::CalTensorDataSize(py_shape, py_data_type);
  if (py_data_size != static_cast<int64_t>(data_size)) {
    MS_LOG(ERROR) << "Expect data size " << data_size << ", but got " << py_data_size;
    return nullptr;
  }
  auto tensor_impl = std::make_shared<TensorNumpyImpl>("", std::move(py_buffer_info), py_shape);
  tensor_impl->SetDevice(device_type);
  tensor_impl->SetDeviceId(device_id);
  auto numpy_tensor = std::make_shared<MSTensor>(tensor_impl);
  if (numpy_tensor == nullptr) {
    MS_LOG(ERROR) << "Create numpy tensor failed.";
    return nullptr;
  }
#ifdef ENABLE_CLOUD_INFERENCE
  if (device_type == "ascend") {
    kernel::AscendAllocatorPlugin::GetInstance().Register();
    device_id = device_id == -1 ? kernel::AscendAllocatorPlugin::GetInstance().GetCurrentDeviceId() : device_id;
    auto device_data = kernel::AscendAllocatorPlugin::GetInstance().Malloc(data_size, device_id);
    if (device_data == nullptr) {
      MS_LOG(ERROR) << "Malloc device data for numpy tensor failed.";
      return nullptr;
    }
    auto status = kernel::AscendAllocatorPlugin::GetInstance().CopyHostDataToDevice(numpy_tensor->MutableData(),
                                                                                    device_data, data_size);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "tensor has device data, then copy host data to device failed.";
      return nullptr;
    }
    numpy_tensor->SetDeviceData(device_data);
  }
#endif
  return numpy_tensor;
}

std::string GetPyTypeFormat(DataType data_type) {
  switch (data_type) {
    case DataType::kNumberTypeFloat32:
      return py::format_descriptor<float>::format();
    case DataType::kNumberTypeFloat64:
      return py::format_descriptor<double>::format();
    case DataType::kNumberTypeUInt8:
      return py::format_descriptor<uint8_t>::format();
    case DataType::kNumberTypeUInt16:
      return py::format_descriptor<uint16_t>::format();
    case DataType::kNumberTypeUInt32:
      return py::format_descriptor<uint32_t>::format();
    case DataType::kNumberTypeUInt64:
      return py::format_descriptor<uint64_t>::format();
    case DataType::kNumberTypeInt8:
      return py::format_descriptor<int8_t>::format();
    case DataType::kNumberTypeInt16:
      return py::format_descriptor<int16_t>::format();
    case DataType::kNumberTypeInt32:
      return py::format_descriptor<int32_t>::format();
    case DataType::kNumberTypeInt64:
      return py::format_descriptor<int64_t>::format();
    case DataType::kNumberTypeBool:
      return py::format_descriptor<bool>::format();
    case DataType::kObjectTypeString:
      return py::format_descriptor<uint8_t>::format();
    case DataType::kNumberTypeFloat16:
      return "e";
    default:
      MS_LOG(ERROR) << "Unsupported DataType " << static_cast<int>(data_type) << ".";
      return "";
  }
}

bool SetTensorNumpyData(const MSTensorPtr &tensor_ptr, const py::array &input) {
  auto &tensor = *tensor_ptr;
  // Check format.
  if (!IsCContiguous(input)) {
    MS_LOG(ERROR) << "Numpy array is not C Contiguous";
    return false;
  }
  auto py_buffer_info = input.request();
  auto py_data_type = TensorNumpyImpl::GetDataType(py_buffer_info);
  if (py_data_type != tensor.DataType()) {
    MS_LOG(ERROR) << "Expect data type " << static_cast<int>(tensor.DataType()) << ", but got "
                  << static_cast<int>(py_data_type);
    return false;
  }
  auto py_data_size = py_buffer_info.size * py_buffer_info.itemsize;
  if (py_data_size != static_cast<int64_t>(tensor.DataSize())) {
    MS_LOG(ERROR) << "Expect data size " << tensor.DataSize() << ", but got " << py_data_size << ", expected shape "
                  << tensor.Shape() << ", got shape " << py_buffer_info.shape;
    return false;
  }
#ifdef ENABLE_CLOUD_INFERENCE
  if (tensor.GetDeviceData() != nullptr) {
    MS_LOG(INFO) << "device tensor data ptr is not nullptr, need copy host data to device data.";
    auto status = kernel::AscendAllocatorPlugin::GetInstance().CopyHostDataToDevice(
      py_buffer_info.ptr, tensor.GetDeviceData(), tensor.DataSize());
    if (status != kSuccess) {
      MS_LOG(ERROR) << "tensor has device data, then copy host data to device failed.";
      return false;
    }
    return true;
  }
#endif
  auto tensor_impl = std::make_shared<TensorNumpyImpl>(tensor.Name(), std::move(py_buffer_info), tensor.Shape());
  MS_CHECK_TRUE_RET(tensor_impl != nullptr, false);
  tensor = MSTensor(tensor_impl);
  return true;
}

py::buffer_info GetPyBufferInfo(const MSTensorPtr &tensor) {
  ssize_t item_size = static_cast<ssize_t>(tensor->DataSize()) / tensor->ElementNum();
  std::string format = GetPyTypeFormat(tensor->DataType());
  auto lite_shape = tensor->Shape();
  ssize_t ndim = lite_shape.size();
  std::vector<ssize_t> shape(lite_shape.begin(), lite_shape.end());
  std::vector<ssize_t> strides(ndim);
  ssize_t element_num = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    strides[i] = element_num * item_size;
    element_num *= shape[i];
  }
#ifdef ENABLE_CLOUD_INFERENCE
  auto device_data = tensor->GetDeviceData();
  if (device_data != nullptr) {
    MS_LOG(INFO) << "need copy host data to device.";
    // device data is not nullptr, data in device, need copy device data to host.
    auto status = kernel::AscendAllocatorPlugin::GetInstance().CopyDeviceDataToHost(
      device_data, tensor->MutableData(), tensor->DataSize(), tensor->GetDeviceId());
    if (status != kSuccess) {
      MS_LOG(ERROR) << "tensor has device data, then copy device data to host failed.";
      return py::buffer_info{nullptr, 0, format, 0, {}, {}};
    }
  }
#endif
  return py::buffer_info{tensor->MutableData(), item_size, format, ndim, shape, strides};
}
}  // namespace mindspore::lite
