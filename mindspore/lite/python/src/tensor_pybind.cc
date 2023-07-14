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
#include <vector>
#include <string>
#include <set>
#include "include/api/types.h"
#include "include/api/data_type.h"
#include "include/api/format.h"
#include "src/common/log_adapter.h"
#include "third_party/securec/include/securec.h"
#include "mindspore/lite/src/common/mutable_tensor_impl.h"
#include "mindspore/core/ir/api_tensor_impl.h"
#include "src/tensor.h"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "pybind11/stl.h"

namespace mindspore::lite {
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
    .def("set_data_from_numpy",
         [](const MSTensorPtr &tensor, const py::array &input) { return SetTensorNumpyData(tensor, input); })
    .def("get_data_to_numpy", [](const MSTensorPtr &tensor) -> py::array {
      if (tensor == nullptr) {
        MS_LOG(ERROR) << "Tensor object cannot be nullptr";
        return py::array();
      }
      auto info = GetPyBufferInfo(tensor);
      py::object self = py::cast(tensor->impl());
      return py::array(py::dtype(info), info.shape, info.strides, info.ptr, self);
    });
}

MSTensorPtr create_tensor(DataType data_type, const std::vector<int64_t> &shape) {
  auto tensor = mindspore::MSTensor::CreateTensor("", data_type, shape, nullptr, 0);
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "create tensor failed.";
    return {};
  }
  mindspore::Format data_format = NCHW;
  tensor->SetFormat(data_format);
  return MSTensorPtr(tensor);
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

bool IsCContiguous(const py::array &input) {
  auto flags = static_cast<unsigned int>(input.flags());
  return (flags & pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_) != 0;
}

enum DataType GetDataType(const py::buffer_info &buf) {
  std::set<char> fp_format = {'e', 'f', 'd'};
  std::set<char> int_format = {'b', 'h', 'i', 'l', 'q'};
  std::set<char> uint_format = {'B', 'H', 'I', 'L', 'Q'};
  if (buf.format.size() == 1) {
    char format = buf.format.front();
    if (fp_format.find(format) != fp_format.end()) {
      switch (buf.itemsize) {
        case 2:
          return DataType::kNumberTypeFloat16;
        case 4:
          return DataType::kNumberTypeFloat32;
        case 8:
          return DataType::kNumberTypeFloat64;
      }
    } else if (int_format.find(format) != int_format.end()) {
      switch (buf.itemsize) {
        case 1:
          return DataType::kNumberTypeInt8;
        case 2:
          return DataType::kNumberTypeInt16;
        case 4:
          return DataType::kNumberTypeInt32;
        case 8:
          return DataType::kNumberTypeInt64;
      }
    } else if (uint_format.find(format) != uint_format.end()) {
      switch (buf.itemsize) {
        case 1:
          return DataType::kNumberTypeUInt8;
        case 2:
          return DataType::kNumberTypeUInt16;
        case 4:
          return DataType::kNumberTypeUInt32;
        case 8:
          return DataType::kNumberTypeUInt64;
      }
    } else if (format == '?') {
      return DataType::kNumberTypeBool;
    }
  }
  MS_LOG(WARNING) << "Unsupported DataType format " << buf.format << " item size " << buf.itemsize;
  return DataType::kTypeUnknown;
}

class PyBindAllocator : public Allocator {
 public:
  explicit PyBindAllocator(py::buffer_info &&py_buffer_info) : buffer_(std::move(py_buffer_info)) {}
  ~PyBindAllocator() override{};
  void *Malloc(size_t size) override { return nullptr; };
  void Free(void *ptr) override {
    if (buffer_.ptr != nullptr) {
      py::gil_scoped_acquire acquire;
      buffer_ = py::buffer_info();
    }
  }
  int RefCount(void *ptr) override { return std::atomic_load(&ref_count_); }
  int SetRefCount(void *ptr, int ref_count) override {
    std::atomic_store(&ref_count_, ref_count);
    return ref_count;
  }

  int DecRefCount(void *ptr, int ref_count) override {
    if (ptr == nullptr) {
      return 0;
    }
    auto ref = std::atomic_fetch_sub(&ref_count_, ref_count);
    if ((ref - ref_count) <= 0) {
      if (buffer_.ptr != nullptr) {
        py::gil_scoped_acquire acquire;
        buffer_ = py::buffer_info();
      }
    }
    return (ref - ref_count);
  }

  int IncRefCount(void *ptr, int ref_count) override {
    auto ref = std::atomic_fetch_add(&ref_count_, ref_count);
    return (ref + ref_count);
  }

 private:
  std::atomic_int ref_count_ = {0};
  py::buffer_info buffer_;
};

bool SetTensorNumpyData(const MSTensorPtr &tensor_ptr, const py::array &input) {
  if (tensor_ptr == nullptr) {
    MS_LOG(ERROR) << "tensor_ptr is nullptr.";
    return false;
  }
  auto &tensor = *tensor_ptr;
  // Check format.
  if (!IsCContiguous(input)) {
    MS_LOG(ERROR) << "Numpy array is not C Contiguous";
    return false;
  }

  auto py_buffer_info = input.request();
  auto py_data_type = GetDataType(py_buffer_info);
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

  std::shared_ptr<PyBindAllocator> py_allocator = std::make_shared<PyBindAllocator>(std::move(py_buffer_info));
  if (py_allocator == nullptr) {
    MS_LOG(ERROR) << "malloc GeAllocator failed.";
    return false;
  }
  if (tensor.Data() != nullptr) {
    auto new_tensor_ptr = MSTensor::CreateTensor(tensor.Name(), tensor.DataType(), tensor.Shape(), nullptr, 0);
    if (new_tensor_ptr == nullptr) {
      MS_LOG(ERROR) << "new tensor failed.";
      return false;
    }
    tensor = *new_tensor_ptr;
  }
  tensor.SetData(py_buffer_info.ptr, true);
  tensor.SetAllocator(py_allocator);
  return true;
}

py::buffer_info GetPyBufferInfo(const MSTensorPtr &tensor) {
  ssize_t item_size = tensor->DataSize() / tensor->ElementNum();
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
  return py::buffer_info{tensor->MutableData(), item_size, format, ndim, shape, strides};
}
}  // namespace mindspore::lite
