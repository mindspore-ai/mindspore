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

#include "pybind_api/pynative/frombuffer.h"
#include <memory>
#include <vector>
#include <Python.h>
#include "pybind11/pybind11.h"
#include "include/utils/exception.h"
#include "include/utils/tensor_py.h"
#include "device_address/device_address_maker.h"
#include "ir/dtype.h"
#include "ir/tensor.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace tensor {

namespace py = pybind11;

void ValidateBufferInput(Py_ssize_t buf_len, int64_t &count, int64_t offset, size_t elsize) {
  if (count < -1) {
    MS_LOG(EXCEPTION) << "count (" << count << ") must be -1 or positive, but got negative value other than -1";
  }
  if (!(buf_len > 0 && count != 0)) {
    MS_LOG(EXCEPTION) << "both buffer length (" << buf_len << ") and count (" << count << ") must not be 0";
  }
  if (!(offset >= 0 && offset < buf_len)) {
    MS_LOG(EXCEPTION) << "offset (" << offset << " bytes) must be non-negative and no greater than "
                      << "buffer length (" << buf_len << " bytes) minus 1.";
  }
  size_t u_offset = static_cast<size_t>(offset);
  size_t u_buf_len = static_cast<size_t>(buf_len);
  size_t remaining_buffer_size = u_buf_len - u_offset;
  if (!(count > 0 || remaining_buffer_size % elsize == 0)) {
    MS_LOG(EXCEPTION) << "buffer length (" << buf_len - offset << " bytes) after offset (" << offset
                      << " bytes) must be a multiple of element size (" << elsize << ").";
  }
  if (count == -1) {
    count = static_cast<int64_t>(remaining_buffer_size / elsize);
  }
  size_t u_count = static_cast<size_t>(count);
  size_t required_size = u_count * elsize;
  if (remaining_buffer_size < required_size) {
    MS_LOG(EXCEPTION) << "buffer length (" << remaining_buffer_size << " bytes) after offset (" << offset
                      << " bytes) is not enough for the requested count of elements (" << count << " elements of size "
                      << elsize << " bytes each, total " << required_size << " bytes).";
  }
}

py::object TensorFrombuffer(const py::object &buffer, const py::object &dtype, int64_t count, int64_t offset) {
  mindspore::TypeId type_id = mindspore::kTypeUnknown;
  TypePtr type_ptr = nullptr;

  if (py::isinstance<mindspore::Type>(dtype)) {
    type_ptr = py::cast<mindspore::TypePtr>(dtype);
    type_id = type_ptr->type_id();
  } else {
    MS_LOG(EXCEPTION) << "dtype must be mindspore dtype object, but got" << py::str(dtype).cast<std::string>();
  }

  if (!PyObject_CheckBuffer(buffer.ptr())) {
    MS_LOG(EXCEPTION) << "object does not implement Python buffer protocol.";
  }

  Py_buffer py_buf;
  if (PyObject_GetBuffer(buffer.ptr(), &py_buf, PyBUF_WRITABLE) < 0) {
    PyErr_Clear();

    if (PyObject_GetBuffer(buffer.ptr(), &py_buf, PyBUF_SIMPLE) < 0) {
      MS_LOG(EXCEPTION) << "could not retrieve buffer from object.";
    }
    MS_LOG(WARNING) << "Buffer is read-only. Tensor operations that require "
                    << "writing may fail or have undefined behavior.";
  }
  auto buf_len = py_buf.len;
  auto buf_ptr = py_buf.buf;
  PyBuffer_Release(&py_buf);

  PyObject *python_buffer_obj = buffer.ptr();

  size_t elsize = GetTypeByte(type_ptr);
  ValidateBufferInput(buf_len, count, offset, elsize);

  // Increase reference count to keep the buffer object alive
  Py_INCREF(python_buffer_obj);
  void *data_ptr = static_cast<char *>(buf_ptr) + offset;
  auto device_address = DeviceAddressMaker(data_ptr, type_id, std::vector<int64_t>{count})
                          .set_deleter([python_buffer_obj](void *, bool) {
                            pybind11::gil_scoped_acquire gil;
                            // Decrease reference count to allow buffer object to be garbage collected
                            Py_DECREF(python_buffer_obj);
                          })
                          .set_maker(GetDeviceAddressMaker(device::DeviceType::kCPU))
                          .make_device_address();
  auto tensor = std::make_shared<tensor::Tensor>(type_id, std::vector<int64_t>{count}, device_address);

  return PackTensorToPyObject(tensor);
}
}  // namespace tensor
}  // namespace mindspore
