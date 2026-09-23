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

#include "pybind_api/tensor_buffer.h"
#include <memory>
#include <vector>
#include <Python.h>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "include/utils/exception.h"
#include "include/utils/tensor_py.h"
#include "device_address/device_address.h"
#include "device_address/device_type.h"
#include "ir/tensor.h"
#include "pybind_api/pynative/frombuffer.h"
#include "include/runtime/pipeline/pipeline.h"
#include "utils/shape_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace tensor {

namespace py = pybind11;

void SaveTensorToBuf(const py::object &tensor_obj, const py::object &buf, const py::object &dtype,
                     const std::vector<int64_t> &shape, int64_t count, int64_t offset) {
  // pybind11 enters this function with the Python GIL held. Keep it held while converting Python objects, creating
  // the destination buffer view, and copying bytes so concurrent Python save/load calls cannot race in these critical
  // sections. WaitForward below is the only intentional release point and reacquires the GIL before it returns.

  // Validate the Python object before extracting its C++ Tensor. A raw reinterpret_cast cannot distinguish a Tensor
  // from another Python object and would make a non-Tensor input unsafe to dereference.
  if (!IsTensorPy(tensor_obj)) {
    MS_EXCEPTION(TypeError) << "tensor must be a MindSpore Tensor.";
  }
  TensorPtr src_tensor = ConvertToTensor(tensor_obj);
  MS_EXCEPTION_IF_NULL(src_tensor);

  // Materialize data produced by asynchronous PyNative work before reading the source address. WaitForward uses
  // GilReleaseWithCheck while it blocks, allowing other Python threads to make progress, and reacquires the GIL before
  // the following Python buffer operations begin.
  runtime::Pipeline::Get().WaitForward();

  // HostCopy below can only read CPU-compatible storage. kUnknown is accepted for host tensors whose device type has
  // not been tagged explicitly yet.
  const auto &src_addr = src_tensor->device_address();
  if (src_addr != nullptr) {
    auto device_type = src_addr->GetDeviceType();
    if (device_type != device::DeviceType::kCPU && device_type != device::DeviceType::kUnknown) {
      MS_LOG(EXCEPTION) << "save_tensor_to_buf only supports CPU tensors, but got "
                        << device::GetDeviceNameByType(device_type) << " tensor.";
    }
  }

  // Reuse TensorFrombuffer to validate dtype/count/offset, retain the Python buffer owner, and create a zero-copy
  // destination view. The shape argument is checkpoint metadata kept for API symmetry; the save path does not reshape
  // or validate against it, so the caller must keep shape/count/dtype consistent with the source Tensor.
  py::object dst_tensor_obj = TensorFrombuffer(buf, dtype, count, offset);

  if (!IsTensorPy(dst_tensor_obj)) {
    MS_LOG(INTERNAL_EXCEPTION) << "failed to create tensor view on buffer.";
  }
  TensorPtr dst_tensor = ConvertToTensor(dst_tensor_obj);
  MS_EXCEPTION_IF_NULL(dst_tensor);

  // Copy source bytes into the buffer-backed Tensor. HostCopy handles same-type memcpy and supported host-side dtype
  // conversions; checkpoint callers normally use an identical source and destination dtype for byte-exact storage.
  const auto &dst_addr = dst_tensor->device_address();
  if (!HostCopy(dst_addr, src_addr)) {
    MS_LOG(EXCEPTION) << "failed to copy tensor data to buffer.";
  }
}

py::object LoadTensorFromBuf(const py::object &buf, const py::object &dtype, const std::vector<int64_t> &shape,
                             int64_t count, int64_t offset) {
  // pybind11 keeps the Python GIL for this entire function. This deliberately serializes buffer inspection, ownership
  // updates, and Tensor construction when multiple Python threads call save/load concurrently.

  // Create a zero-copy 1-D view. TensorFrombuffer validates dtype/count/offset and retains the Python buffer object for
  // as long as the resulting device address is alive.
  py::object flat_tensor = TensorFrombuffer(buf, dtype, count, offset);

  // Extract the C++ Tensor without copying the buffer-backed data.
  if (!IsTensorPy(flat_tensor)) {
    MS_LOG(INTERNAL_EXCEPTION) << "failed to create tensor from buffer.";
  }
  TensorPtr tensor_ptr = ConvertToTensor(flat_tensor);
  MS_EXCEPTION_IF_NULL(tensor_ptr);

  // Apply the checkpoint shape as metadata while reusing the same device address. The flat view exposes exactly the
  // elements the buffer holds, so a shape describing a different number of elements would make the returned Tensor
  // read and write past the end of the buffer.
  auto device_addr = tensor_ptr->device_address();
  const auto buffer_elements = SizeOf(tensor_ptr->shape());
  const auto shape_elements = SizeOf(shape);
  if (shape_elements != buffer_elements) {
    MS_EXCEPTION(ValueError) << "The buffer holds " << buffer_elements << " elements, but the shape "
                             << ShapeVectorToString(shape) << " holds " << shape_elements
                             << " elements. The shape must match the number of elements in the buffer.";
  }
  auto reshaped_tensor = std::make_shared<Tensor>(tensor_ptr->data_type(), ShapeVector(shape), device_addr);
  // Match other Python-visible tensors: downstream data access performs the normal pipeline synchronization before
  // observing the shared buffer.
  reshaped_tensor->set_need_pipeline_sync(true);

  return PackTensorToPyObject(reshaped_tensor);
}

}  // namespace tensor
}  // namespace mindspore
