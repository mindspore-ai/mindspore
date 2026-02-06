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
#include "pybind_api/runtime/utils_py.h"
#include <string>
#include "include/runtime/pipeline/pipeline.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "utils/ms_context.h"

#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "include/utils/convert_utils_py.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/stream/multi_stream_controller.h"
#include "pybind_api/runtime/runtime_api.h"

namespace mindspore {
namespace hal {
namespace {
void Synchronize() {
  auto device_ctx = GetDeviceCtx();
  runtime::Pipeline::Get().WaitAll();
  auto &controller =
    device::DeviceContextManager::GetInstance().GetMultiStreamController(device_ctx->device_context_key().device_type_);
  controller->Refresh();
  (void)controller->SyncAllStreams();
}

std::vector<tensor::TensorPtr> ValuePtrListToTensorList(const ValuePtrList &value_list) {
  std::vector<tensor::TensorPtr> tensor_list;
  for (size_t i = 0; i < value_list.size(); ++i) {
    auto value = value_list[i];
    if (!value->isa<tensor::Tensor>()) {
      MS_EXCEPTION(TypeError) << "For func combine_tensor_list_contiguous, input only support list[Tensor].";
    }
    (void)tensor_list.emplace_back(value->cast<tensor::TensorPtr>());
  }
  return tensor_list;
}

py::object VectorSizeToPyData(const std::vector<size_t> &data) {
  py::list py_data;
  for (const auto &value : data) {
    py_data.append(value);
  }
  return py_data;
}
std::vector<size_t> PyListToVectorSize(const py::object &py_list) {
  std::vector<size_t> result;
  if (!py::isinstance<py::list>(py_list)) {
    MS_LOG(EXCEPTION) << "Input object should be list";
  }
  py::list list_obj = py_list.cast<py::list>();
  for (auto &item : list_obj) {
    size_t value = item.cast<size_t>();
    result.push_back(value);
  }
  return result;
}
}  // namespace

device::DeviceContext *GetDeviceCtx() {
  const auto &device_name = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto device_ctx = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device::GetDeviceTypeByName(device_name), MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_ctx);

  device_ctx->Initialize();
  return device_ctx;
}

py::object AllocDeviceMemoryForTensorList(const py::object &object, bool enable_mem_align) {
  Synchronize();
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "PyNative", "AllocDeviceMemoryForTensorList", "");
  auto device_ctx = GetDeviceCtx();
  ValuePtrList value_list;
  ConvertPyObjectToCTensor(object, &value_list);
  const auto &tensor_list = ValuePtrListToTensorList(value_list);
  auto size_list_pair = device_ctx->device_res_manager_->AllocDeviceMemoryForTensorList(tensor_list, enable_mem_align);
  auto before_size_list = size_list_pair.first;
  auto after_size_list = size_list_pair.second;
  py::list py_data;
  py_data.append(VectorSizeToPyData(before_size_list));
  py_data.append(VectorSizeToPyData(after_size_list));
  return py_data;
}

py::object GetSliceByTensorListIndexHandle(const py::object &object, const py::object &before_size_obj,
                                           const py::object &after_size_obj, size_t start, size_t end) {
  auto device_ctx = GetDeviceCtx();
  ValuePtrList value_list;
  ConvertPyObjectToCTensor(object, &value_list);
  const auto &tensor_list = ValuePtrListToTensorList(value_list);
  auto before_padding_size = PyListToVectorSize(before_size_obj);
  auto after_padding_size = PyListToVectorSize(after_size_obj);
  auto res_tensor = device_ctx->device_res_manager_->GetSliceByTensorListIndexHandle(tensor_list, before_padding_size,
                                                                                     after_padding_size, start, end);
  return ValueToPyData(res_tensor);
}

py::object GetSliceByPaddingShapeHandle(const py::object &object, size_t start, size_t end) {
  auto device_ctx = GetDeviceCtx();
  auto value = ConvertPyObjectToCTensor(object);
  const auto &tensor = value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor);
  auto res_tensor = device_ctx->device_res_manager_->GetSliceByPaddingShapeHandle(tensor, start, end);
  return ValueToPyData(res_tensor);
}

void RegUtils(py::module *m) {
  (void)m->def("combine_tensor_list_contiguous", &mindspore::hal::AllocDeviceMemoryForTensorList,
               "Alloc contiguous memory");
  (void)m->def("slice_by_tensor_index", &mindspore::hal::GetSliceByTensorListIndexHandle, "Slice By TensorList Index");
  (void)m->def("slice_by_padding_shape", &mindspore::hal::GetSliceByPaddingShapeHandle, "Slice By Padding Shape");
}
}  // namespace hal
}  // namespace mindspore
