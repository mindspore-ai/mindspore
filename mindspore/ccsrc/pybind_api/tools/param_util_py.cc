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
#include <utility>
#include <vector>
#include <algorithm>
#include "ir/anf.h"
#include "ir/tensor_new.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "ir/tensor_storage_info.h"
#include "tools/error_handler/error_handler.h"
#include "utils/ms_context.h"

#include "include/utils/tensor_py.h"
#include "include/runtime/hardware_abstract/stream/multi_stream_controller.h"
#include "runtime/hardware_abstract/utils.h"
#include "mindspore/ccsrc/pybind_api/tools/tools_api.h"

namespace mindspore {
namespace {
int SendRecv(const std::vector<py::object> &params, int src_rank, int dst_rank) {
  device::DeviceContextManager::GetInstance().SyncAllStreams();
  tensor::TensorPtrList tensors;
  (void)std::transform(params.begin(), params.end(), std::back_inserter(tensors),
                       [](const py::object &p) { return tensor::ConvertToTensor(p); });
  return tools::ErrorHandler::GetInstance().SendRecv(tensors, src_rank, dst_rank);
}

int ResetParams(const std::vector<py::object> &params) {
  const auto &device_name = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto device_ctx = device::DeviceContextManager::GetInstance().GetDeviceContext(device_name);
  MS_EXCEPTION_IF_NULL(device_ctx);
  MS_EXCEPTION_IF_NULL(device_ctx->device_res_manager_);
  device::DeviceContextManager::GetInstance().SyncAllStreams();
  tensor::TensorPtrList params_;
  (void)std::transform(params.begin(), params.end(), std::back_inserter(params_),
                       [](const py::object &p) { return tensor::ConvertToTensor(p); });
  return device_ctx->device_res_manager_->ResetParams(params_);
}

py::object DirectCopyToHost(const py::object &param) {
  const auto &device_name = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto device_ctx = device::DeviceContextManager::GetInstance().GetDeviceContext(device_name);
  MS_EXCEPTION_IF_NULL(device_ctx);
  MS_EXCEPTION_IF_NULL(device_ctx->device_res_manager_);
  auto tensor = tensor::ConvertToTensor(param);
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->device_address() == nullptr) {
    MS_EXCEPTION(ValueError) << "Can not copy uninitialized tensor " << tensor->ToString();
  }
  if (tensor->device_address()->GetDeviceType() == device::DeviceType::kCPU) {
    return tensor::PackTensorToPyObject(tensor);
  }

  auto host_tensor = tensor::from_spec(static_cast<mindspore::TypeId>(tensor->data_type_c()), tensor->shape_c(),
                                       device::DeviceType::kCPU);
  auto size = tensor->Size();
  if (!device_ctx->device_res_manager_->CopyDirectly(
        host_tensor->data_c(), size, tensor->device_address()->GetMutablePtr(), size, device::CopyType::kD2H)) {
    MS_LOG(EXCEPTION) << "Copy tensor data from device to host failed.";
  }
  return tensor::PackTensorToPyObject(host_tensor);
}
}  // namespace

void RegParamUtils(py::module *m) {
  (void)m->def("send_recv", &mindspore::SendRecv, "Send and receive parameters", py::arg("params"), py::arg("src_rank"),
               py::arg("dst_rank"));
  (void)m->def("reset_params", &mindspore::ResetParams, "Reset parameter's value to zero", py::arg("params"));
  (void)m->def("direct_copy_to_host", &mindspore::DirectCopyToHost, "Copy tensor data from device to host directly",
               py::arg("param"));
}
}  // namespace mindspore
