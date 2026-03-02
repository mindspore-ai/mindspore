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

#include "pynative/forward/pyboost/comm_handle_py.h"
#include <sstream>
#include "include/pynative/utils/runtime/task/device_task.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "utils/ms_context.h"
#include "include/pynative/forward/pyboost/forward_task.h"
#include "include/pynative/utils/pynative_utils.h"
#include "pybind_api/runtime/runtime_api.h"

namespace mindspore {
namespace hal {
CommHandlePy::~CommHandlePy() {
  if (!py_wait_called_) {
    MS_LOG(WARNING)
      << "CommHandlePy destroyed without calling wait(): comm_handle=" << GetCommHandlePtrStr()
      << ". Please ensure to call wait() on the CommHandlePy returned from async communication operations.";
  } else {
    if (comm_handle_ != nullptr) {
      // Check if underlying CommHandle's wait_called_ is true, if not wait for it
      if (!comm_handle_->IsWaitCalled()) {
        MS_LOG(INFO) << "CommHandlePy destroyed, waiting for CommHandle wait_called_ to become true: comm_handle="
                     << GetCommHandlePtrStr();
        comm_handle_->WaitForWaitCalled();
        MS_LOG(INFO) << "CommHandlePy wait completed: comm_handle=" << GetCommHandlePtrStr();
      }
    }
  }
}

void CommHandlePy::Wait() {
  py_wait_called_ = true;
  if (comm_handle_ == nullptr) {
    MS_LOG(DEBUG) << "handle is null, no need to wait.";
    return;
  }

  MS_EXCEPTION_IF_NULL(device_ctx_);
  auto cur_stream_id = device_ctx_->device_res_manager_->GetCurrentStreamId();
  auto ret = wait_streams_.insert(cur_stream_id);
  if (!ret.second) {
    MS_LOG(INFO) << "Event already wait on stream " << cur_stream_id << ". Skip wait.";
    return;
  }

  // Wait event async.
  pynative::DispatchOp(
    std::make_shared<pynative::PassthroughFrontendTask>([comm_handle = comm_handle_]() { WaitTaskFunc(comm_handle); }));
  MS_LOG(DEBUG) << "release handle after wait";
}

std::string CommHandlePy::GetCommHandlePtrStr() const {
  if (comm_handle_ == nullptr) {
    return "nullptr";
  }
  std::ostringstream oss;
  oss << comm_handle_.get();
  return oss.str();
}

void RegCommHandle(py::module *m) {
  (void)py::class_<CommHandlePy, std::shared_ptr<CommHandlePy>>(*m, "CommHandle")
    .def(py::init<>())
    .def("wait", &CommHandlePy::Wait)
    .def("get_comm_handle_ptr_str", &CommHandlePy::GetCommHandlePtrStr);
}
}  // namespace hal
}  // namespace mindspore
