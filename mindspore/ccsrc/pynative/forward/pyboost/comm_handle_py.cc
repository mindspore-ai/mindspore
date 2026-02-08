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
#include "include/pynative/utils/runtime/task/device_task.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "utils/ms_context.h"
#include "include/pynative/forward/pyboost/forward_task.h"
#include "include/pynative/utils/pynative_utils.h"
#include "pybind_api/runtime/runtime_api.h"

namespace mindspore {
namespace hal {
CommHandlePy::~CommHandlePy() {}

void CommHandlePy::Wait() {
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

void RegCommHandle(py::module *m) {
  (void)py::class_<CommHandlePy, std::shared_ptr<CommHandlePy>>(*m, "CommHandle")
    .def(py::init<>())
    .def("wait", &CommHandlePy::Wait);
}
}  // namespace hal
}  // namespace mindspore
