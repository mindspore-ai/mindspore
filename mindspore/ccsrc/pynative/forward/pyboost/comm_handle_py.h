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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_OP_FUNCTION_COMM_HANDLE_PY_H
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_OP_FUNCTION_COMM_HANDLE_PY_H

#include <string>
#include <memory>
#include <set>
#include "include/pynative/utils/pyboost/comm_handle.h"

namespace mindspore {
namespace hal {
class CommHandlePy {
 public:
  CommHandlePy() = default;

  explicit CommHandlePy(const device::DeviceContext *device_ctx)
      : device_ctx_(device_ctx), comm_handle_(std::make_shared<kernel::pyboost::CommHandle>(device_ctx)) {}

  ~CommHandlePy();

  void Wait();

  kernel::pyboost::CommHandlePtr comm_handle() const {
    MS_EXCEPTION_IF_NULL(comm_handle_);
    return comm_handle_;
  }

 private:
  const device::DeviceContext *device_ctx_;
  kernel::pyboost::CommHandlePtr comm_handle_;
  std::set<size_t> wait_streams_;
};

using CommHandlePyPtr = std::shared_ptr<CommHandlePy>;
}  // namespace hal
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_OP_FUNCTION_COMM_HANDLE_PY_H
