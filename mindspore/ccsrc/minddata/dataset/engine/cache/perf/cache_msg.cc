/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/cache/perf/cache_msg.h"
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>

namespace mindspore {
namespace dataset {
Status CachePerfMsg::Send(int32_t qID) {
  auto err = msgsnd(qID, reinterpret_cast<void *>(&small_msg_), sizeof(small_msg_.body.msg), IPC_NOWAIT);
  if (err == -1) {
    std::string errMsg = "Failed to call msgsnd. Errno = " + std::to_string(errno);
    RETURN_STATUS_UNEXPECTED(errMsg);
  }
  return Status::OK();
}

Status CachePerfMsg::Receive(int32_t qID) {
  // This is a blocking call. Either there is some message or we the queue is removed when
  // the destructor is called.
  auto err = msgrcv(qID, reinterpret_cast<void *>(&small_msg_), sizeof(small_msg_.body.msg), 0, MSG_NOERROR);
  if (err == -1) {
    if (errno == EIDRM) {
      return Status(StatusCode::kMDInterrupted);
    } else {
      std::string errMsg = "Failed to call msgrcv. Errno = " + std::to_string(errno);
      RETURN_STATUS_UNEXPECTED(errMsg);
    }
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
