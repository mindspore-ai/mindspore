/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FL_SERVER_COMMON_H_
#define MINDSPORE_CCSRC_FL_SERVER_COMMON_H_

#include <map>
#include <string>
#include <numeric>
#include <climits>
#include <memory>
#include <functional>
#include <iomanip>
#include "proto/ps.pb.h"
#include "ir/anf.h"
#include "include/common/utils/utils.h"
#include "ir/dtype/type_id.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "ps/ps_context.h"
#include "ps/core/communicator/http_message_handler.h"
#include "ps/core/communicator/tcp_server.h"
#include "ps/core/communicator/message_handler.h"

namespace mindspore {
namespace fl {
namespace server {
// Definitions for the server framework.
enum CommType { HTTP = 0, TCP };

enum class IterationResult {
  // The iteration is failed.
  kFail,
  // The iteration is successful aggregation.
  kSuccess
};

using mindspore::kernel::Address;
using mindspore::kernel::AddressPtr;
using mindspore::kernel::DeprecatedNativeCpuKernelMod;

constexpr auto kSuccess = "Success";

// The result code used for round kernels.
enum class ResultCode {
  // If the method is successfully called and round kernel's residual methods should be called, return kSuccess.
  kSuccess = 0,
  // If there's error happened, return kFail.
  kFail
};
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_COMMON_H_
