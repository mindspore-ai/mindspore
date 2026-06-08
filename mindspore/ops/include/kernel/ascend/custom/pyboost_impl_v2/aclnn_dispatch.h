/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_EXTENSION_ASCEND_ACLNN_DISPATCH_H_
#define MINDSPORE_CCSRC_EXTENSION_ASCEND_ACLNN_DISPATCH_H_

#include "include/kernel/ascend/custom/pyboost_impl/aclnn_op_runner.h"
#include "include/pynative/utils/pyboost/custom_v2/op_api.h"

/// @brief Stateless custom_v2 ACLNN dispatch macro.
/// Converts ms::Tensor arguments to TensorPtr via Arg(), dispatches LAUNCH_ACLNN asynchronously
/// on the device task thread.
///
/// Usage:
/// @code
///   ms::TensorToDevice(x);
///   ms::TensorAllocate(result);
///   MS_DISPATCH_ACLNN(aclnnCast, x, type_id, result);
/// @endcode
#define MS_DISPATCH_ACLNN(aclnn_api, ...)                                                                            \
  do {                                                                                                               \
    [](auto &&...args) {                                                                                             \
      auto args_t = std::make_tuple(ms::pynative::Arg(std::forward<decltype(args)>(args))...);                       \
      auto *dc = ms::GetCurrentDeviceContext();                                                                      \
      auto sid = static_cast<size_t>(mindspore::CurrentStream::id());                                                \
      mindspore::kernel::pyboost::PyBoostUtils::DispatchRun(std::make_shared<mindspore::runtime::PyBoostDeviceTask>( \
        [=]() { std::apply([&](auto &&...a) { LAUNCH_ACLNN(aclnn_api, dc, sid, a...); }, args_t); }));               \
    }(__VA_ARGS__);                                                                                                  \
  } while (false)

#endif  // MINDSPORE_CCSRC_EXTENSION_ASCEND_ACLNN_DISPATCH_H_
