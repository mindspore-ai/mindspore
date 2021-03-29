/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_ACTOR_COMMON_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_ACTOR_COMMON_H_

#include <utility>
#include "mindrt/include/actor/op_actor.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
// The execution result of actor.
constexpr int kSuccess = 0;
constexpr int kFailure = 1;

#define SET_OPCONTEXT_FAIL_RET_WITH_ERROR(op_context, message) \
  {                                                            \
    MS_LOG(ERROR) << message;                                  \
    op_context.SetFailed(kFailure);                            \
    return;                                                    \
  }

#define SET_OPCONTEXT_SUCCESS_RET(op_context) \
  {                                           \
    op_context.SetSuccess(kSuccess);          \
    return;                                   \
  }

}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_ACTOR_COMMON_H_
