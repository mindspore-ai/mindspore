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

#include "runtime/framework/actor/loop_count_actor.h"
#include "runtime/framework/actor/data_source_actor.h"
#include "runtime/framework/actor/kernel_actor.h"
#include "mindrt/include/async/async.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
void LoopCountActor::RunOpControl(AID *input_control, OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  auto sequential_num = context->sequential_num_;
  input_op_controls_[sequential_num].emplace_back(input_control);
  if (input_op_controls_[sequential_num].size() == input_controls_num_) {
    current_count_++;
    if (current_count_ == loop_count_) {
      current_count_ = 0;
      SET_OPCONTEXT_SUCCESS_RET((*context));
    }

    // Send output control.
    for (auto &data_source_aid : data_source_aids_) {
      Async(data_source_aid, &DataSourceActor::FetchData, context);
    }
    auto source_aid = const_cast<AID *>(&GetAID());
    for (auto &kernel_aid : no_input_kernel_aids_) {
      Async(kernel_aid, &KernelActor::RunOpControl, source_aid, context);
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
