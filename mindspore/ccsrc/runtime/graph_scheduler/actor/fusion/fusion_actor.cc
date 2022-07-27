/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/actor/fusion/fusion_actor.h"

namespace mindspore {
namespace runtime {
void FusionActor::RunOpData(OpData<DeviceTensor> *const input_data, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(input_data);
  MS_EXCEPTION_IF_NULL(context);
  MS_LOG(DEBUG) << "Actor(" << GetAID().Name() << ") receive the input op data.";

  if (IntToSize(input_data->index_) >= real_input_data_.size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The input index is out of range.");
  }
  // Update the input data using the real input info.
  auto &real_input_data = real_input_data_[IntToSize(input_data->index_)];
  MS_EXCEPTION_IF_NULL(real_input_data.first);
  input_data->index_ = SizeToInt(real_input_data.second);

  real_input_data.first->RunOpData(input_data, context);
}

void FusionActor::RunOpControl(AID *const input_control, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(input_control);
  MS_EXCEPTION_IF_NULL(context);
  MS_LOG(DEBUG) << "Actor(" << GetAID().Name() << ") receive the input op control: " << input_control->Name();

  // Because the input controls of fusion actor are difficult to eliminate duplication, so only process the first input
  // control when receive the same input controls.
  if (recv_input_control_actors_.count(input_control->Name()) == 0) {
    (void)recv_input_control_actors_.insert(input_control->Name());
    if (real_input_controls_.count(input_control->Name()) == 0) {
      std::string err_info = GetAID().Name() + " has no real input control from receiving " + input_control->Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), err_info);
    }
    for (auto &real_input_control_actor : real_input_controls_[input_control->Name()]) {
      MS_EXCEPTION_IF_NULL(real_input_control_actor);
      real_input_control_actor->RunOpControl(input_control, context);
    }
  }

  ++recv_input_controls_num_;
  if (recv_input_controls_num_ == input_controls_num_) {
    recv_input_controls_num_ = 0;
    recv_input_control_actors_.clear();
  }
}
}  // namespace runtime
}  // namespace mindspore
