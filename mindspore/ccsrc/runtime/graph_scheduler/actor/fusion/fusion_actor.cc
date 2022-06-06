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
void FusionActor::Run(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  // The real actor run op data.
  const auto &data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter != input_op_datas_.end()) {
    for (auto &input_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(input_data);
      if (IntToSize(input_data->index_) >= real_input_data_.size()) {
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The input index is out of range.");
      }
      // Update the input data using the real input info.
      auto &real_input_data = real_input_data_[input_data->index_];
      MS_EXCEPTION_IF_NULL(real_input_data.first);
      input_data->index_ = real_input_data.second;

      real_input_data.first->RunOpData(input_data, context);
    }
  }

  // The real actor run op control.
  auto from_aid = const_cast<AID *>(&GetAID());
  for (auto &real_input_control : real_input_controls_) {
    MS_EXCEPTION_IF_NULL(real_input_control);
    real_input_control->RunOpControl(from_aid, context);
  }

  EraseInput(context);
}
}  // namespace runtime
}  // namespace mindspore
