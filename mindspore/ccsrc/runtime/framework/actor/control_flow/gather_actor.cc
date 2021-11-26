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

#include "runtime/framework/actor/control_flow/gather_actor.h"
#include "runtime/framework/actor/control_flow/entrance_actor.h"

namespace mindspore {
namespace runtime {
GatherActor::GatherActor(const std::string &name, const std::vector<KernelWithIndex> &parameters,
                         const AnfNodePtr &node)
    : ControlActor(name, KernelTransformType::kGatherActor, parameters, node) {
  device_contexts_.resize(parameters.size());
}

void GatherActor::FetchInput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);

  ControlActor::FetchInput(context);
  MS_EXCEPTION_IF_NULL(input_partials_[0].first);

  // Put other real parameter in partial.
  for (const auto &device_tensor : input_device_tensors_) {
    if (device_tensor != nullptr) {
      input_partials_[0].second.emplace_back(device_tensor);
    }
  }
}

void GatherActor::SendOutput(OpContext<DeviceTensor> *const context) {
  // Send data with branch id.
  const auto &iter = output_data_with_branch_id_arrows_.find(input_partials_[0].first);
  if (iter != output_data_with_branch_id_arrows_.end()) {
    for (const auto &data_with_branch_id_arrow : iter->second) {
      ActorDispatcher::Send(data_with_branch_id_arrow, &EntranceActor::RunOpDataWithBranchID, input_partials_[0].second,
                            output_branch_id_, context);
    }
  }

  // Control arrow needs to be sent after the real parameter data and branch id.
  ControlActor::SendOutput(context);
}
}  // namespace runtime
}  // namespace mindspore
