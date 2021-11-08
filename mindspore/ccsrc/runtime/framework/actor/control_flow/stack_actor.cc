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

#include "runtime/framework/actor/control_flow/stack_actor.h"
#include "runtime/framework/actor/memory_manager_actor.h"
#include "runtime/framework/control_node_parser.h"

namespace mindspore {
namespace runtime {
StackActor::StackActor(const std::string &name, const std::vector<KernelWithIndex> &parameters)
    : ControlActor(name, KernelTransformType::kStackActor, parameters, nullptr) {
  input_device_tensors_.resize(parameters.size());
}

bool StackActor::CheckRunningCondition(const OpContext<DeviceTensor> *context) const {
  MS_EXCEPTION_IF_NULL(context);
  return false;
}

void StackActor::FetchInput(OpContext<DeviceTensor> *const context) { MS_EXCEPTION_IF_NULL(context); }

void StackActor::EraseInput(const OpContext<DeviceTensor> *const context) { MS_EXCEPTION_IF_NULL(context); }
}  // namespace runtime
}  // namespace mindspore
