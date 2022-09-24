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

#include "pipeline/jit/event_message_print.h"
#include "utils/log_adapter.h"
#include "pipeline/jit/pipeline.h"

namespace mindspore {
namespace pipeline {
void EventMessage::PrintCompileStartMsg(const std::string &phase, const std::string &obj_desc) {
  if (IsPhaseLoadFromMindIR(phase)) {
    return;
  }
  PrintEventMessage("Start compiling " + obj_desc + " and it will take a while. Please wait...");
}

void EventMessage::PrintCompileEndMsg(const std::string &phase, const std::string &obj_desc) {
  if (IsPhaseLoadFromMindIR(phase)) {
    return;
  }
  PrintEventMessage("End compiling " + obj_desc + ".");
}

void EventMessage::PrintEventMessage(const std::string &message) { MS_LOG(INFO) << message; }
}  // namespace pipeline
}  // namespace mindspore
