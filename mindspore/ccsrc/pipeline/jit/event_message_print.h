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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_EVENT_MESSAGE_PRINT_H
#define MINDSPORE_CCSRC_PIPELINE_JIT_EVENT_MESSAGE_PRINT_H

#include <iostream>
#include <string>

namespace mindspore {
namespace pipeline {
class EventMessage {
 public:
  static void PrintCompileStartMsg(const std::string &phase);
  static void PrintCompileEndMsg(const std::string &phase);

  static void PrintEventMessage(const std::string &message);
};
}  // namespace pipeline
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_EVENT_MESSAGE_PRINT_H
