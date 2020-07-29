/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_INCLUDE_LITE_SESSION_H
#define MINDSPORE_LITE_INCLUDE_LITE_SESSION_H

#include <memory>
#include <vector>
#include <string>
#include "include/ms_tensor.h"
#include "include/model.h"
#include "include/context.h"

namespace mindspore {
namespace session {
class MS_API LiteSession {
 public:
  virtual ~LiteSession() = default;

  virtual void BindThread(bool ifBind) = 0;

  static LiteSession *CreateSession(lite::Context *context);

  virtual int CompileGraph(lite::Model *model) = 0;

  virtual std::vector<tensor::MSTensor *> GetInputs() = 0;

  virtual std::vector<tensor::MSTensor *> GetInputsByName(std::string name) = 0;

  virtual int RunGraph() = 0;

  virtual std::vector<tensor::MSTensor *> GetOutputs() = 0;

  virtual std::vector<tensor::MSTensor *> GetOutputsByName(std::string name) = 0;
};
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_LITE_INCLUDE_LITE_SESSION_H

