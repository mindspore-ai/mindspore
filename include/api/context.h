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
#ifndef MINDSPORE_INCLUDE_API_CONTEXT_H
#define MINDSPORE_INCLUDE_API_CONTEXT_H

#include <string>
#include <memory>
#include "include/api/types.h"

namespace mindspore {
namespace api {
class MS_API Context {
 public:
  static Context &Instance();
  const std::string &GetDeviceTarget() const;
  Context &SetDeviceTarget(const std::string &device_target);
  uint32_t GetDeviceID() const;
  Context &SetDeviceID(uint32_t device_id);

 private:
  Context();
  ~Context();
  class ContextImpl;
  std::shared_ptr<ContextImpl> impl_;
};
}  // namespace api
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_CONTEXT_H
