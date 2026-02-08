/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_ACL_ADAPTER_H_
#define MINDSPORE_MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_ACL_ADAPTER_H_
#include <vector>
#include <string>
#include "utils/callback_handler.h"
#include "include/utils/visible.h"

namespace mindspore {
namespace acl_adapter {
class AclAdapterCallback {
  HANDLER_DEFINE(std::string, GetAclGraphInfoFunc, const std::vector<ValuePtr> &, const PrimitivePtr &,
                 const std::string &);
};
}  // namespace acl_adapter
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_ACL_ADAPTER_H_
