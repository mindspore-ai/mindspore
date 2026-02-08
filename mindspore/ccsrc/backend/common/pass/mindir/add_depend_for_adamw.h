/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_MINDIR_ADD_DEPEND_FOR_ADAMW_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_MINDIR_ADD_DEPEND_FOR_ADAMW_H_

#include <vector>
#include <memory>
#include <string>
#include "include/backend/common/pass_manager/pass.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace opt {
class BACKEND_COMMON_EXPORT AddDependForAdamW : public Pass {
 public:
  AddDependForAdamW() : Pass("add_depend_for_adamw") {}
  ~AddDependForAdamW() override = default;
  bool Run(const FuncGraphPtr &graph) override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_MINDIR_ADD_DEPEND_FOR_ADAMW_H_
