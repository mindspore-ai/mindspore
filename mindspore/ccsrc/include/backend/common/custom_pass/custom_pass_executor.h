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

#ifndef MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_CUSTOM_PASS_CUSTOM_PASS_EXECUTOR_H_
#define MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_CUSTOM_PASS_CUSTOM_PASS_EXECUTOR_H_

#include <string>
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace opt {

class BACKEND_COMMON_EXPORT CustomPassExecutor {
 public:
  // Execute custom passes for specified device
  static void ExecuteCustomPasses(const KernelGraphPtr &graph, const std::string &device_target);

 private:
  CustomPassExecutor() = delete;
  ~CustomPassExecutor() = delete;
};

}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_CUSTOM_PASS_CUSTOM_PASS_EXECUTOR_H_
