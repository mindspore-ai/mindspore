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
#ifndef MINDSPORE_CCSRC_BACKEND_SESSION_ASCEND_AUTO_MONAD_H
#define MINDSPORE_CCSRC_BACKEND_SESSION_ASCEND_AUTO_MONAD_H

#include "backend/session/kernel_graph.h"

namespace mindspore {
namespace session {
//
// AscendAutoMonad handle control flow with auto-monad for Ascend backend.
//
class AscendAutoMonad {
 public:
  explicit AscendAutoMonad(NotNull<KernelGraphPtr> kg) : kernel_graph_(kg) {}
  ~AscendAutoMonad() = default;

  // Handle control flow calls by auto-monad.
  void Run();

  // Generate execute order by join sub graphs.
  void GenerateExecuteOrder();

 private:
  NotNull<KernelGraphPtr> kernel_graph_;
};
}  // namespace session
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_SESSION_ASCEND_AUTO_MONAD_H
