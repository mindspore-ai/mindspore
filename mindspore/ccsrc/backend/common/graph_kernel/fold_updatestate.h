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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_FOLD_UPDATESTATES_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_FOLD_UPDATESTATES_H_
#include "include/backend/optimizer/pass.h"

namespace mindspore::graphkernel {
/**
 * @brief Fold UpdateStates If Order is not guaranteed by those UpdateStates
 * @example
 *   %1 = UpdateState(...)
 *   %2 = call1(...)
 *   %3 = UpdateState(%1, %2)
 *   %4 = call2(...)
 *   %5 = UpdateState(%3, %4)
 *   ---------->
 *   %1 = UpdateState(...)
 *   %2 = call1(...)
 *   %3 = call2(...)
 *   %4 = UpdateState(%1, %2, %3)
 */
class FoldUpdateState : public opt::Pass {
 public:
  FoldUpdateState() : Pass("fold_updatestates") {}
  ~FoldUpdateState() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_FOLD_UPDATESTATES_H_
