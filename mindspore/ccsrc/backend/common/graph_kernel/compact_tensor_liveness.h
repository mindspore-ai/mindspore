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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_COMPACT_TENSOR_LIVENESS_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_COMPACT_TENSOR_LIVENESS_H_

#include "include/backend/optimizer/pass.h"

namespace mindspore::graphkernel {
/**
 * @brief For memory efficiency, insert UpdateState for op with no cnode/param inputs to avoid early launching
 * @example
 * main(x, y){
 *   %1 = Op1(x, y)
 *   %2 = Op2()
 *   %3 = Op3(%1, %2)
 *   return %3
 * }
 *  --------------->
 * main(x, y){
 *   %1 = Op1(x, y)
 *   %2 = UpdateState(U, %1)
 *   %3 = Op2(%2)
 *   %4 = Op3(%1, %3)
 *   return %4
 * }
 * ------------------
 * Here, Op2 has no cnode/param inputs and will be launched early, memory space of %2 will be taken up early and only
 * be released after execution of Op3. By insertion of UpdateState, memory space of %2 will be created right before
 * execution of Op3.
 */
class CompactTensorLiveness : public opt::Pass {
 public:
  CompactTensorLiveness() : Pass("compact_tensor_liveness") {}
  ~CompactTensorLiveness() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_COMPACT_TENSOR_LIVENESS_H_
