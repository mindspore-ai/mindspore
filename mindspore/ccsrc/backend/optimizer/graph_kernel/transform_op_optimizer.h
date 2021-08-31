/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_TRANSFORM_OP_OPTIMIZER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_TRANSFORM_OP_OPTIMIZER_H_

#include <string>
#include <vector>
#include "backend/optimizer/common/pass.h"
#include "ir/func_graph.h"
#include "backend/optimizer/graph_kernel/model/lite_graph.h"

namespace mindspore {
namespace opt {
/**
 * @brief Eliminate the unnecessary transformation ops when the other operators
 *        are format flexible.
 * @example
 *   %1 = Transpose(p0) // NCHW to NHWC
 *   %2 = Transpose(p1) // NCHW to NHWC
 *   %3 = Add(%1, %2)
 *   return %3
 *  -->
 *   %1 = Add(p0, p1)
 *   %2 = Transpose(%1) // NCHW to NHWC
 *   return %2
 * @example
 *   %1 = Transpose(p0) // NCHW to NHWC
 *   %2 = Transpose(p1) // NCHW to NHWC
 *   %3 = Add(%1, %2)
 *   %4 = Transpose(%3) // NHWC to NCHW
 *   return %4
 *  -->
 *   %1 = Add(p0, p1)
 *   return %1
 */
class TransformOpOptimizer : public Pass {
 public:
  TransformOpOptimizer() : Pass("transform_op_optimizer") {}
  ~TransformOpOptimizer() = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  bool Process(const graphkernel::LiteGraphPtr &litegraph, const std::string &trans_op_name = "Transpose");
  bool IsFlexibleOp(const graphkernel::NodePtr &node);
  size_t ori_trans_op_num_{0};
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_TRANSFORM_OP_OPTIMIZER_H_
