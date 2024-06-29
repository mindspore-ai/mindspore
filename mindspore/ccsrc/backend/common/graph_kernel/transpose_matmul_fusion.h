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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_TRANSPOSE_MATMUL_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_TRANSPOSE_MATMUL_FUSION_H_
#include "include/backend/optimizer/pass.h"

namespace mindspore::graphkernel {
/**
 * @brief Transform Transpose + MutMul to a single MatMul with attribute trans_a/trans_b
 * @example
 *   %1 = Transpose(B, (1, 0))
 *   %2 = MatMul(A, %1, trans_a=false, trans_b=false)
 *   ---------->
 *   %1 = MatMul(A, B, trans_a=false, trans_b=true)
 * @example
 *   %1 = Transpose(A, (0, 1, 3, 2))
 *   %2 = BatchMatMul(%1, B, trans_a=false, trans_b=false)
 *   ---------->
 *   %1 = BatchMatMul(A, B, trans_a=true, trans_b=false)
 */
class TransposeMatmulFusion : public opt::Pass {
 public:
  TransposeMatmulFusion() : Pass("transpose_matmul_fusion") {}
  ~TransposeMatmulFusion() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_TRANSPOSE_MATMUL_FUSION_H_
