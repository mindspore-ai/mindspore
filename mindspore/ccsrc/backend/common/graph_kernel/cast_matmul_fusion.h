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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CAST_MATMUL_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CAST_MATMUL_FUSION_H_

#include <map>
#include <memory>

#include "backend/common/optimizer/pass.h"
#include "ir/func_graph.h"

namespace mindspore::graphkernel {
class CastMatmulFusion : public opt::Pass {
 public:
  CastMatmulFusion() : Pass("cast_matmul_fusion") {}
  ~CastMatmulFusion() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};
using OptimizeMatmulPtr = std::shared_ptr<CastMatmulFusion>;
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CAST_MATMUL_FUSION_H_
