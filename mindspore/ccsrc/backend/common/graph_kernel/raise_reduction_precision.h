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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_RAISE_REDUCTION_PRECISION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_RAISE_REDUCTION_PRECISION_H_

#include <string>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore::graphkernel {
class RaiseReductionPrecision : public opt::Pass {
 public:
  RaiseReductionPrecision() : Pass("raise_reduction_precision") {}
  ~RaiseReductionPrecision() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  bool IsFp16ReduceSum(const AnfNodePtr &node) const;
  bool Process(const FuncGraphPtr &func_graph) const;
  AnfNodePtr CreateCast(const AnfNodePtr &input, const TypePtr &dst_type, const std::string &format) const;
  AnfNodePtr CreateReduceSum(const AnfNodePtr &node, const AnfNodePtr &input) const;
  void ReplaceNode(const AnfNodePtr &reduce_node, const AnfNodePtr &cast_node) const;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_RAISE_REDUCTION_PRECISION_H_
