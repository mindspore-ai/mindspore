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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_FLOATSTATUS_FUSION__H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_FLOATSTATUS_FUSION__H_

#include <memory>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore::graphkernel {
/**
 * @brief Fuse IsFinite and its user to FloatStatus
 * @example
 *   main_graph {
 *     %1 = IsFinite(%0)
 *     %2 = ReduceAll(%1)
 *     %3 = Cast(%2)
 *     %4 = Sub(1, %3)
 *     %5 = Reshape(%4, (1,))
 *     return %5
 *   }
 *   ---------->
 *   main_graph {
 *     %1 = FloatStatus(%0)
 *     return %1
 *   }
 */
class FloatStatusFusion : public opt::PatternProcessPass {
 public:
  explicit FloatStatusFusion(bool multigraph = true)
      : PatternProcessPass("floatstatus_fusion", multigraph),
        input_{std::make_shared<Var>()},
        axis_{std::make_shared<Var>()},
        keep_dims_{std::make_shared<Var>()},
        type_{std::make_shared<Var>()},
        s_{std::make_shared<Var>()},
        to_shape_{std::make_shared<Var>()} {}
  ~FloatStatusFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  VarPtr input_;
  VarPtr axis_;
  VarPtr keep_dims_;
  VarPtr type_;
  VarPtr s_;
  VarPtr to_shape_;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_FLOATSTATUS_FUSION__H_
