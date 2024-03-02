/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_RECOGNIZE_SOFTMAX_GRAD_EXT_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_RECOGNIZE_SOFTMAX_GRAD_EXT_H_

#include <memory>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace graphkernel {
class RecognizeSoftmaxGradExt : public opt::PatternProcessPass {
 public:
  explicit RecognizeSoftmaxGradExt(bool multigraph = true)
      : PatternProcessPass("recognize_softmax_grad_ext", multigraph) {
    mul1_ = std::make_shared<Var>(std::make_shared<Primitive>("Mul"));
    mul2_ = std::make_shared<Var>(std::make_shared<Primitive>("Mul"));
    sub_ = std::make_shared<Var>(std::make_shared<Primitive>("Sub"));
    reduce_sum_ = std::make_shared<Var>(std::make_shared<Primitive>("ReduceSum"));
  }
  ~RecognizeSoftmaxGradExt() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 protected:
  VarPtr mul1_;
  VarPtr mul2_;
  VarPtr sub_;
  VarPtr reduce_sum_;
};
}  // namespace graphkernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_RECOGNIZE_SOFTMAX_GRAD_EXT_H_
