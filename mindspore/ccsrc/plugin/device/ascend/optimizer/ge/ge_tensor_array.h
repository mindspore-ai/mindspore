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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_GE_OPTIMIZER_IRPASS_GE_TENSOR_ARRAY_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_GE_OPTIMIZER_IRPASS_GE_TENSOR_ARRAY_H_

#include <memory>
#include <string>
#include "include/backend/optimizer/optimizer.h"
namespace mindspore {
namespace opt {
class GeTensorArrayAddFlow : public PatternProcessPass {
 public:
  explicit GeTensorArrayAddFlow(const std::string &name = "", bool multigraph = true)
      : PatternProcessPass(name, multigraph) {}
  ~GeTensorArrayAddFlow() override = default;

  const BaseRef DefinePattern() const override = 0;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};

class GeTensorArrayAddFlowCond1 : public GeTensorArrayAddFlow {
 public:
  explicit GeTensorArrayAddFlowCond1(bool multigraph = true)
      : GeTensorArrayAddFlow("ge_tensor_arry_add_flow_cond1", multigraph) {}
  ~GeTensorArrayAddFlowCond1() override = default;

  const BaseRef DefinePattern() const override;
};

class GeTensorArrayAddFlowCond2 : public GeTensorArrayAddFlow {
 public:
  explicit GeTensorArrayAddFlowCond2(bool multigraph = true)
      : GeTensorArrayAddFlow("ge_tensor_arry_add_flow_cond2", multigraph) {}
  ~GeTensorArrayAddFlowCond2() override = default;

  const BaseRef DefinePattern() const override;
};

class GeTensorArrayCastIndex : public PatternProcessPass {
 public:
  explicit GeTensorArrayCastIndex(const std::string &name = "ge_tensor_arry_cast_index", bool multigraph = true)
      : PatternProcessPass(name, multigraph) {}
  ~GeTensorArrayCastIndex() override = default;

  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_GE_OPTIMIZER_IRPASS_GE_TENSOR_ARRAY_H_
