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

#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_PASS_SILENT_CHECK_V2_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_PASS_SILENT_CHECK_V2_H_

#include <vector>
#include "base/base.h"
#include "include/backend/optimizer/optimizer.h"
#include "ir/anf.h"
#include "ir/func_graph.h"

namespace mindspore {
namespace opt {
bool IsNpuAsdEnable();

class SilentCheckV2 : public PatternProcessPass {
 public:
  explicit SilentCheckV2(const FuncGraphPtr &root, bool multigraph = true)
      : PatternProcessPass("insert_silent_check_v2", multigraph), root_(root) {
    GetLossScale();
  }
  ~SilentCheckV2() override = default;

  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  void GetLossScale();

  FuncGraphPtr root_ = nullptr;
  ParameterPtr loss_scale_ = nullptr;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_PASS_SILENT_CHECK_V2_H_
