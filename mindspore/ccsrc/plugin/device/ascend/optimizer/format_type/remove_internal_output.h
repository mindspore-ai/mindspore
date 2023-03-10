/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FORMAT_TYPE_REMOVE_INTERNAL_OUTPUT_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FORMAT_TYPE_REMOVE_INTERNAL_OUTPUT_H_

#include <string>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class RemoveInternalOutput : public PatternProcessPass {
 public:
  explicit RemoveInternalOutput(const std::string &name, bool multigraph = true)
      : PatternProcessPass(name, multigraph) {}
  ~RemoveInternalOutput() override = default;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;
};

class RemoveInternalOutputTransOp : public RemoveInternalOutput {
 public:
  explicit RemoveInternalOutputTransOp(bool multigraph = true)
      : RemoveInternalOutput("remove_internal_output_trans_op", multigraph) {}
  ~RemoveInternalOutputTransOp() override = default;
  const BaseRef DefinePattern() const override;
};

class RemoveInternalOutputCast : public RemoveInternalOutput {
 public:
  explicit RemoveInternalOutputCast(bool multigraph = true)
      : RemoveInternalOutput("remove_internal_output_cast", multigraph) {}
  ~RemoveInternalOutputCast() override = default;
  const BaseRef DefinePattern() const override;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FORMAT_TYPE_REMOVE_INTERNAL_OUTPUT_H_
