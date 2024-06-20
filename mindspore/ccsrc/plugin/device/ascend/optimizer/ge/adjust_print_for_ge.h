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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_GE_ADJUST_PRINT_FOR_GE_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_GE_ADJUST_PRINT_FOR_GE_H_

#include <vector>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class AdjustPrintForGe : public PatternProcessPass {
 public:
  explicit AdjustPrintForGe(bool multigraph = true)
      : PatternProcessPass("print_insert_placeholder_for_tensor_name", multigraph) {}
  ~AdjustPrintForGe() override = default;

  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  void UnfoldMakeTuple(const AnfNodePtr &input, int64_t *ptr_num_inputs, std::vector<AnfNodePtr> *ptr_new_inputs) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_GE_ADJUST_PRINT_FOR_GE_H_
