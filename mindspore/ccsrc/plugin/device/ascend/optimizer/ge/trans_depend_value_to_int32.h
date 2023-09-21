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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_MINDIR_TRANS_DEPEND_VALUE_TO_INT32_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_MINDIR_TRANS_DEPEND_VALUE_TO_INT32_

#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/pattern_to_pattern.h"

namespace mindspore::opt {
class TransDependValueToInt32 : public PatternProcessPass {
 public:
  explicit TransDependValueToInt32(bool multigraph = true)
      : PatternProcessPass("trans_depend_value_to_int32", multigraph) {}
  ~TransDependValueToInt32() override = default;

  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const override;
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_MINDIR_TRANS_DEPEND_VALUE_TO_INT32_
