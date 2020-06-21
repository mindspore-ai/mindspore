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

#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_FORMAT_TYPE_MODIFY_OPS_ATTRS_H
#define MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_FORMAT_TYPE_MODIFY_OPS_ATTRS_H

#include "pre_activate/common/optimizer.h"

namespace mindspore {
namespace opt {
class ModifyOpAttrs : public PatternProcessPass {
 public:
  explicit ModifyOpAttrs(bool multigraph = true) : PatternProcessPass("modify_ops_attrs", multigraph) {}
  ~ModifyOpAttrs() override = default;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_FORMAT_TYPE_MODIFY_OPS_ATTRS_H
