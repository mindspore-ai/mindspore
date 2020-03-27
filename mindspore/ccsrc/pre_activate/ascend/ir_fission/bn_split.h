/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_IR_FISSION_BN_SPLIT_H_
#define MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_IR_FISSION_BN_SPLIT_H_

#include "pre_activate/common/optimizer.h"
#include "pre_activate/common/helper.h"

namespace mindspore {
namespace opt {
class BnSplit : public PatternProcessPass {
 public:
  explicit BnSplit(bool multigraph = true) : PatternProcessPass("bn_split", multigraph) {}
  ~BnSplit() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_IR_FISSION_BN_SPLIT_H_
