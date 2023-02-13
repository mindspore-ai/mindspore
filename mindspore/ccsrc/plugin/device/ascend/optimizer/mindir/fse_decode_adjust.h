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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FSE_DECODE_ADJUST_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FSE_DECODE_ADJUST_H_

#include <vector>
#include <string>
#include "backend/common/optimizer/optimizer.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace opt {
class FSEDecodeAdjust : public PatternProcessPass {
 public:
  explicit FSEDecodeAdjust(bool multigraph = true) : PatternProcessPass("fse_decode_adjust", multigraph) {}
  ~FSEDecodeAdjust() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  std::vector<std::string> MustExistPrimitiveName() const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FSE_DECODE_ADJUST_H_
