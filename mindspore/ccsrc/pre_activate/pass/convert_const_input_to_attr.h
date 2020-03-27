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
#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_PASS_CONVERT_CONST_INPUT_TO_ATTR_H_
#define MINDSPORE_CCSRC_PRE_ACTIVATE_PASS_CONVERT_CONST_INPUT_TO_ATTR_H_
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "ir/anf.h"
#include "pre_activate/common/optimizer.h"

namespace mindspore {
namespace opt {
class ConvertConstInputToAttr : public PatternProcessPass {
 public:
  explicit ConvertConstInputToAttr(bool multigraph = true)
      : PatternProcessPass("convert_const_input_to_attr", multigraph) {
    Init();
  }
  ~ConvertConstInputToAttr() override = default;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  void Init();
  std::unordered_map<std::string, std::unordered_set<size_t>> op_input_attr_map_;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_PASS_CONVERT_CONST_INPUT_TO_ATTR_H_
