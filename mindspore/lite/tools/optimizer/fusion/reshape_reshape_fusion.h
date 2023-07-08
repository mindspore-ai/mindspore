/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_RESHAPE_RESHAPE_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_RESHAPE_RESHAPE_FUSION_H_

#include <string>
#include <memory>
#include <unordered_map>
#include "tools/optimizer/common/multiple_pattern_process_pass.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace opt {
class ReshapeReshapeFusion : public MultiplePatternProcessPass {
 public:
  explicit ReshapeReshapeFusion(bool multigraph = true, const std::string &name = "ReshapeReshapeFusion")
      : MultiplePatternProcessPass(name, multigraph) {}
  ~ReshapeReshapeFusion() override = default;
  AnfNodePtr Process(const std::string &pattern_name, const FuncGraphPtr &, const AnfNodePtr &,
                     const EquivPtr &) const override;
  std::unordered_map<std::string, VectorRef> DefinePatterns() const override;

 private:
  VectorRef DefinePreReshapePattern() const;
  VectorRef DefinePostReshapePattern() const;
  VectorRef DefineReshapeReshapePattern() const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_RESHAPE_RESHAPE_FUSION_H_
