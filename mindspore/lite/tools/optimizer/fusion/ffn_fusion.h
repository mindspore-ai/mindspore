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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_FFN_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_FFN_FUSION_H_

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include "tools/optimizer/common/multiple_pattern_process_pass.h"

namespace mindspore {
namespace opt {
enum PatternType {
  kDynamicDims,
  kConstFold,
  kMaxPatternNum,
};

class FFNFusion : public MultiplePatternProcessPass {
 public:
  explicit FFNFusion(const std::string &name = "FFNFusion", bool multigraph = true)
      : MultiplePatternProcessPass(name, multigraph) {}

  ~FFNFusion() override = default;

  std::unordered_map<std::string, VectorRef> DefinePatterns() const override;

  AnfNodePtr Process(const std::string &pattern_name, const FuncGraphPtr &, const AnfNodePtr &,
                     const EquivPtr &) const override;

 private:
  bool Init() const;
  bool CheckPattern(const std::string &pattern_name, const EquivPtr &equiv) const;

  const VectorRef DefineFFNPatternForConstFolding() const;
  const VectorRef DefineFFNPatternForDynamicDims() const;

  CNodePtr CreateFFNFusionNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &equiv,
                               int index) const;

  mutable VarPtr input_[kMaxPatternNum] = {nullptr};

  mutable VarPtr gather_y_{nullptr};
  mutable VarPtr add2_y_{nullptr};
  mutable VarPtr div1_y_{nullptr};
  mutable VarPtr mul1_y_{nullptr};
  mutable VarPtr mul2_y_{nullptr};

  mutable VarPtr div2_y_[kMaxPatternNum] = {nullptr};
  mutable VarPtr add3_y_[kMaxPatternNum] = {nullptr};
  mutable VarPtr mul4_y_[kMaxPatternNum] = {nullptr};
  mutable VarPtr matmul1_b_[kMaxPatternNum] = {nullptr};
  mutable VarPtr add1_x_[kMaxPatternNum] = {nullptr};
  mutable VarPtr matmul2_b_[kMaxPatternNum] = {nullptr};
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_FFN_FUSION_H_
