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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_FLASH_ATTENTION_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_FLASH_ATTENTION_FUSION_H_

#include <string>
#include <memory>
#include "tools/optimizer/common/pattern_process_pass_extends.h"

namespace mindspore {
namespace opt {
class FlashAttentionFusion : public LitePatternProcessPass {
 public:
  explicit FlashAttentionFusion(bool multigraph = true, const std::string &name = "FlashAttentionFusion")
      : LitePatternProcessPass(name, multigraph) {}

  ~FlashAttentionFusion() override = default;

  const BaseRef DefinePattern() const override;

  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &node, const EquivPtr &equiv) const override;

 private:
  bool InitVar() const;
  bool CheckBatchMatmulTranspose(const CNodePtr &batchmm_cnode, const bool expected_transpose_a,
                                 const bool expected_transpose_b) const;
  bool CheckInputShape(const CNodePtr &cnode, const uint32_t input_index, const uint32_t expected_rank_num,
                       const uint32_t expected_seq_len) const;
  CNodePtr CreateFlashAttentionNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                    const EquivPtr &equiv) const;

 protected:
  mutable VarPtr input_0_batchmm_qk_ = nullptr;
  mutable VarPtr input_1_batchmm_qk_ = nullptr;
  mutable VarPtr input_1_batchmm_sv_ = nullptr;
  mutable VarPtr input_0_mul_ = nullptr;
  mutable bool has_add_cast_ = false;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_FLASH_ATTENTION_FUSION_H_
