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
#include <unordered_map>
#include <vector>
#include <map>
#include "tools/optimizer/common/multiple_pattern_process_pass.h"
namespace mindspore {
namespace opt {
/*
 *
 * --------------------------------------------------------------------------------------------------------
 *  Pattern 1: [vae_decoder\vae_encoder]            |   Pattern 2: [controlNet\Unet]
 *    transpose input[0] is input[K] -> transpose   |     transpose input[0] is input[K] -> transpose
 *      matmul  input[0] is input[Q] ->   matmul    |       matmul  input[0] is input[Q] ->   matmul
 *                                         mul      |                                          mul
 *                                        cast      |                                        softMax
 *                                       softMax    |                                         cast
 *                                        cast      |       matmul  input[0] is input[V] ->  matmul
 *      matmul  input[0] is input[V] ->  matmul     |
 * --------------------------------------------------------------------------------------------------------
 *
 */
class FlashAttentionFusionForCustom : public MultiplePatternProcessPass {
 public:
  explicit FlashAttentionFusionForCustom(
    const std::vector<std::string> &plugin_custom_ops = {},
    const std::map<std::string, std::vector<std::string>> &enable_pattern_names = {},
    const std::map<std::string, std::vector<std::string>> &disable_pattern_names = {},
    const std::string &name = "FlashAttentionFusionForCustom", bool multigraph = true)
      : MultiplePatternProcessPass(name, multigraph) {
    plugin_custom_ops_ = plugin_custom_ops;
    enable_pattern_names_ = enable_pattern_names;
    disable_pattern_names_ = disable_pattern_names;
  }

  ~FlashAttentionFusionForCustom() override = default;

  std::unordered_map<std::string, VectorRef> DefinePatterns() const override;

  AnfNodePtr Process(const std::string &, const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  const VectorRef DefineFlashAttentionPattern1() const;
  const VectorRef DefineFlashAttentionPattern2() const;
  const VectorRef DefineFlashAttentionPattern3() const;
  bool InitVar() const;
  bool CheckBatchMatmulTranspose(const CNodePtr &batchmm_cnode, const bool expected_transpose_a,
                                 const bool expected_transpose_b) const;
  bool CheckInputShape(const CNodePtr &cnode, const uint32_t input_index, const uint32_t expected_rank_num,
                       const uint32_t expected_seq_len) const;
  CNodePtr CreateFlashAttentionNodePart1(const std::string &pattern_name, const FuncGraphPtr &func_graph,
                                         const AnfNodePtr &node, const EquivPtr &equiv) const;
  CNodePtr CreateFlashAttentionNodePart2(const std::string &pattern_name, const FuncGraphPtr &func_graph,
                                         const AnfNodePtr &node, const EquivPtr &equiv) const;
  bool CheckNeedFusion(std::vector<std::string> cnode_names) const;

 protected:
  mutable VarPtr input_0_batchmm_qk_ = nullptr;
  mutable VarPtr input_1_batchmm_qk_ = nullptr;
  mutable VarPtr input_1_batchmm_sv_ = nullptr;
  mutable VarPtr input_0_mul_ = nullptr;
  mutable bool has_add_cast_ = false;
  std::vector<std::string> plugin_custom_ops_;
  std::map<std::string, std::vector<std::string>> enable_pattern_names_;
  std::map<std::string, std::vector<std::string>> disable_pattern_names_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_FLASH_ATTENTION_FUSION_H_
