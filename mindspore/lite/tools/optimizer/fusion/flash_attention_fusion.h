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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_FLASH_ATTENTION_BASE_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_FLASH_ATTENTION_BASE_FUSION_H_

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include "tools/optimizer/common/multiple_pattern_process_pass.h"
namespace mindspore {
namespace opt {
/*
 *
 * --------------------------------------------------------------------------------------------------------
 *  Pattern 1:                                      |   Pattern 2:
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
class FlashAttentionFusion : public MultiplePatternProcessPass {
 public:
  explicit FlashAttentionFusion(const std::string &name = "FlashAttentionFusion", bool multigraph = true)
      : MultiplePatternProcessPass(name, multigraph) {}

  ~FlashAttentionFusion() override = default;

  std::unordered_map<std::string, VectorRef> DefinePatterns() const override;

  AnfNodePtr Process(const std::string &, const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  CNodePtr CreatePromptFlashAttentionCnodeForBNSD(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                  const AnfNodePtr &q, const AnfNodePtr &k, const AnfNodePtr &v,
                                                  const AnfNodePtr &atten_mask, int64_t num_heads, int64_t next_token,
                                                  float scale_value, int64_t num_key_value_heads = 1) const;

  CNodePtr CreatePromptFlashAttentionCnodeForBSH(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                 const AnfNodePtr &q, const AnfNodePtr &k, const AnfNodePtr &v,
                                                 const AnfNodePtr &atten_mask, int64_t num_heads, int64_t next_token,
                                                 float scale_value) const;

  CNodePtr CreateIncreFlashAttentionCnodeForBNSD(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                 const AnfNodePtr &q, const AnfNodePtr &k, const AnfNodePtr &v,
                                                 const AnfNodePtr &atten_mask, int64_t num_heads, int64_t next_token,
                                                 float scale_value) const;

  CNodePtr CreateFlashAttentionNodeForSD(const std::string &pattern_name, const FuncGraphPtr &func_graph,
                                         const AnfNodePtr &node, const EquivPtr &equiv) const;
  CNodePtr CreateFlashAttentionNodeForPg(const std::string &pattern_name, const FuncGraphPtr &func_graph,
                                         const AnfNodePtr &node, const EquivPtr &equiv) const;
  CNodePtr CreateFlashAttentionNodeForLLAMAPatternV1(const std::string &pattern_name, const FuncGraphPtr &func_graph,
                                                     const AnfNodePtr &node, const EquivPtr &equiv) const;
  CNodePtr CreateFlashAttentionNodeForLLAMAPatternV2(const std::string &pattern_name, const FuncGraphPtr &func_graph,
                                                     const AnfNodePtr &node, const EquivPtr &equiv) const;
  CNodePtr CreateFlashAttentionNodeForBaiChuanPattern(const std::string &pattern_name, const FuncGraphPtr &func_graph,
                                                      const AnfNodePtr &node, const EquivPtr &equiv) const;

  CNodePtr CreatePadCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, int32_t pad_size) const;
  CNodePtr CreateSliceCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, int32_t slice_size) const;
  CNodePtr GetSDDynamicShapeParam(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const;
  float GetScaleValueForDynamicShape(const AnfNodePtr &mul_const_input) const;
  CNodePtr CreateFAForSD15(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const AnfNodePtr &q_trans,
                           const AnfNodePtr &k_trans, const AnfNodePtr &v_trans, int64_t num_head, int64_t next_token,
                           float scale_value) const;
  CNodePtr CreateGQACNodeForBNSD(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const CNodePtr &matmul_1,
                                 const CNodePtr &matmul_2, const CNodePtr &attention_mask_mul) const;
  CNodePtr CreateFAForBNSDWithAttenMask(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                        const CNodePtr &qk_matmul, const CNodePtr &v_matmul,
                                        const CNodePtr &attention_mask_mul) const;

  const VectorRef DefineFlashAttentionPatternForSDBNSD() const;
  const VectorRef DefineFlashAttentionPatternForSDBSH() const;
  const VectorRef DefineFlashAttentionPatternForPg() const;
  const VectorRef DefineFlashAttentionPatternForLLAMAPatternV1() const;
  const VectorRef DefineFlashAttentionPatternForLLAMAPatternV2() const;
  const VectorRef DefineFlashAttentionPatternForBaiChuan() const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_FLASH_ATTENTION_BASE_FUSION_H_
