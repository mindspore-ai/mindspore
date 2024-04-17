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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_FLASH_ATTENTION_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_FLASH_ATTENTION_FUSION_H_

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class FlashAttentionFusion : public PatternProcessPass {
 public:
  explicit FlashAttentionFusion(const std::string &name = "", bool multigraph = true)
      : PatternProcessPass(name, multigraph) {}
  ~FlashAttentionFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &equiv) const override;

 protected:
  CNodePtr CreatePromptFlashAttentionCnodeForBNSD(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                  const AnfNodePtr &q, const AnfNodePtr &k, const AnfNodePtr &v,
                                                  const AnfNodePtr &atten_mask, const int64_t num_heads,
                                                  const int64_t next_token, const float scale_value,
                                                  const int64_t num_key_value_heads) const;

 private:
  virtual CNodePtr CreateFlashAttentionNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                            const EquivPtr &equiv) const = 0;
  virtual const VectorRef DefineFlashAttentionPattern() const = 0;
};

class FlashAttentionFusionV1 : public FlashAttentionFusion {
 public:
  explicit FlashAttentionFusionV1(bool multigraph = true)
      : FlashAttentionFusion("FlashAttentionFusionV1", multigraph) {}
  ~FlashAttentionFusionV1() override = default;

 private:
  CNodePtr CreateFlashAttentionNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                    const EquivPtr &equiv) const override;
  const VectorRef DefineFlashAttentionPattern() const override;
};

class FlashAttentionFusionV2 : public FlashAttentionFusion {
 public:
  explicit FlashAttentionFusionV2(bool multigraph = true)
      : FlashAttentionFusion("FlashAttentionFusionV2", multigraph) {}
  ~FlashAttentionFusionV2() override = default;

 private:
  CNodePtr CreateFlashAttentionNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                    const EquivPtr &equiv) const override;
  const VectorRef DefineFlashAttentionPattern() const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_FLASH_ATTENTION_FUSION_H_
