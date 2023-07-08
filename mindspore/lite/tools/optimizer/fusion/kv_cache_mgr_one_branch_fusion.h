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
#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_KV_CACHE_MGR_ONE_BRANCH_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_KV_CACHE_MGR_ONE_BRANCH_FUSION_H_

#include <string>
#include "tools/optimizer/common/pattern_process_pass_extends.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace opt {
class KVCacheMgrOneBranchFusion : public LitePatternProcessPass {
 public:
  explicit KVCacheMgrOneBranchFusion(bool multigraph = true, const std::string &name = "KVCacheMgrOneBranchFusion")
      : LitePatternProcessPass(name, multigraph) {}
  ~KVCacheMgrOneBranchFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  bool InitVar() const;
  CNodePtr CreateConcatNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &equiv) const;
  tensor::TensorPtr ConstData(int32_t padding_length) const;

 protected:
  mutable VarPtr input_0_batch_valid_length_ = nullptr;
  mutable VarPtr input_1_key_ = nullptr;
  mutable VarPtr input_2_key_past_ = nullptr;
};
}  // namespace opt
}  // namespace mindspore

#endif
