/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_ADAM_APPLY_ONE_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_ADAM_APPLY_ONE_FUSION_H_

#include <vector>
#include <memory>
#include <string>
#include "include/backend/optimizer/optimizer.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace opt {
constexpr size_t kAdamApplyOneInputVarNum = 5;
constexpr size_t kAdamApplyOneMulInputVarNum = 4;

class AdamApplyOneFusion : public PatternProcessPass {
 public:
  explicit AdamApplyOneFusion(const std::string &name = "adam_apply_one_fusion", bool multigraph = true)
      : PatternProcessPass(name, multigraph) {
    for (size_t i = 0; i < kAdamApplyOneInputVarNum; ++i) {
      input_vars_.push_back(std::make_shared<Var>());
    }
    for (size_t i = 0; i < kAdamApplyOneMulInputVarNum; ++i) {
      mul_x_input_vars_.push_back(std::make_shared<Var>());
    }
    add2_y_ = std::make_shared<Var>();
    add0_var_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimAdd->name()));
    add1_var_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimAdd->name()));
    sub0_var_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimSub->name()));
  }

  ~AdamApplyOneFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                           const EquivPtr &equiv) const override;

 protected:
  AnfNodePtr CreateAdamApplyOneNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                    const AnfNodePtr &final_node) const;
  std::vector<VarPtr> input_vars_;
  std::vector<VarPtr> mul_x_input_vars_;
  VarPtr add2_y_;
  VarPtr add0_var_;
  VarPtr add1_var_;
  VarPtr sub0_var_;
};

class AdamApplyOneCond1Fusion : public AdamApplyOneFusion {
 public:
  explicit AdamApplyOneCond1Fusion(bool multigraph = true)
      : AdamApplyOneFusion("adam_apply_one_cond1_fusion", multigraph) {}

  ~AdamApplyOneCond1Fusion() override = default;
  const BaseRef DefinePattern() const override;
};

class AdamApplyOneCond2Fusion : public AdamApplyOneFusion {
 public:
  explicit AdamApplyOneCond2Fusion(bool multigraph = true)
      : AdamApplyOneFusion("adam_apply_one_cond2_fusion", multigraph) {}

  ~AdamApplyOneCond2Fusion() override = default;
  const BaseRef DefinePattern() const override;
};

class AdamApplyOneCond3Fusion : public AdamApplyOneFusion {
 public:
  explicit AdamApplyOneCond3Fusion(bool multigraph = true)
      : AdamApplyOneFusion("adam_apply_one_cond3_fusion", multigraph) {}

  ~AdamApplyOneCond3Fusion() override = default;
  const BaseRef DefinePattern() const override;
};

class AdamApplyOneCond4Fusion : public AdamApplyOneFusion {
 public:
  explicit AdamApplyOneCond4Fusion(bool multigraph = true)
      : AdamApplyOneFusion("adam_apply_one_cond4_fusion", multigraph) {}

  ~AdamApplyOneCond4Fusion() override = default;
  const BaseRef DefinePattern() const override;
};

class AdamApplyOneAssignFusion : public AdamApplyOneFusion {
 public:
  explicit AdamApplyOneAssignFusion(bool multigraph = true)
      : AdamApplyOneFusion("adam_apply_one_assign_fusion", multigraph) {}

  ~AdamApplyOneAssignFusion() override = default;
  const BaseRef DefinePattern() const override;
};

class AdamApplyOneAssignCond1Fusion : public AdamApplyOneFusion {
 public:
  explicit AdamApplyOneAssignCond1Fusion(bool multigraph = true)
      : AdamApplyOneFusion("adam_apply_one_assign_cond1_fusion", multigraph) {}

  ~AdamApplyOneAssignCond1Fusion() override = default;
  const BaseRef DefinePattern() const override;
};

class AdamApplyOneAssignCond2Fusion : public AdamApplyOneFusion {
 public:
  explicit AdamApplyOneAssignCond2Fusion(bool multigraph = true)
      : AdamApplyOneFusion("adam_apply_one_assign_cond2_fusion", multigraph) {}

  ~AdamApplyOneAssignCond2Fusion() override = default;
  const BaseRef DefinePattern() const override;
};

class AdamApplyOneAssignCond3Fusion : public AdamApplyOneFusion {
 public:
  explicit AdamApplyOneAssignCond3Fusion(bool multigraph = true)
      : AdamApplyOneFusion("adam_apply_one_assign_cond3_fusion", multigraph) {}

  ~AdamApplyOneAssignCond3Fusion() override = default;
  const BaseRef DefinePattern() const override;
};

class AdamApplyOneAssignCond4Fusion : public AdamApplyOneFusion {
 public:
  explicit AdamApplyOneAssignCond4Fusion(bool multigraph = true)
      : AdamApplyOneFusion("adam_apply_one_assign_cond4_fusion", multigraph) {}

  ~AdamApplyOneAssignCond4Fusion() override = default;
  const BaseRef DefinePattern() const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_ADAM_APPLY_ONE_FUSION_H_
