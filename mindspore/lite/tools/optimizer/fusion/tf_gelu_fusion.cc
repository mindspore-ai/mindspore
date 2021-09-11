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

#include "tools/optimizer/fusion/tf_gelu_fusion.h"
#include "ops/op_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr float DIFF_THRESHOLD = 0.0001;
constexpr float POW_Y = 3;
constexpr float MUL1_Y = 0.044715;
constexpr float MUL2_X = 0.79788;
constexpr float ADD2_X = 1.0;
constexpr float MUL3_X = 0.5;
bool CheckTanh(const EquivPtr &equiv, const VarPtr &input) {
  MS_ASSERT(equiv != nullptr && input != nullptr);
  auto anf_node = utils::cast<AnfNodePtr>((*equiv)[input]);
  MS_ASSERT(anf_node != nullptr);
  AnfNodePtr value_node = anf_node;
  if (anf_node->isa<CNode>()) {
    value_node = anf_node->cast<CNodePtr>()->input(0);
  }
  auto act_prim = GetValueNode<PrimitivePtr>(value_node);
  if (act_prim == nullptr) {
    return false;
  }
  return act_prim->GetAttr(ops::kActivationType) != nullptr &&
         GetValue<int64_t>(act_prim->GetAttr(ops::kActivationType)) == mindspore::TANH;
}
}  // namespace

bool TfGeLUFusion::Init() const {
  if (!GeLUFusion::Init()) {
    MS_LOG(ERROR) << "basic class initial member failed.";
    return false;
  }
  power_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(power_ != nullptr, false);
  power_y_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(power_y_ != nullptr, false);
  mul1_x_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(mul1_x_ != nullptr, false);
  mul2_x_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(mul2_x_ != nullptr, false);
  tanh_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(tanh_ != nullptr, false);
  add2_x_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(add2_x_ != nullptr, false);
  mul3_x_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(mul3_x_ != nullptr, false);
  return true;
}

// gelu(x) = 1/2 * x * [1 + tanh(0.79788 * (x + 0.044715 * x ^ 3))]
const BaseRef TfGeLUFusion::DefinePattern() const {
  if (!Init()) {
    MS_LOG(ERROR) << "initial member failed.";
    return {};
  }
  VectorRef pow_ref({power_, input_, power_y_});
  auto is_mul1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul1 != nullptr, {});
  VectorRef mul1_ref({is_mul1, mul1_x_, pow_ref});
  auto is_add1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add1 != nullptr, {});
  VectorRef add1_ref({is_add1, input_, mul1_ref});
  auto is_mul2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul2 != nullptr, {});
  VectorRef mul2_ref({is_mul2, mul2_x_, add1_ref});
  VectorRef tanh_ref({tanh_, mul2_ref});
  auto is_add2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add2 != nullptr, {});
  VectorRef add2_ref({is_add2, add2_x_, tanh_ref});
  auto is_mul3 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul3 != nullptr, {});
  VectorRef mul3_ref({is_mul3, mul3_x_, add2_ref});
  auto is_mul4 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul4 != nullptr, {});
  VectorRef mul4_ref({is_mul4, input_, mul3_ref});
  return mul4_ref;
}

bool TfGeLUFusion::CheckPattern(const EquivPtr &equiv) const {
  MS_ASSERT(equiv != nullptr);
  if (!CheckTanh(equiv, tanh_)) {
    return false;
  }
  float pow_y = GetParameterValue(equiv, power_y_);
  if (pow_y < 0 || fabs(pow_y - POW_Y) > DIFF_THRESHOLD) {
    return false;
  }
  float mul1_y = GetParameterValue(equiv, mul1_x_);
  if (mul1_y < 0 || fabs(mul1_y - MUL1_Y) > DIFF_THRESHOLD) {
    return false;
  }
  float mul2_x = GetParameterValue(equiv, mul2_x_);
  if (mul2_x < 0 || fabs(mul2_x - MUL2_X) > DIFF_THRESHOLD) {
    return false;
  }
  float add2_x = GetParameterValue(equiv, add2_x_);
  if (add2_x < 0 || fabs(add2_x - ADD2_X) > DIFF_THRESHOLD) {
    return false;
  }
  float mul3_x = GetParameterValue(equiv, mul3_x_);
  if (mul3_x < 0 || fabs(mul3_x - MUL3_X) > DIFF_THRESHOLD) {
    return false;
  }
  approximate_ = true;
  return true;
}
}  // namespace opt
}  // namespace mindspore
