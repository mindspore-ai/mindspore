/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/onnx_gelu_fusion.h"
#include <unordered_map>
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr int kStructureNum = 2;
constexpr float DIFF_THRESHOLD = 0.0001;
constexpr float DIV_Y = 1.41421;
constexpr float ADD_Y = 1.0;
constexpr float MUL1_y = 0.5;
}  // namespace

bool OnnxGeLUFusion::Init() const {
  inputs_.resize(kStructureNum);
  div_y_.resize(kStructureNum);
  add_y_.resize(kStructureNum);
  mul1_y_.resize(kStructureNum);
  for (int i = 0; i < kStructureNum; ++i) {
    inputs_[i] = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(inputs_[i] != nullptr, false);
    div_y_[i] = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(div_y_[i] != nullptr, false);
    add_y_[i] = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(add_y_[i] != nullptr, false);
    mul1_y_[i] = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(mul1_y_[i] != nullptr, false);
  }

  return true;
}

// gelu(x) = 1/2 * x * [1 + erf(x / sqrt(2))]
VectorRef OnnxGeLUFusion::DefineFirstStructurePattern() const {
  auto is_div = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimDivFusion>);
  MS_CHECK_TRUE_RET(is_div != nullptr, {});
  VectorRef div_ref({is_div, inputs_[kIndex0], div_y_[kIndex0]});
  auto is_erf = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimErf>);
  MS_CHECK_TRUE_RET(is_erf != nullptr, {});
  VectorRef erf_ref({is_erf, div_ref});
  auto is_add = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  VectorRef add_ref({is_add, erf_ref, add_y_[kIndex0]});
  auto is_mul1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul1 != nullptr, {});
  VectorRef mul1_ref({is_mul1, inputs_[kIndex0], mul1_y_[kIndex0]});
  auto is_mul2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul2 != nullptr, {});
  VectorRef mul2_ref({is_mul2, mul1_ref, add_ref});
  return mul2_ref;
}

VectorRef OnnxGeLUFusion::DefineSecondStructurePattern() const {
  auto is_div = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimDivFusion>);
  MS_CHECK_TRUE_RET(is_div != nullptr, {});
  VectorRef div_ref({is_div, inputs_[kIndex1], div_y_[kIndex1]});
  auto is_erf = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimErf>);
  MS_CHECK_TRUE_RET(is_erf != nullptr, {});
  VectorRef erf_ref({is_erf, div_ref});
  auto is_add = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  VectorRef add_ref({is_add, erf_ref, add_y_[kIndex1]});
  auto is_mul1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul1 != nullptr, {});
  VectorRef mul1_ref({is_mul1, inputs_[kIndex1], add_ref});
  auto is_mul2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul2 != nullptr, {});
  VectorRef mul2_ref({is_mul2, mul1_ref, mul1_y_[kIndex1]});
  return mul2_ref;
}

std::unordered_map<std::string, VectorRef> OnnxGeLUFusion::DefinePatterns() const {
  if (!Init()) {
    MS_LOG(ERROR) << "initial member failed.";
    return {};
  }
  std::unordered_map<std::string, VectorRef> patterns;
  patterns["FirstStructure"] = DefineFirstStructurePattern();
  patterns["SecondStructure"] = DefineSecondStructurePattern();
  return patterns;
}

bool OnnxGeLUFusion::CheckPattern(const std::string &pattern_name, const EquivPtr &equiv) const {
  MS_ASSERT(equiv != nullptr);
  int index = 0;
  if (pattern_name == "SecondStructure") {
    index = kIndex1;
  }
  input_ = inputs_[index];
  float div_y = GetParameterValue(equiv, div_y_[index]);
  if (div_y < 0 || fabs(div_y - DIV_Y) > DIFF_THRESHOLD) {
    return false;
  }
  float add_y = GetParameterValue(equiv, add_y_[index]);
  if (add_y < 0 || fabs(add_y - ADD_Y) > DIFF_THRESHOLD) {
    return false;
  }
  float mul1_y = GetParameterValue(equiv, mul1_y_[index]);
  if (mul1_y < 0 || fabs(mul1_y - MUL1_y) > DIFF_THRESHOLD) {
    return false;
  }
  approximate_ = false;
  return true;
}
}  // namespace opt
}  // namespace mindspore
