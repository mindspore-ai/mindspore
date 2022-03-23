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

#define USE_DEPRECATED_API
#include <memory>
#include "tools/optimizer/fusion/transpose_matmul_fusion.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "test/ut/tools/optimizer/fusion/fusion_inout_test/matmul_fusion_inout_test.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "ops/transpose.h"
#include "ops/mat_mul.h"

namespace mindspore {
namespace {
inline const int kHeight = 5;
inline const int kChannel = 3;
inline const std::vector<int> kMatMulTransPerm = {0, 2, 1};
}  // namespace

class TransMatMulFusionInoutTest : public MatMulFusionInoutTest {
 public:
  TransMatMulFusionInoutTest() = default;

 protected:
  void InitPass() override { this->pass_ = std::make_shared<opt::TransposeMatMulFusion>(); }

  void InitGraph() override {
    this->graph_ = std::make_shared<FuncGraph>();
    MS_CHECK_TRUE_MSG(graph_ != nullptr, , "Create FuncGraph failed");
    auto input_1 = AddParameter(graph_, 0, {1, kHeight, kChannel}, kNumberTypeFloat32, "graph_input_1");
    if (input_1 == nullptr) {
      this->graph_ = nullptr;
      return;
    }
    auto input_2 = AddParameter(graph_, 0, {1, kChannel, kHeight}, kNumberTypeFloat32, "graph_input_2");
    if (input_2 == nullptr) {
      this->graph_ = nullptr;
      return;
    }
    auto trans_a = AddTranspose(graph_, input_1, kMatMulTransPerm, "trans_a");
    if (trans_a == nullptr) {
      this->graph_ = nullptr;
      return;
    }
    auto trans_b = AddTranspose(graph_, input_2, kMatMulTransPerm, "trans_b");
    if (trans_b == nullptr) {
      this->graph_ = nullptr;
      return;
    }
    auto matmul = AddMatMul(graph_, input_1, input_2, ActivationType::NO_ACTIVATION, "matmul");
    if (matmul == nullptr) {
      this->graph_ = nullptr;
      return;
    }
    auto ret = AddReturn(graph_, {matmul});
    if (ret == nullptr) {
      this->graph_ = nullptr;
      return;
    }
  }

 private:
  static CNodePtr AddTranspose(const FuncGraphPtr &graph, const AnfNodePtr &input, const std::vector<int> &perm_val,
                               const std::string &name) {
    auto prim = std::make_unique<ops::Transpose>();
    MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "create Act primitivec failed");
    auto prim_c = prim->GetPrim();
    MS_CHECK_TRUE_MSG(prim_c != nullptr, nullptr, "prim_c is nullptr");
    prim->Init();
    auto trans_primitive = NewValueNode(prim_c);
    auto perm = opt::BuildIntVecParameterNode(graph, perm_val, name + "_perm");
    auto transpose = graph->NewCNode({trans_primitive, perm});
    MS_CHECK_TRUE_MSG(transpose != nullptr, nullptr, "create Transpose failed");
    transpose->set_fullname_with_scope(name);
    return transpose;
  }
};

TEST_F(TransMatMulFusionInoutTest, test) { ASSERT_EQ(DoTest(), true); }
}  // namespace mindspore
