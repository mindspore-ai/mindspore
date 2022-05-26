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
#include "tools/optimizer/fusion/matmul_activation_fusion.h"
#include "test/ut/tools/optimizer/fusion/fusion_inout_test/fusion_inout_test.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "ops/fusion/mat_mul_fusion.h"
#include "ops/fusion/activation.h"

namespace mindspore {
class MatMulActivationFusionInoutTest : public FusionInoutTest {
 public:
  MatMulActivationFusionInoutTest() = default;

 protected:
  void InitPass() override { this->pass_ = std::make_shared<opt::MatMulActivationFusion>(nullptr); }

  void InitGraph() override {
    this->graph_ = std::make_shared<FuncGraph>();
    MS_CHECK_TRUE_MSG(graph_ != nullptr, , "Create FuncGraph failed");
    auto matmul_node = AddMatmul(graph_, "matmul_node");
    if (matmul_node == nullptr) {
      this->graph_ = nullptr;
      return;
    }
    auto act_node = AddAct(graph_, matmul_node, "act_node");
    if (act_node == nullptr) {
      this->graph_ = nullptr;
      return;
    }

    auto ret = AddReturn(graph_, {act_node});
    if (ret == nullptr) {
      this->graph_ = nullptr;
      return;
    }
  }

 private:
  CNodePtr AddAct(const FuncGraphPtr &graph, const AnfNodePtr &input, const std::string &name) {
    auto prim = std::make_unique<ops::Activation>();
    MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "create Act primitivec failed");
    auto prim_c = prim->GetPrim();
    MS_CHECK_TRUE_MSG(prim_c != nullptr, nullptr, "prim_c is nullptr");
    prim->Init();
    prim->set_activation_type(ActivationType::RELU);
    auto act_primitive = NewValueNode(prim_c);
    MS_CHECK_TRUE_RET(act_primitive != nullptr, nullptr);
    auto act = graph->NewCNode({act_primitive, input});
    MS_CHECK_TRUE_MSG(act != nullptr, nullptr, "create Act failed");
    act->set_fullname_with_scope(name);
    return act;
  }
  CNodePtr AddMatmul(const FuncGraphPtr &graph, const std::string &name) {
    AnfNodePtr input1 = AddParameter(graph_, 0, {im_, ik_}, kNumberTypeFloat32, "graph_" + name + "_input1");
    AnfNodePtr input2 = AddParameter(graph_, 0, {ik_, in_}, kNumberTypeFloat32, "graph_" + name + "_input2");
    AnfNodePtr bias = AddParameter(graph_, 0, {in_}, kNumberTypeFloat32, "graph_" + name + "_input3");
    auto prim = std::make_unique<ops::MatMulFusion>();
    MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "create MatMul primitivec failed");
    auto prim_c = prim->GetPrim();
    MS_CHECK_TRUE_MSG(prim_c != nullptr, nullptr, "prim_c is nullptr");
    prim->Init(false, false, ActivationType::NO_ACTIVATION);
    auto matmul_primitive = NewValueNode(prim_c);
    MS_CHECK_TRUE_RET(matmul_primitive != nullptr, nullptr);
    auto matmul_fusion = graph->NewCNode({matmul_primitive, input1, input2, bias});
    MS_CHECK_TRUE_MSG(matmul_fusion != nullptr, nullptr, "create matmul fusion failed");
    matmul_fusion->set_fullname_with_scope(name);
    return matmul_fusion;
  }

 private:
  int64_t im_ = 128;
  int64_t ik_ = 256;
  int64_t in_ = 96;
};

TEST_F(MatMulActivationFusionInoutTest, test) { ASSERT_EQ(DoTest(), true); }
}  // namespace mindspore
