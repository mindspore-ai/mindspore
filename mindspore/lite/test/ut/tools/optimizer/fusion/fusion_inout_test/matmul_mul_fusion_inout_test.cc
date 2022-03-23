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
#include "tools/optimizer/fusion/matmul_mul_fusion.h"
#include "test/ut/tools/optimizer/fusion/fusion_inout_test/conv_fusion_inout_test.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "ops/fusion/mul_fusion.h"
#include "ops/fusion/mat_mul_fusion.h"

namespace mindspore {
class MatmulMulFusionInoutTest : public FusionInoutTest {
 public:
  MatmulMulFusionInoutTest() = default;

 protected:
  void InitPass() override { this->pass_ = std::make_shared<opt::MatMulMulFusion>(); }

  void InitGraph() override {
    this->graph_ = std::make_shared<FuncGraph>();
    MS_CHECK_TRUE_MSG(graph_ != nullptr, , "Create FuncGraph failed");
    auto matmul_node = AddMatmul(graph_, "matmul_node");
    if (matmul_node == nullptr) {
      this->graph_ = nullptr;
      return;
    }
    auto mul_node = AddMul(graph_, matmul_node, "mul_node");
    if (mul_node == nullptr) {
      this->graph_ = nullptr;
      return;
    }

    auto ret = AddReturn(graph_, {mul_node});
    if (ret == nullptr) {
      this->graph_ = nullptr;
      return;
    }
  }

 private:
  CNodePtr AddMul(const FuncGraphPtr &graph, const AnfNodePtr &input, const std::string &name) {
    AnfNodePtr param = AddParameter(graph_, 0, {in_}, kNumberTypeFloat32, name);
    auto prim = std::make_unique<ops::MulFusion>();
    MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "create MulFusion primitivec failed");
    auto prim_c = prim->GetPrim();
    MS_CHECK_TRUE_MSG(prim_c != nullptr, nullptr, "prim_c is nullptr");
    prim->Init(ActivationType::NO_ACTIVATION);
    auto add_primitive = NewValueNode(prim_c);
    MS_CHECK_TRUE_RET(add_primitive != nullptr, nullptr);
    auto add_fusion = graph->NewCNode({add_primitive, input, param});
    MS_CHECK_TRUE_MSG(add_fusion != nullptr, nullptr, "create AddFusion failed");
    add_fusion->set_fullname_with_scope(name);
    return add_fusion;
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
    auto add_primitive = NewValueNode(prim_c);
    MS_CHECK_TRUE_RET(add_primitive != nullptr, nullptr);
    auto add_fusion = graph->NewCNode({add_primitive, input1, input2, bias});
    MS_CHECK_TRUE_MSG(add_fusion != nullptr, nullptr, "create AddFusion failed");
    add_fusion->set_fullname_with_scope(name);
    return add_fusion;
  }

 private:
  int64_t im_ = 128;
  int64_t ik_ = 256;
  int64_t in_ = 96;
};

TEST_F(MatmulMulFusionInoutTest, test) { ASSERT_EQ(DoTest(), true); }
}  // namespace mindspore
