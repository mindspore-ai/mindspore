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
#include "tools/optimizer/fusion/conv_biasadd_fusion.h"
#include "test/ut/tools/optimizer/fusion/fusion_inout_test/conv_fusion_inout_test.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "ops/bias_add.h"

namespace mindspore {
class ConvBiasFusionInoutTest : public ConvFusionInoutTest {
 public:
  ConvBiasFusionInoutTest() = default;

 protected:
  void InitPass() override { this->pass_ = std::make_shared<opt::ConvBiasaddFusion>(); }

  void InitGraph() override {
    this->graph_ = std::make_shared<FuncGraph>();
    MS_CHECK_TRUE_MSG(graph_ != nullptr, , "Create FuncGraph failed");
    auto input = AddParameter(graph_, 0, {1, ih_, iw_, ic_}, kNumberTypeFloat32, "graph_input");
    if (input == nullptr) {
      this->graph_ = nullptr;
      return;
    }
    auto conv = AddConv(graph_, input, "conv");
    if (conv == nullptr) {
      this->graph_ = nullptr;
      return;
    }
    auto bias = AddBias(graph_, conv, "conv_bias");
    if (bias == nullptr) {
      this->graph_ = nullptr;
      return;
    }
    auto ret = AddReturn(graph_, {bias});
    if (ret == nullptr) {
      this->graph_ = nullptr;
      return;
    }
  }

 private:
  static CNodePtr AddBias(const FuncGraphPtr &graph, const AnfNodePtr &input, const std::string &name) {
    auto prim = std::make_unique<ops::BiasAdd>();
    MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "create BiasAdd primitivec failed");
    auto prim_c = prim->GetPrim();
    MS_CHECK_TRUE_MSG(prim_c != nullptr, nullptr, "prim_c is nullptr");
    prim->Init();
    auto bias_primitive = NewValueNode(prim_c);
    MS_CHECK_TRUE_RET(bias_primitive != nullptr, nullptr);
    auto bias = AddParameter(graph, oc_ * sizeof(float), {oc_}, kNumberTypeFloat32, name + "_bias");
    auto bias_add = graph->NewCNode({bias_primitive, input, bias});
    MS_CHECK_TRUE_MSG(bias_add != nullptr, nullptr, "create BiasAdd failed");
    bias_add->set_fullname_with_scope(name);
    return bias_add;
  }
};

TEST_F(ConvBiasFusionInoutTest, test) { ASSERT_EQ(DoTest(), true); }
}  // namespace mindspore
