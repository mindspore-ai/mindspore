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

#include <memory>
#include "tools/optimizer/fusion/conv_activation_fusion.h"
#include "test/ut/tools/optimizer/fusion/fusion_inout_test/conv_fusion_inout_test.h"
#include "backend/kernel_compiler/cpu/nnacl/op_base.h"
#include "ops/fusion/activation.h"

namespace mindspore {
class ConvActFusionInoutTest : public ConvFusionInoutTest {
 public:
  ConvActFusionInoutTest() = default;

 protected:
  void InitPass() override { this->pass_ = std::make_shared<opt::ConvActivationFusion>(); }

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
    auto act = AddAct(graph_, conv, "conv_act");
    if (act == nullptr) {
      this->graph_ = nullptr;
      return;
    }
    auto ret = AddReturn(graph_, {act});
    if (ret == nullptr) {
      this->graph_ = nullptr;
      return;
    }
  }

 private:
  static CNodePtr AddAct(const FuncGraphPtr &graph, const AnfNodePtr &input, const std::string &name) {
    auto prim = std::make_unique<ops::Activation>();
    MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "create Act primitivec failed");
    prim->Init();
    prim->set_activation_type(ActivationType::RELU);
    auto act_primitive = NewValueNode(std::shared_ptr<ops::PrimitiveC>(prim.release()));
    MS_CHECK_TRUE_RET(act_primitive != nullptr, nullptr);
    auto act = graph->NewCNode({act_primitive, input});
    MS_CHECK_TRUE_MSG(act != nullptr, nullptr, "create Act failed");
    act->set_fullname_with_scope(name);
    return act;
  }
};

TEST_F(ConvActFusionInoutTest, test) { ASSERT_EQ(DoTest(), true); }
}  // namespace mindspore
