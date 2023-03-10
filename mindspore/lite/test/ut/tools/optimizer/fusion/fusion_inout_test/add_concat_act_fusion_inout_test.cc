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
#include <string>
#include <memory>
#include "test/ut/tools/optimizer/fusion/fusion_inout_test/fusion_inout_test.h"
#include "ir/anf.h"
#include "include/backend/optimizer/pass.h"
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/pass_manager.h"
#include "tools/optimizer/fusion/add_concat_activation_fusion.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "ops/fusion/activation.h"
#include "ops/concat.h"
#include "ops/fusion/add_fusion.h"

namespace mindspore {
namespace {
constexpr size_t kAddInputTensorWSize = 128;
}  // namespace
class ConcatActFusionInoutTest : public FusionInoutTest {
 public:
  ConcatActFusionInoutTest() = default;

 protected:
  void InitPass() override { this->pass_ = std::make_shared<opt::AddConcatActivationFusion>(); }

  void InitGraph() override {
    this->graph_ = std::make_shared<FuncGraph>();
    MS_CHECK_TRUE_MSG(graph_ != nullptr, , "Create FuncGraph failed");
    auto left_add_node = AddAdd(graph_, "left_add_node");
    if (left_add_node == nullptr) {
      this->graph_ = nullptr;
      return;
    }
    auto right_add_node = AddAdd(graph_, "right_add_node");
    if (right_add_node == nullptr) {
      this->graph_ = nullptr;
      return;
    }
    auto concat_node = AddConcat(graph_, left_add_node, right_add_node, "concat");
    if (concat_node == nullptr) {
      this->graph_ = nullptr;
      return;
    }

    auto act = AddAct(graph_, concat_node, "concat_act");
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
  CNodePtr AddAdd(const FuncGraphPtr &graph, const std::string &name) {
    AnfNodePtr input1 =
      AddParameter(graph_, 0, {add_left_h_, add_left_w_}, kNumberTypeFloat32, "graph_" + name + "_input1");
    AnfNodePtr input2 =
      AddParameter(graph_, 0, {add_left_h_, add_left_w_}, kNumberTypeFloat32, "graph_" + name + "_input2");
    auto prim = std::make_unique<ops::AddFusion>();
    MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "create AddFusion primitivec failed");
    auto prim_c = prim->GetPrim();
    MS_CHECK_TRUE_MSG(prim_c != nullptr, nullptr, "prim_c is nullptr");
    prim->Init(ActivationType::NO_ACTIVATION);
    auto add_primitive = NewValueNode(prim_c);
    MS_CHECK_TRUE_RET(add_primitive != nullptr, nullptr);
    auto add_fusion = graph->NewCNode({add_primitive, input1, input2});
    MS_CHECK_TRUE_MSG(add_fusion != nullptr, nullptr, "create AddFusion failed");
    add_fusion->set_fullname_with_scope(name);
    return add_fusion;
  }

  CNodePtr AddConcat(const FuncGraphPtr &graph, const AnfNodePtr &input1, const AnfNodePtr &input2,
                     const std::string &name) {
    auto concat_primitive = std::make_unique<ops::Concat>();
    MS_CHECK_TRUE_MSG(concat_primitive != nullptr, nullptr, "create concat primitivec failed");
    auto prim_c = concat_primitive->GetPrim();
    MS_CHECK_TRUE_MSG(prim_c != nullptr, nullptr, "prim_c is nullptr");
    concat_primitive->Init();
    concat_primitive->set_axis(1);
    auto concat_primc = NewValueNode(prim_c);
    MS_CHECK_TRUE_RET(concat_primc != nullptr, nullptr);
    auto concat = graph->NewCNode({concat_primc, input1, input2});
    MS_CHECK_TRUE_MSG(concat != nullptr, nullptr, "create Concat failed");
    concat->set_fullname_with_scope(name);
    return concat;
  }

  CNodePtr AddAct(const FuncGraphPtr &graph, const AnfNodePtr &input, const std::string &name) {
    auto prim = std::make_unique<ops::Activation>();
    MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "create Act primitivec failed");
    auto prim_c = prim->GetPrim();
    MS_CHECK_TRUE_MSG(prim_c != nullptr, nullptr, "prim_c is nullptr");
    prim->Init();
    prim->set_activation_type(ActivationType::RELU6);
    auto act_primitive = NewValueNode(prim_c);
    MS_CHECK_TRUE_RET(act_primitive != nullptr, nullptr);
    auto act = graph->NewCNode({act_primitive, input});
    MS_CHECK_TRUE_MSG(act != nullptr, nullptr, "create Act failed");
    act->set_fullname_with_scope(name);
    return act;
  }

 private:
  int add_left_h_ = 1;
  int add_left_w_ = kAddInputTensorWSize;
};

TEST_F(ConcatActFusionInoutTest, test) { ASSERT_EQ(DoTest(), true); }
}  // namespace mindspore
