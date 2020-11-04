/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <iostream>
#include <unordered_map>
#include <string>

#include "common/common_test.h"

#include "transform/graph_ir/op_adapter.h"
#include "transform/graph_ir/op_declare/array_ops_declare.h"
#include "frontend/operator/ops.h"

using std::cout;
using std::endl;
using std::string;
using std::unordered_map;

namespace mindspore {
namespace transform {
class TestOpAdapter : public UT::Common {
 public:
  TestOpAdapter() {}
};

#if (!defined ENABLE_GE)
#if 0
// fix conv2d ut
TEST_F(TestOpAdapter, TestSpecilization_Conv2D) {
    BaseOpAdapter *adpt = new OpAdapter<Conv2D>();

    auto input = std::make_shared<ge::Operator>();
    auto conv = std::make_shared<Conv2D>();

    ASSERT_EQ(adpt->setInput(conv, 1, input), 0);
    ASSERT_EQ(adpt->setInput(conv, 2, input), 0);
    ASSERT_EQ(adpt->setInput(conv, 3, input), NOT_FOUND);

    ASSERT_EQ(0, adpt->setAttr(conv, "group", 1));
    ASSERT_EQ(0, adpt->setAttr(conv, "mode", 1));

    delete adpt;
}
#endif
TEST_F(TestOpAdapter, TestSpecilization_Const) {
  BaseOpAdapter *adpt = new OpAdapter<Const>();
  auto valuenode = std::make_shared<Const>();
  auto input = std::make_shared<Const>();

  ASSERT_EQ(adpt->setInput(valuenode, 1, input), NOT_FOUND);
  delete adpt;
}
#if 0
// fix conv2d ut
TEST_F(TestOpAdapter, TestSetAttr_Conv2d_Primitive) {
    BaseOpAdapter *adpt = new OpAdapter<Conv2D>();
    auto conv = std::make_shared<Conv2D>();

    ASSERT_EQ(adpt->setAttr(conv, "padding", 1), NOT_FOUND);
    ASSERT_EQ(adpt->setAttr(conv, "pad", 1), 0);
    ASSERT_EQ(adpt->setAttr(conv, "pad_mode", string("same")), 0);
    ASSERT_EQ(adpt->setAttr(conv, "nothing", "test"), NOT_FOUND);

    const unordered_map<std::string, ValuePtr> attrs = {
        {"padding", MakeValue(2)},
        {"padding_mode", MakeValue(string("normal"))},
        {"stride", MakeValue(8)}
    };

    auto prim = prim::kPrimConv2D;
    prim->SetAttrs({
        {"strides", MakeValue(3)},
        {"padding", MakeValue(1)},
    });
    ASSERT_EQ(prim->name(), prim::kPrimConv2D->name());

    Int64Imm strides(3);
    Int64Imm padding(1);
    ASSERT_EQ(*(prim->GetAttr("strides")), strides);
    ASSERT_EQ(*(prim->GetAttr("padding")), padding);

    ASSERT_EQ(adpt->setAttr(conv, prim), 0);

    delete adpt;
}
#endif
#endif
}  // namespace transform
}  // namespace mindspore
