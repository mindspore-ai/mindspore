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
#include "test/ut/tools/optimizer/fusion/fusion_inout_test/conv_fusion_inout_test.h"
#include <memory>
#include "src/common/log_adapter.h"
#include "ir/func_graph.h"
#include "ops/fusion/conv2d_fusion.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"

namespace mindspore {
ValueNodePtr ConvFusionInoutTest::CreateConvPrimitiveValue() {
  auto prim = std::make_unique<ops::Conv2DFusion>();
  MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "create Conv2d primitivec failed");
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_MSG(prim_c != nullptr, nullptr, "prim_c is nullptr");
  prim->Init(ic_, oc_, {kh_, kw_});
  prim->set_pad_mode(PadMode::SAME);
  return NewValueNode(prim_c);
}

CNodePtr ConvFusionInoutTest::AddConv(const FuncGraphPtr &graph, const AnfNodePtr &input, const std::string &name) {
  auto conv_primitive = CreateConvPrimitiveValue();
  MS_CHECK_TRUE_RET(conv_primitive != nullptr, nullptr);
  auto weight = AddParameter(graph, ic_ * oc_ * kh_ * kw_ * sizeof(float), {oc_, kh_, kw_, ic_}, kNumberTypeFloat32,
                             name + "_weight");
  auto bias = AddParameter(graph, oc_ * sizeof(float), {oc_}, kNumberTypeFloat32, name + "_bias");
  auto conv = graph->NewCNode({conv_primitive, input, weight, bias});
  MS_CHECK_TRUE_MSG(conv != nullptr, nullptr, "create Conv2d failed");
  conv->set_fullname_with_scope(name);
  return conv;
}
}  // namespace mindspore
