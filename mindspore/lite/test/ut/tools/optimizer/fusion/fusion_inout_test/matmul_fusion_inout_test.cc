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

#include "test/ut/tools/optimizer/fusion/fusion_inout_test/matmul_fusion_inout_test.h"
#include <memory>
#include "src/common/log_adapter.h"
#include "ir/func_graph.h"
#include "ops/fusion/mat_mul_fusion.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"

namespace mindspore {
CNodePtr MatMulFusionInoutTest::AddMatMul(const FuncGraphPtr &graph, const AnfNodePtr &input1, const AnfNodePtr &input2,
                                          const ActivationType &act_type, const std::string &name) {
  auto prim = std::make_unique<ops::MatMulFusion>();
  MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "create MatMul primitivec failed");
  prim->Init(false, false, act_type);
  auto matmul_primitive = NewValueNode(std::shared_ptr<ops::PrimitiveC>(prim.release()));
  auto matmul = graph->NewCNode({matmul_primitive, input1, input2});
  MS_CHECK_TRUE_MSG(matmul != nullptr, nullptr, "create MatMul failed");
  matmul->set_fullname_with_scope(name);
  return matmul;
}
}  // namespace mindspore
