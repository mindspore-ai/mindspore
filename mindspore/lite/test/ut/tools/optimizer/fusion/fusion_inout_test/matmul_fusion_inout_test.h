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

#ifndef MINDSPORE_LITE_TEST_UT_TOOLS_OPTIMIZER_FUSION_FUSION_INOUT_TEST_MATMUL_FUSION_INOUT_TEST_H_
#define MINDSPORE_LITE_TEST_UT_TOOLS_OPTIMIZER_FUSION_FUSION_INOUT_TEST_MATMUL_FUSION_INOUT_TEST_H_

#include <string>
#include "test/ut/tools/optimizer/fusion/fusion_inout_test/fusion_inout_test.h"
#include "ir/anf.h"
#include "nnacl/matmul_parameter.h"
#include "include/backend/optimizer/pass.h"
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/pass_manager.h"
#include "ops/fusion/activation.h"

namespace mindspore {
class MatMulFusionInoutTest : public FusionInoutTest {
 public:
  MatMulFusionInoutTest() = default;

 protected:
  CNodePtr AddMatMul(const FuncGraphPtr &graph, const AnfNodePtr &input1, const AnfNodePtr &input2,
                     const ActivationType &act_type, const std::string &name);
};
}  // namespace mindspore
#endif
