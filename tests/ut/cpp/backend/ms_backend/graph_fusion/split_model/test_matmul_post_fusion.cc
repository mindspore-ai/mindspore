/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include <vector>
#include <string>

#include "utils/ms_context.h"
#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "common/graph_optimizer_test_framework.h"
#include "backend/ms_backend/graph_fusion/model/graph_builder.h"
#include "backend/ms_backend/graph_fusion/adapter/split_model_ascend.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"

namespace mindspore::graphkernel::test {
class TestMatMulPostFusion : public GraphKernelCommonTestSuite {
 public:
  TestMatMulPostFusion() {}
  void SetUp() override {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    context->set_param<std::string>(MS_CTX_DEVICE_TARGET, kAscendDevice);

    std::map<std::string, std::string> jit_config;
    jit_config["graph_kernel_flags"] = "--kernel_generator=DVM";
    graphkernel::GraphKernelFlags::SaveJitConfig(jit_config);
  }
};

struct TestMatMulPostFusionNumParams {
  size_t post_num;
};

class TestMatMulPostFusionNum : public TestMatMulPostFusion,
                                public testing::WithParamInterface<TestMatMulPostFusionNumParams> {
 public:
  TestMatMulPostFusionNum() {}
};

struct TestMatMulReshapePostFusionParams {
  ShapeVector shape_a;
  ShapeVector shape_b;
  ShapeVector reshape_shape;
  ShapeVector shape_c;
  bool can_fusion;
};

class TestMatMulReshapePostFusion : public TestMatMulPostFusion,
                                    public testing::WithParamInterface<TestMatMulReshapePostFusionParams> {
 public:
  TestMatMulReshapePostFusion() {}
};

TEST_P(TestMatMulPostFusionNum, test_matmul_fusion_num) {
  // get params
  const auto &param = GetParam();
  inner::GraphBuilder gb;
  auto x = gb.Parameter({{1024, 1024}, kNumberTypeFloat16, "DefaultFormat"});
  auto y = gb.Parameter({{1024, 1024}, kNumberTypeFloat16, "DefaultFormat"});
  auto res = gb.MatMul(x, y);
  std::vector<inner::NodePtr> outputs;
  for (size_t i = 0; i < param.post_num; i++) {
    auto scalar = gb.Tensor(i, kNumberTypeFloat16);
    outputs.emplace_back(gb.Add(res, scalar));
  }
  gb.SetOutputs(outputs);

  auto split_model = std::make_shared<graphkernel::inner::SplitModelAscend>();
  split_model->Run(gb.Get());
  EXPECT_EQ(split_model->areas().size(), param.post_num > 5 ? param.post_num - 4 : 1);
}

TEST_P(TestMatMulReshapePostFusion, test_matmul_reshape_fusion) {
  // get params
  const auto &param = GetParam();
  inner::GraphBuilder gb;
  auto x = gb.Parameter({param.shape_a, kNumberTypeFloat16, "DefaultFormat"});
  auto y = gb.Parameter({param.shape_b, kNumberTypeFloat16, "DefaultFormat"});
  auto c = gb.Parameter({param.shape_c, kNumberTypeFloat16, "DefaultFormat"});
  auto res = gb.Reshape(gb.MatMul(x, y), param.reshape_shape);
  gb.SetOutputs({gb.Add(res, c)});

  auto split_model = std::make_shared<graphkernel::inner::SplitModelAscend>();
  split_model->Run(gb.Get());
  EXPECT_EQ(split_model->areas().size() == 1, param.can_fusion);
}

INSTANTIATE_TEST_CASE_P(TestMatMulPostFusionCases, TestMatMulPostFusionNum,
                        testing::Values(TestMatMulPostFusionNumParams{7}, TestMatMulPostFusionNumParams{3},
                                        TestMatMulPostFusionNumParams{20}, TestMatMulPostFusionNumParams{1}));
INSTANTIATE_TEST_CASE_P(
  TestMatMulPostFusionCases, TestMatMulReshapePostFusion,
  testing::Values(TestMatMulReshapePostFusionParams{{64, 128}, {128, 256}, {1, 64 * 256}, {10, 64 * 256}, false},
                  TestMatMulReshapePostFusionParams{{64, 128}, {128, 256}, {1, 64 * 256}, {1, 64 * 256}, true},
                  TestMatMulReshapePostFusionParams{{64, 128}, {128, 256}, {32, 512}, {32, 512}, true}));
}  // namespace mindspore::graphkernel::test
