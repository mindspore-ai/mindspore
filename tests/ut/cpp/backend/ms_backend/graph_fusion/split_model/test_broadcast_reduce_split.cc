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
#include <string>

#include "utils/ms_context.h"
#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "common/graph_optimizer_test_framework.h"
#include "backend/ms_backend/graph_fusion/model/graph_builder.h"
#include "backend/ms_backend/graph_fusion/adapter/split_model_ascend.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"

namespace mindspore::graphkernel::test {
class TestBroadcastReduceSplit : public GraphKernelCommonTestSuite {
 public:
  TestBroadcastReduceSplit() {}
  void SetUp() override {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    context->set_param<std::string>(MS_CTX_DEVICE_TARGET, kAscendDevice);

    std::map<std::string, std::string> jit_config;
    jit_config["graph_kernel_flags"] = "--kernel_generator=DVM";
    graphkernel::GraphKernelFlags::SaveJitConfig(jit_config);
  }
};

/// Feature: Test split pass in broadcast output
/// Description: when an area has broadcast output, reduce will not fuse it
/// Expectation: the reduce area not fuse with the broadcast area
TEST_F(TestBroadcastReduceSplit, test_broadcast_reduce_split) {
  // get params
  inner::GraphBuilder gb;
  auto x = gb.Parameter({{4, 1}, kNumberTypeFloat16, "DefaultFormat"});
  auto y = gb.Parameter({{4, 1}, kNumberTypeFloat16, "DefaultFormat"});
  auto z = gb.Parameter({{4, 4}, kNumberTypeFloat16, "DefaultFormat"});
  auto neg = gb.Neg(x);
  auto mul = gb.Mul(neg, y);
  auto broadcast = gb.BroadcastTo(mul, {4, 4});
  auto div = gb.Div(broadcast, z);
  auto res = gb.ReduceSum(div, {1}, true);
  gb.SetOutputs({neg, div, res});

  auto split_model = std::make_shared<graphkernel::inner::SplitModelAscend>();
  split_model->Run(gb.Get());
  EXPECT_EQ(split_model->areas().size(), 2);
}

}  // namespace mindspore::graphkernel::test