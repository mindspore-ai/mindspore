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
#include "ir/tensor_new.h"
#include "utils/ms_context.h"
#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "utils/anf_utils.h"
#include "abstract/abstract_value.h"
#include "common/mockcpp.h"
#include "common/graph_optimizer_test_framework.h"
#include "backend/ms_backend/graph_fusion/model/graph_builder.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_cluster_cloud.h"
#include "backend/ms_backend/graph_fusion/convert_input_and_attr.h"
#include "backend/ms_backend/graph_fusion/core/arithmetic_simplify.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"

namespace mindspore::graphkernel {
bool ConvertTensorToParameter(const FuncGraphPtr &fg, AnfNodePtrList *inputs_ptr);
}

namespace mindspore::graphkernel::test {
struct TestArithmeticSimplifyParams {
  ShapeVector a_shape;
  ShapeVector const_shape;
  ShapeVector axis;
  bool keep_dims;
  bool skip_mode;
  bool need_simplify;
};

class TestArithmeticSimplify : public GraphKernelCommonTestSuite,
                               public testing::WithParamInterface<TestArithmeticSimplifyParams> {
 public:
  void SetUp() override {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    context->set_param<std::string>(MS_CTX_DEVICE_TARGET, kAscendDevice);
  }
};

TEST_P(TestArithmeticSimplify, test_arithmetic_simplify) {
  SetGraphKernelFlags("--exact_precision_mode=false");
  // get params
  MOCKER_CPP(ConvertTensorToParameter, bool (*)(const FuncGraphPtr &, AnfNodePtrList *))
    .stubs()
    .will(returnValue(true));
  const auto &param = GetParam();
  ConstructGraph c;
  auto input_a = c.NewTensorInput("input_a", kFloat32, param.a_shape);
  std::vector<float> const_value(
    std::accumulate(param.const_shape.begin(), param.const_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>()), 3);
  auto input_const = c.NewValueNode(
    tensor::from_buffer(kNumberTypeFloat32, param.const_shape, &const_value[0], kNumberTypeFloat32));
  auto axis = c.NewValueNode(tensor::from_vector(param.axis));
  auto keep_dims = c.NewValueNode(MakeValue<bool>(param.keep_dims));
  auto skip_mode = c.NewValueNode(MakeValue<bool>(param.skip_mode));
  auto mul = c.NewCNodeWithBuildInfo("Mul", {input_a, input_const}, {});
  auto reduce = c.NewCNodeWithBuildInfo("ReduceSum", {mul, axis, keep_dims, skip_mode}, {});
  c.SetOutput(reduce);
  RunPass(c.GetGraph(),
          {std::make_shared<graphkernel::ConvertFrontEndToGraphKernel>(),
           std::make_shared<graphkernel::StaticShapeCluster>(), std::make_shared<graphkernel::ArithmeticSimplify>()});
  auto g = c.GetGraph();
  UT_CHECK_NULL(g);
  auto gknodes = GetAllGKNodes(g);
  ASSERT_EQ(gknodes.size(), 1);
  auto sub_graph = GetCNodeFuncGraph(gknodes[0]);
  mindspore::opt::CheckPattern checker;
  checker.src_pattern_.AddVar("A")
    .AddVar("const")
    .AddVar("axis")
    .AddCNode("reduce", {std::make_shared<Primitive>("ReduceSum"), "A", "axis"})
    .AddCNode("mul", {std::make_shared<Primitive>("Mul"), "reduce", "const"});
  if (param.need_simplify) {
    EXPECT_TRUE(checker.build_pattern_map(sub_graph->output()));
  } else {
    EXPECT_FALSE(checker.build_pattern_map(sub_graph->output()));
  }
  GlobalMockObject::verify();
}

INSTANTIATE_TEST_CASE_P(TestArithmeticSimplifyCases, TestArithmeticSimplify,
                        testing::Values(TestArithmeticSimplifyParams{{16}, {1, 1}, {0, 1}, true, false, false},
                                        TestArithmeticSimplifyParams{{16, 16}, {1, 1}, {0, 1}, false, false, false},
                                        TestArithmeticSimplifyParams{{1, 16}, {16, 16}, {1}, true, false, false},
                                        TestArithmeticSimplifyParams{{16, 16}, {16, 1}, {1}, true, false, true},
                                        TestArithmeticSimplifyParams{{16, 16}, {1, 1}, {0, 1}, true, false, true},
                                        TestArithmeticSimplifyParams{{16, 16, 16}, {1, 1}, {0}, true, false, true}));

}  // namespace mindspore::graphkernel::test
