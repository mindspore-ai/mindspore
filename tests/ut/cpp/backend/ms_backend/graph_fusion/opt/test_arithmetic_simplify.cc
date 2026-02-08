/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "backend/ms_backend/graph_fusion/convert_input_and_attr.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_cluster_cloud.h"
#include "backend/ms_backend/graph_fusion/axis_normalizer.h"
#include "backend/ms_backend/graph_fusion/core/arithmetic_simplify.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "utils/anf_utils.h"
#include "include/utils/anfalgo.h"

namespace mindspore::graphkernel::test {
namespace {
struct ReduceParams {
  ShapeVector input_shape;
  ShapeArray axis;
  std::vector<bool> keep_dims;
  bool can_simplify;
};

struct Transpose2Params {
  ShapeArray perm;
  bool can_simplify;
};

struct StridedSliceParams {
  ShapeVector input_shape;
  ShapeVector begin;
  ShapeVector end;
  ShapeVector stride;
  ShapeVector marks;
  bool can_simplify;
};
}  // namespace

class TestPassArithmeticSimplify : public GraphKernelCommonTestSuite {};

/// Feature: Test graph kernel ArithmeticSimplify pass
/// Description: Transpose(Transpose(A,B),C)=Transpose(A,D)
/// Expectation: Convert src pattern to dst pattern
TEST_F(TestPassArithmeticSimplify, transpose1) {
  SetDeviceTarget(kAscendDevice);
  SetGraphKernelFlags("--enable_cluster_ops=Transpose");
  ConstructGraph c;
  auto x0 = c.NewTensorInput("x0", kFloat16, {4, 16, 8});
  auto y0 = c.NewCNodeWithBuildInfo("Transpose", {x0, c.NewValueNode(MakeValue(ShapeVector{1, 0, 2}))});
  auto y1 = c.NewCNodeWithBuildInfo("Transpose", {y0, c.NewValueNode(MakeValue(ShapeVector{0, 2, 1}))});
  c.SetOutput(y1);
  auto fg = c.GetGraph();
  auto cb = graphkernel::Callback::Instance();
  MS_EXCEPTION_IF_NULL(cb);
  auto output_shape = cb->GetOutputShape(y1, 0);
  RunPass(fg, {std::make_shared<graphkernel::StaticShapeCluster>(), std::make_shared<ArithmeticSimplify>()});
  ShapeVector cur_shape;
  auto nodes = TopoSort(c.GetGraph()->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
      auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
      MS_EXCEPTION_IF_NULL(sub_graph);
      auto out = sub_graph->get_return()->input(1);
      cur_shape = cb->GetOutputShape(out, 0);
      EXPECT_EQ(out->cast<CNodePtr>()->input(1)->isa<CNode>(), false);
      break;
    }
  }
  EXPECT_EQ(cur_shape, output_shape);
}

/// Feature: Test graph kernel ArithmeticSimplify pass
/// Description: Transpose(Transpose(Reshape(A,B),C),D)=Reshape(A,E)
/// Expectation: Convert src pattern to dst pattern
TEST_F(TestPassArithmeticSimplify, transpose2) {
  SetDeviceTarget(kCPUDevice);
  SetGraphKernelFlags("--enable_cluster_ops=Transpose,Reshape");
  std::vector<Transpose2Params> params{{{{2, 0, 1}, {1, 2, 0}}, true}, {{{1, 0, 2}, {0, 2, 1}}, false}};
  SetDeviceTarget(kAscendDevice);
  for (const auto &param : params) {
    ConstructGraph c;
    auto x0 = c.NewTensorInput("x0", kFloat32, {10, 20});
    auto y0 = c.NewCNodeWithBuildInfo("Reshape", {x0, c.NewValueNode(MakeValue(ShapeVector{2, 5, 20}))});
    auto y1 = c.NewCNodeWithBuildInfo("Transpose", {y0, c.NewValueNode(MakeValue(param.perm[0]))});
    auto y2 = c.NewCNodeWithBuildInfo("Transpose", {y1, c.NewValueNode(MakeValue(param.perm[1]))});
    c.SetOutput(y2);
    auto fg = c.GetGraph();
    auto cb = graphkernel::Callback::Instance();
    MS_EXCEPTION_IF_NULL(cb);
    auto output_shape = cb->GetOutputShape(y2, 0);
    RunPass(fg, {std::make_shared<graphkernel::StaticShapeCluster>(), std::make_shared<graphkernel::AxisNormalizer>(),
                 std::make_shared<ArithmeticSimplify>()});
    ShapeVector cur_shape;
    auto nodes = TopoSort(c.GetGraph()->get_return());
    for (const auto &node : nodes) {
      if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
        auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
        MS_EXCEPTION_IF_NULL(sub_graph);
        auto out = sub_graph->get_return()->input(1);
        cur_shape = cb->GetOutputShape(out, 0);
        EXPECT_EQ(IsPrimitiveCNode(out, prim::kPrimReshape), param.can_simplify);
        break;
      }
    }
    EXPECT_EQ(cur_shape, output_shape);
  }
}

/// Feature: Test graph kernel ArithmeticSimplify pass
/// Description: Reshape(Reshape(A,B),C)=Reshape(A,C)
/// Expectation: Convert src pattern to dst pattern
TEST_F(TestPassArithmeticSimplify, reshape) {
  SetDeviceTarget(kCPUDevice);
  SetGraphKernelFlags("--enable_cluster_ops=Reshape");
  ConstructGraph c;
  auto x0 = c.NewTensorInput("x0", kFloat16, {10, 20});
  auto y0 = c.NewCNodeWithBuildInfo("Reshape", {x0, c.NewValueNode(MakeValue(ShapeVector{2, 5, 20}))});
  auto y1 = c.NewCNodeWithBuildInfo("Reshape", {y0, c.NewValueNode(MakeValue(ShapeVector{2, 100}))});
  c.SetOutput(y1);
  auto fg = c.GetGraph();
  auto cb = graphkernel::Callback::Instance();
  MS_EXCEPTION_IF_NULL(cb);
  auto output_shape = cb->GetOutputShape(y1, 0);
  RunPass(fg, {std::make_shared<graphkernel::StaticShapeCluster>(), std::make_shared<ArithmeticSimplify>()});
  ShapeVector cur_shape;
  auto nodes = TopoSort(c.GetGraph()->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
      auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
      MS_EXCEPTION_IF_NULL(sub_graph);
      auto out = sub_graph->get_return()->input(1);
      cur_shape = cb->GetOutputShape(out, 0);
      EXPECT_EQ(out->cast<CNodePtr>()->input(1)->isa<CNode>(), false);
      break;
    }
  }
  EXPECT_EQ(cur_shape, output_shape);
}

/// Feature: Test graph kernel ArithmeticSimplify pass
/// Description: Reduce(Reduce(A,B),C) = Reduce(A,D)
/// Expectation: Convert src pattern to dst pattern
TEST_F(TestPassArithmeticSimplify, reduce) {
  SetDeviceTarget(kAscendDevice);
  std::vector<ReduceParams> params{{{10, 20, 30}, {{0}, {-2}}, {true, true}, true},
                                   {{10, 20, 30, 40}, {{0, 1}, {1}}, {false, false}, true},
                                   {{10, 20, 30, 40}, {{2}, {0, 1, 2}}, {false, false}, true},
                                   {{10, 20, 30}, {{0}, {1}}, {false, false}, true},
                                   {{10, 20, 30}, {{0}, {}}, {false, false}, true},
                                   {{10, 20, 30}, {{0}, {1}}, {false, true}, false}};
  for (const auto &param : params) {
    ConstructGraph c;
    auto x0 = c.NewTensorInput("x0", kFloat32, param.input_shape);
    auto y0 = c.NewCNodeWithBuildInfo(
      "ReduceSum", {x0, c.NewValueNode(MakeValue(param.axis[0])), c.NewValueNode(MakeValue<bool>(param.keep_dims[0])),
                    c.NewValueNode(MakeValue<bool>(false))});
    auto y1 = c.NewCNodeWithBuildInfo(
      "ReduceSum", {y0, c.NewValueNode(MakeValue(param.axis[1])), c.NewValueNode(MakeValue<bool>(param.keep_dims[1])),
                    c.NewValueNode(MakeValue<bool>(false))});
    c.SetOutput(y1);
    auto fg = c.GetGraph();
    auto cb = graphkernel::Callback::Instance();
    MS_EXCEPTION_IF_NULL(cb);
    auto output_shape = cb->GetOutputShape(y1, 0);
    RunPass(fg, {std::make_shared<graphkernel::ConvertFrontEndToGraphKernel>(),
                 std::make_shared<graphkernel::StaticShapeCluster>(), std::make_shared<graphkernel::AxisNormalizer>(),
                 std::make_shared<ArithmeticSimplify>()});
    ShapeVector cur_shape;
    auto nodes = TopoSort(c.GetGraph()->get_return());
    for (const auto &node : nodes) {
      if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
        auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
        MS_EXCEPTION_IF_NULL(sub_graph);
        auto out = sub_graph->get_return()->input(1);
        cur_shape = cb->GetOutputShape(out, 0);
        EXPECT_EQ(out->cast<CNodePtr>()->input(1)->isa<CNode>(), !param.can_simplify);
        break;
      }
    }
    EXPECT_EQ(cur_shape, output_shape);
  }
}

/// Feature: Test graph kernel ArithmeticSimplify pass
/// Description: StridedSlice(A,B,C,D)=Reshape(A,E)
/// Expectation: Convert src pattern to dst pattern
TEST_F(TestPassArithmeticSimplify, strided_slice) {
  SetDeviceTarget(kAscendDevice);
  SetGraphKernelFlags("--enable_cluster_ops=StridedSlice");
  std::vector<StridedSliceParams> params{{{16, 32}, {0, 0}, {16, 32}, {1, 1}, {0, 0, 0, 0, 0}, true},
                                         {{16, 1}, {0, 0}, {16, 1}, {1, 1}, {0, 0, 0, 0, 2}, true},
                                         {{-1, -1}, {0, 0}, {16, 32}, {1, 1}, {0, 0, 0, 0, 0}, false},
                                         {{16, 32}, {0, 0}, {16, 32}, {1, 1}, {0, 0, 0, 0, 2}, false},
                                         {{16, 32}, {0, 1}, {16, 32}, {1, 1}, {0, 0, 0, 0, 0}, false},
                                         {{16, 32}, {0, 0}, {16, 30}, {1, 1}, {0, 0, 0, 0, 0}, false},
                                         {{16, 32}, {0, 0}, {16, 32}, {1, 2}, {0, 0, 0, 0, 0}, false}};
  for (const auto &param : params) {
    ConstructGraph c;
    auto x0 = c.NewTensorInput("x0", kFloat32, param.input_shape);
    auto y0 = c.NewCNodeWithBuildInfo(
      "StridedSlice", {x0, c.NewValueNode(MakeValue(param.begin)), c.NewValueNode(MakeValue(param.end)),
                       c.NewValueNode(MakeValue(param.stride)), c.NewValueNode(MakeValue(param.marks[0])),
                       c.NewValueNode(MakeValue(param.marks[1])), c.NewValueNode(MakeValue(param.marks[2])),
                       c.NewValueNode(MakeValue(param.marks[3])), c.NewValueNode(MakeValue(param.marks[4]))});
    c.SetOutput(y0);
    auto fg = c.GetGraph();
    auto cb = graphkernel::Callback::Instance();
    MS_EXCEPTION_IF_NULL(cb);
    auto output_shape = cb->GetOutputShape(y0, 0);
    RunPass(fg, {std::make_shared<graphkernel::ConvertFrontEndToGraphKernel>(),
                 std::make_shared<graphkernel::StaticShapeCluster>(), std::make_shared<ArithmeticSimplify>()});
    ShapeVector cur_shape;
    auto nodes = TopoSort(c.GetGraph()->get_return());
    for (const auto &node : nodes) {
      if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
        auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
        MS_EXCEPTION_IF_NULL(sub_graph);
        auto out = sub_graph->get_return()->input(1);
        cur_shape = cb->GetOutputShape(out, 0);
        EXPECT_EQ(cur_shape, output_shape);
        EXPECT_EQ(IsPrimitiveCNode(out, prim::kPrimReshape), param.can_simplify);
        break;
      }
    }
  }
}
}  // namespace mindspore::graphkernel::test
