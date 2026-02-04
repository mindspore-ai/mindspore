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

#include <string>
#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "backend/ms_backend/graph_fusion/convert_input_and_attr.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_expander_cloud.h"
#include "backend/ms_backend/graph_fusion/expander/base.h"
#include "ir/graph_utils.h"
#include "utils/anf_utils.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"

namespace mindspore::graphkernel::test {
namespace {
struct Params {
  bool can_expand;
  TypePtr input_type;
  ShapeArray input_shape;
  std::string data_format;
  ShapeVector expect_shape;
  bool is_grad;
};

class UpdateAttr : public opt::Pass {
 public:
  UpdateAttr(const std::string &data_format) : Pass("update_attr"), data_format_(data_format) {}
  ~UpdateAttr() override = default;
  bool Run(const FuncGraphPtr &func_graph) override {
    auto nodes = TopoSort(func_graph->get_return());
    for (const auto &node : nodes) {
      if (IsPrimitiveCNode(node, prim::kPrimBiasAdd) || IsPrimitiveCNode(node, prim::kPrimBiasAddGrad)) {
        auto prim = GetCNodePrimitive(node);
        MS_EXCEPTION_IF_NULL(prim);
        prim->set_attr("data_format", MakeValue(data_format_));
      }
    }
    return false;
  }

 private:
  const std::string &data_format_;
};
}  // namespace

/// Feature: Test BiasAdd/BiasAddGrad expander
/// Description: test op with different inputs
/// Expectation: Can be expanded only when its input data types are supported.
class TestBiasAddExpander : public TestGraphKernelExpander, public testing::WithParamInterface<Params> {
  void SetUp() override { SetDeviceTarget(kAscendDevice); }
};

TEST_P(TestBiasAddExpander, bias_add) {
  const auto &param = GetParam();
  ConstructGraph c;
  CNodePtr op;
  if (!param.is_grad) {
    auto x0 = c.NewTensorInput("x0", param.input_type, param.input_shape[0]);
    auto x1 = c.NewTensorInput("x1", param.input_type, param.input_shape[1]);
    auto data_format = c.NewValueNode(MakeValue(static_cast<int64_t>(Format::NCHW)));
    op = c.NewCNodeWithBuildInfo("BiasAdd", {x0, x1, data_format});
  } else {
    auto x0 = c.NewTensorInput("x0", param.input_type, param.input_shape[0]);
    auto data_format = c.NewValueNode(MakeValue(static_cast<int64_t>(Format::NCHW)));
    op = c.NewCNodeWithBuildInfo("BiasAddGrad", {x0, data_format});
  }

  // update data format
  auto build_info = AnfAlgo::GetSelectKernelBuildInfo(op);
  if (build_info != nullptr) {
    auto formats = build_info->GetAllInputFormats();
    formats[0] = param.data_format;
    build_info->SetInputsFormat(formats);
    if (!param.is_grad) {
      build_info->SetOutputsFormat({param.data_format});
    }
  }

  c.SetOutput(op);
  auto fg = c.GetGraph();
  RunPass(
    fg, {std::make_shared<graphkernel::ConvertFrontEndToGraphKernel>(), std::make_shared<UpdateAttr>(param.data_format),
         std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
  size_t gk_node{0};
  auto nodes = TopoSort(c.GetGraph()->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
      CompareShapeAndType(node, 0, param.expect_shape, param.input_type->type_id());
      gk_node += 1;
    }
  }
  size_t gk_size = param.can_expand ? 1 : 0;
  ASSERT_EQ(gk_node, gk_size);
}

// can not set DefaultFormat
// need FRACTAL_NZ
INSTANTIATE_TEST_CASE_P(TestOpBiasAdd, TestBiasAddExpander,
                        testing::Values(
                          // BiasAdd
                          Params{true, kFloat16, {{16, 6, 28, 28}, {6}}, "NCHW", {16, 6, 28, 28}, false},
                          Params{true, kFloat16, {{16, 6, 28, 28}, {6}}, "NHWC", {16, 28, 28, 6}, false},
                          Params{true, kFloat16, {{16, 120}, {120}}, "NCHW", {16, 120}, false},
                          Params{true, kFloat32, {{16, 6, 28, 28}, {6}}, "NCHW", {16, 6, 28, 28}, false},
                          Params{true, kBFloat16, {{16, 6, 28, 28}, {6}}, "NCHW", {16, 6, 28, 28}, false},
                          Params{true, kInt32, {{16, 6, 28, 28}, {6}}, "NCHW", {16, 6, 28, 28}, false},
                          Params{false, kInt64, {{16, 6, 28, 28}, {6}}, "NCHW", {16, 6, 28, 28}, false},
                          Params{false, kInt16, {{16, 6, 28, 28}, {6}}, "NCHW", {16, 6, 28, 28}, false},
                          Params{false, kInt8, {{16, 6, 28, 28}, {6}}, "NCHW", {16, 6, 28, 28}, false},
                          // BiasAddGrad
                          Params{true, kFloat16, {{16, 6, 28, 28}}, "NHWC", {6}, true},
                          Params{true, kFloat16, {{16, 6, 28, 28}}, "NCHW", {6}, true},
                          Params{true, kFloat16, {{16, 6, 28, 28}}, "DefaultFormat", {6}, true},
                          Params{true, kFloat16, {{16, 6, 28}}, "DefaultFormat", {6}, true},
                          Params{true, kFloat16, {{16, 120}}, "DefaultFormat", {120}, true},
                          Params{true, kFloat32, {{16, 6, 28, 28}}, "DefaultFormat", {6}, true},
                          Params{true, kBFloat16, {{16, 6, 28, 28}}, "DefaultFormat", {6}, true},
                          Params{false, kInt32, {{16, 6, 28, 28}}, "DefaultFormat", {6}, true},
                          Params{false, kInt64, {{16, 6, 28, 28}}, "DefaultFormat", {6}, true},
                          Params{false, kInt16, {{16, 6, 28, 28}}, "DefaultFormat", {6}, true},
                          Params{false, kInt8, {{16, 6, 28, 28}}, "DefaultFormat", {6}, true}));
}  // namespace mindspore::graphkernel::test
