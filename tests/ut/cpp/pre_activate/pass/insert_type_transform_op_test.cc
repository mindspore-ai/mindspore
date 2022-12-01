/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "common/backend_common_test.h"
#include "ir/anf.h"
#include "ir/tensor.h"
#include "include/common/debug/anf_ir_dump.h"
#include "common/py_func_graph_fetcher.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "backend/common/optimizer/optimizer.h"
#include "backend/common/optimizer/pass_manager.h"
#include "include/common/utils/utils.h"

#define private public
#define protected public
#include "backend/common/pass/insert_type_transform_op.h"
#undef private
#undef protected

namespace mindspore {
namespace opt {
using kernel::KernelObjectType;
using KernelBuildInfoBuilder = kernel::KernelBuildInfo::KernelBuildInfoBuilder;

class TestInsertTypeTransformOp : public BackendCommon {
 public:
  TestInsertTypeTransformOp() : getPyFun_("gtest_input.pre_activate.insert_type_transform_op_test", true) {}
  ~TestInsertTypeTransformOp() override = default;

 public:
  void SetTupleUnfoldToTupleUnfoldKernelBuildInfo(const FuncGraphPtr &func_graph);
  void SetKernelBuildInfo(const AnfNodePtr &node, const std::vector<std::string> &input_formats,
                          const std::vector<TypeId> &input_types, const std::vector<std::string> &output_formats,
                          const std::vector<TypeId> &output_types, const std::vector<KernelObjectType> &input_obj_types,
                          const std::vector<KernelObjectType> &output_obj_types);
  UT::PyFuncGraphFetcher getPyFun_;
};

void TestInsertTypeTransformOp::SetTupleUnfoldToTupleUnfoldKernelBuildInfo(const FuncGraphPtr &g) {
  auto ret = g->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  auto addn2 = ret->input(1)->cast<CNodePtr>();
  MS_LOG(INFO) << "addn2 is " << addn2->fullname_with_scope();
  SetKernelBuildInfo(addn2, {"NCHW"}, {kNumberTypeFloat32}, {"NCHW"}, {kNumberTypeFloat32},
                     {KernelObjectType::TUPLE_UNFOLD}, {KernelObjectType::TUPLE_UNFOLD});

  auto split2_input_make_tuple = addn2->input(1)->cast<CNodePtr>();
  MS_LOG(INFO) << "split2_input_make_tuple is " << split2_input_make_tuple->fullname_with_scope();
  SetKernelBuildInfo(split2_input_make_tuple, {"NCHW", "NCHW"}, {kNumberTypeFloat32, kNumberTypeFloat32}, {"NCHW"},
                     {kNumberTypeFloat32}, {KernelObjectType::TENSOR, KernelObjectType::TENSOR},
                     {KernelObjectType::TUPLE_UNFOLD});

  auto split2_get_item1 = split2_input_make_tuple->input(1)->cast<CNodePtr>();
  MS_LOG(INFO) << "split2_get_item1 is " << split2_get_item1->fullname_with_scope();
  SetKernelBuildInfo(split2_get_item1, {"NCHW", "NCHW"}, {kNumberTypeFloat32, kNumberTypeInt64}, {"NCHW"},
                     {kNumberTypeFloat32}, {KernelObjectType::TUPLE_UNFOLD, KernelObjectType::TENSOR},
                     {KernelObjectType::TENSOR});

  auto split2_get_item2 = split2_input_make_tuple->input(1)->cast<CNodePtr>();
  MS_LOG(INFO) << "split2_get_item2 is " << split2_get_item2->fullname_with_scope();
  SetKernelBuildInfo(split2_get_item2, {"NCHW", "NCHW"}, {kNumberTypeFloat32, kNumberTypeInt64}, {"NCHW"},
                     {kNumberTypeFloat32}, {KernelObjectType::TUPLE_UNFOLD, KernelObjectType::TENSOR},
                     {KernelObjectType::TENSOR});

  auto split2_1 = split2_get_item2->input(1)->cast<CNodePtr>();
  auto split2_2 = split2_get_item2->input(1)->cast<CNodePtr>();
  ASSERT_TRUE(split2_1 == split2_2);
  MS_LOG(INFO) << "split2 is " << split2_1->fullname_with_scope();
  SetKernelBuildInfo(split2_2, {"NCHW"}, {kNumberTypeFloat32}, {"NCHW", "NCHW"},
                     {kNumberTypeFloat32, kNumberTypeFloat32}, {KernelObjectType::TUPLE_UNFOLD},
                     {KernelObjectType::TENSOR});

  auto addn1 = split2_2->input(1)->cast<CNodePtr>();
  MS_LOG(INFO) << "addn1 is " << addn1->fullname_with_scope();
  SetKernelBuildInfo(addn1, {"NCHW"}, {kNumberTypeFloat32}, {"NCHW"}, {kNumberTypeFloat32},
                     {KernelObjectType::TUPLE_UNFOLD}, {KernelObjectType::TENSOR});

  auto split1 = addn1->input(1)->cast<CNodePtr>();
  MS_LOG(INFO) << "split1 is " << split1->fullname_with_scope();
  SetKernelBuildInfo(split1, {"NCHW"}, {kNumberTypeFloat32}, {"NCHW", "NCHW"}, {kNumberTypeFloat32, kNumberTypeFloat32},
                     {KernelObjectType::TENSOR}, {KernelObjectType::TUPLE_UNFOLD});

  // The input is a value.
  auto input_node = split1->input(1);
  MS_LOG(INFO) << "input_node is " << input_node->fullname_with_scope();
  SetKernelBuildInfo(input_node, {"NCHW"}, {kNumberTypeFloat32}, {"NCHW"}, {kNumberTypeFloat32},
                     {KernelObjectType::TENSOR}, {KernelObjectType::TENSOR});
}

void TestInsertTypeTransformOp::SetKernelBuildInfo(
  const AnfNodePtr &node, const std::vector<std::string> &input_formats, const std::vector<TypeId> &input_types,
  const std::vector<std::string> &output_formats, const std::vector<TypeId> &output_types,
  const std::vector<KernelObjectType> &input_obj_types, const std::vector<KernelObjectType> &output_obj_types) {
  KernelBuildInfoBuilder builder;
  builder.SetInputsFormat(input_formats);
  builder.SetInputsDeviceType(input_types);
  builder.SetOutputsFormat(output_formats);
  builder.SetOutputsDeviceType(output_types);
  builder.SetInputsKernelObjectType(input_obj_types);
  builder.SetOutputsKernelObjectType(output_obj_types);
  node->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), node.get());
}

/// Feature: Dynamic shape.
/// Description: Test TupleUnfold to TupleUnfold type transforming pass.
/// Expectation: After InsertTypeTransformOp pass, the graph is identical to the expected graph expressed by python.
TEST_F(TestInsertTypeTransformOp, test_tuple_unfold_to_tuple_unfold_transform) {
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_tuple_unfold_to_tuple_unfold_transform", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int64_t> shp_x{2, 4};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto func_graph = GetFuncGraph(g, args_spec_list);
  ASSERT_TRUE(func_graph != nullptr);
  SetTupleUnfoldToTupleUnfoldKernelBuildInfo(func_graph);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::InsertTypeTransformOp>());
  optimizer->AddPassManager(pm);
  optimizer->Optimize(func_graph);

  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_tuple_unfold_to_tuple_unfold_transform", "after");
  ASSERT_TRUE(g_after != nullptr);
  EXPECT_TRUE(CheckEqualGraph(func_graph, g_after));
}
}  // namespace opt
}  // namespace mindspore
