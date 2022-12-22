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
using KernelBuildInfoPtr = kernel::KernelBuildInfoPtr;
using KernelBuildInfoBuilder = kernel::KernelBuildInfo::KernelBuildInfoBuilder;

class TestInsertTypeTransformOp : public BackendCommon {
 public:
  TestInsertTypeTransformOp() : getPyFun_("gtest_input.pre_activate.insert_type_transform_op_test", true) {}
  ~TestInsertTypeTransformOp() override = default;

 public:
  void SetTupleUnfoldToTupleUnfoldKernelBuildInfo(const FuncGraphPtr &g, AnfNodePtr *split1_ptr, AnfNodePtr *addn1_ptr,
                                                  AnfNodePtr *split2_ptr, AnfNodePtr *addn2_ptr);
  void SetTupleUnfoldToTupleKernelBuildInfo(const FuncGraphPtr &g, AnfNodePtr *make_tuple_ptr, AnfNodePtr *split_ptr,
                                            AnfNodePtr *tuple_add1_ptr, AnfNodePtr *tuple_add2_ptr);
  void SetTupleUnfoldToTensorKernelBuildInfo(const FuncGraphPtr &g, AnfNodePtr *make_tuple, AnfNodePtr *reshape);
  void SetTupleToTensorKernelBuildInfo(const FuncGraphPtr &g, AnfNodePtr *reshape_ptr);

  void SetKernelBuildInfo(const AnfNodePtr &node, const std::vector<std::string> &input_formats,
                          const std::vector<TypeId> &input_types, const std::vector<std::string> &output_formats,
                          const std::vector<TypeId> &output_types, const std::vector<KernelObjectType> &input_obj_types,
                          const std::vector<KernelObjectType> &output_obj_types);

  // Check whether input and output kernel info are set as expected.
  // This check is quite important to ensure there's no issue after this InsertTypeTransformOp pass.
  void CheckInputKernelInfo(const AnfNodePtr &node, size_t input_format_size, size_t input_type_size,
                            size_t input_obj_type_size);
  void CheckOutputKernelInfo(const AnfNodePtr &node, size_t output_format_size, size_t output_type_size,
                             size_t output_obj_type_size);

  UT::PyFuncGraphFetcher getPyFun_;
};

void TestInsertTypeTransformOp::SetTupleUnfoldToTupleUnfoldKernelBuildInfo(
  const FuncGraphPtr &g, AnfNodePtr *split1_ptr, AnfNodePtr *addn1_ptr, AnfNodePtr *split2_ptr, AnfNodePtr *addn2_ptr) {
  auto ret = g->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  auto addn2 = ret->input(1)->cast<CNodePtr>();
  *addn2_ptr = addn2;
  MS_LOG(INFO) << "addn2 is " << addn2->fullname_with_scope();
  SetKernelBuildInfo(addn2, {"NCHW"}, {kNumberTypeFloat32}, {"NCHW"}, {kNumberTypeFloat32},
                     {KernelObjectType::TUPLE_UNFOLD}, {KernelObjectType::TENSOR});

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

  auto split2_get_item2 = split2_input_make_tuple->input(2)->cast<CNodePtr>();
  MS_LOG(INFO) << "split2_get_item2 is " << split2_get_item2->fullname_with_scope();
  SetKernelBuildInfo(split2_get_item2, {"NCHW", "NCHW"}, {kNumberTypeFloat32, kNumberTypeInt64}, {"NCHW"},
                     {kNumberTypeFloat32}, {KernelObjectType::TUPLE_UNFOLD, KernelObjectType::TENSOR},
                     {KernelObjectType::TENSOR});

  auto split2_1 = split2_get_item1->input(1)->cast<CNodePtr>();
  auto split2_2 = split2_get_item2->input(1)->cast<CNodePtr>();
  ASSERT_TRUE(split2_1 == split2_2);
  *split2_ptr = split2_2;
  MS_LOG(INFO) << "split2 is " << split2_1->fullname_with_scope();
  SetKernelBuildInfo(split2_2, {"NCHW"}, {kNumberTypeFloat32}, {"NCHW", "NCHW"},
                     {kNumberTypeFloat32, kNumberTypeFloat32}, {KernelObjectType::TENSOR},
                     {KernelObjectType::TUPLE_UNFOLD});

  auto addn1 = split2_2->input(1)->cast<CNodePtr>();
  *addn1_ptr = addn1;
  MS_LOG(INFO) << "addn1 is " << addn1->fullname_with_scope();
  SetKernelBuildInfo(addn1, {"NCHW"}, {kNumberTypeFloat32}, {"NCHW"}, {kNumberTypeFloat32},
                     {KernelObjectType::TUPLE_UNFOLD}, {KernelObjectType::TENSOR});

  auto split1 = addn1->input(1)->cast<CNodePtr>();
  *split1_ptr = split1;
  MS_LOG(INFO) << "split1 is " << split1->fullname_with_scope();
  SetKernelBuildInfo(split1, {"NCHW"}, {kNumberTypeFloat32}, {"NCHW", "NCHW"}, {kNumberTypeFloat32, kNumberTypeFloat32},
                     {KernelObjectType::TENSOR}, {KernelObjectType::TUPLE_UNFOLD});

  // The input is a value.
  auto input_node = split1->input(1);
  MS_LOG(INFO) << "input_node is " << input_node->fullname_with_scope();
  SetKernelBuildInfo(input_node, {"NCHW"}, {kNumberTypeFloat32}, {"NCHW"}, {kNumberTypeFloat32},
                     {KernelObjectType::TENSOR}, {KernelObjectType::TENSOR});
}

void TestInsertTypeTransformOp::SetTupleUnfoldToTupleKernelBuildInfo(const FuncGraphPtr &g, AnfNodePtr *make_tuple_ptr,
                                                                     AnfNodePtr *split_ptr, AnfNodePtr *tuple_add1_ptr,
                                                                     AnfNodePtr *tuple_add2_ptr) {
  auto ret = g->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  auto tuple_add2 = ret->input(1)->cast<CNodePtr>();
  *tuple_add2_ptr = tuple_add2;
  SetKernelBuildInfo(tuple_add2, {"NCHW"}, {kNumberTypeFloat32}, {"NCHW"}, {kNumberTypeFloat32},
                     {KernelObjectType::TUPLE}, {KernelObjectType::TENSOR});

  auto split = tuple_add2->input(1)->cast<CNodePtr>();
  *split_ptr = split;
  MS_LOG(INFO) << "split is " << split->fullname_with_scope();
  SetKernelBuildInfo(split, {"NCHW"}, {kNumberTypeFloat32}, {"NCHW", "NCHW"}, {kNumberTypeFloat32, kNumberTypeFloat32},
                     {KernelObjectType::TENSOR}, {KernelObjectType::TUPLE_UNFOLD});

  auto tuple_add1 = split->input(1)->cast<CNodePtr>();
  *tuple_add1_ptr = tuple_add1;
  SetKernelBuildInfo(tuple_add1, {"NCHW"}, {kNumberTypeFloat32}, {"NCHW"}, {kNumberTypeFloat32},
                     {KernelObjectType::TUPLE}, {KernelObjectType::TENSOR});

  auto make_tuple = tuple_add1->input(1)->cast<CNodePtr>();
  *make_tuple_ptr = make_tuple;
  MS_LOG(INFO) << "make_tuple is " << make_tuple->fullname_with_scope();
  SetKernelBuildInfo(make_tuple, {"NCHW", "NCHW"}, {kNumberTypeFloat32, kNumberTypeFloat32}, {"NCHW"},
                     {kNumberTypeFloat32}, {KernelObjectType::TENSOR, KernelObjectType::TENSOR},
                     {KernelObjectType::TUPLE_UNFOLD});

  auto input_node1 = make_tuple->input(1);
  SetKernelBuildInfo(input_node1, {"NCHW"}, {kNumberTypeFloat32}, {"NCHW"}, {kNumberTypeFloat32},
                     {KernelObjectType::TENSOR}, {KernelObjectType::TENSOR});

  auto input_node2 = make_tuple->input(2);
  SetKernelBuildInfo(input_node2, {"NCHW"}, {kNumberTypeFloat32}, {"NCHW"}, {kNumberTypeFloat32},
                     {KernelObjectType::TENSOR}, {KernelObjectType::TENSOR});
}

void TestInsertTypeTransformOp::SetTupleUnfoldToTensorKernelBuildInfo(const FuncGraphPtr &g, AnfNodePtr *make_tuple_ptr,
                                                                      AnfNodePtr *reshape_ptr) {
  auto ret = g->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  auto reshape = ret->input(1)->cast<CNodePtr>();
  *reshape_ptr = reshape;
  SetKernelBuildInfo(reshape, {"NCHW", "NCHW"}, {kNumberTypeFloat32, kNumberTypeFloat32}, {"NCHW"},
                     {kNumberTypeFloat32}, {KernelObjectType::TENSOR, KernelObjectType::TENSOR},
                     {KernelObjectType::TENSOR});

  auto input_node3 = reshape->input(1);
  SetKernelBuildInfo(input_node3, {"NCHW"}, {kNumberTypeFloat32}, {"NCHW"}, {kNumberTypeFloat32},
                     {KernelObjectType::TENSOR}, {KernelObjectType::TENSOR});

  auto make_tuple = reshape->input(2)->cast<CNodePtr>();
  *make_tuple_ptr = make_tuple;
  SetKernelBuildInfo(make_tuple, {"NCHW", "NCHW"}, {kNumberTypeFloat32, kNumberTypeFloat32}, {"NCHW"},
                     {kNumberTypeFloat32}, {KernelObjectType::TENSOR, KernelObjectType::TENSOR},
                     {KernelObjectType::TUPLE_UNFOLD});

  auto input_node1 = make_tuple->input(1);
  SetKernelBuildInfo(input_node1, {"NCHW"}, {kNumberTypeFloat32}, {"NCHW"}, {kNumberTypeFloat32},
                     {KernelObjectType::TENSOR}, {KernelObjectType::TENSOR});

  auto input_node2 = make_tuple->input(2);
  SetKernelBuildInfo(input_node2, {"NCHW"}, {kNumberTypeFloat32}, {"NCHW"}, {kNumberTypeFloat32},
                     {KernelObjectType::TENSOR}, {KernelObjectType::TENSOR});
}

void TestInsertTypeTransformOp::SetTupleToTensorKernelBuildInfo(const FuncGraphPtr &g, AnfNodePtr *reshape_ptr) {
  auto ret = g->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  auto reshape = ret->input(1)->cast<CNodePtr>();
  *reshape_ptr = reshape;
  SetKernelBuildInfo(reshape, {"NCHW", "NCHW"}, {kNumberTypeFloat32, kNumberTypeFloat32}, {"NCHW"},
                     {kNumberTypeFloat32}, {KernelObjectType::TENSOR, KernelObjectType::TENSOR},
                     {KernelObjectType::TENSOR});
  auto input2 = reshape->input(2);
  SetKernelBuildInfo(input2, {"NCHW"}, {kNumberTypeFloat32}, {"NCHW"}, {kNumberTypeFloat32}, {KernelObjectType::TENSOR},
                     {KernelObjectType::TUPLE});

  auto input1 = reshape->input(1);
  SetKernelBuildInfo(input1, {"NCHW"}, {kNumberTypeFloat32}, {"NCHW"}, {kNumberTypeFloat32}, {KernelObjectType::TENSOR},
                     {KernelObjectType::TENSOR});
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

void TestInsertTypeTransformOp::CheckInputKernelInfo(const AnfNodePtr &node, size_t input_format_size,
                                                     size_t input_type_size, size_t input_obj_type_size) {
  MS_EXCEPTION_IF_NULL(node);
  KernelBuildInfoPtr kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(node);
  auto input_format = kernel_build_info->GetAllInputFormats();
  auto input_device_type = kernel_build_info->GetAllInputDeviceTypes();
  auto input_obj_type = kernel_build_info->GetAllInputKernelObjectTypes();
}

void TestInsertTypeTransformOp::CheckOutputKernelInfo(const AnfNodePtr &node, size_t output_format_size,
                                                      size_t output_type_size, size_t output_obj_type_size) {
  MS_EXCEPTION_IF_NULL(node);
  KernelBuildInfoPtr kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(node);
  auto output_format = kernel_build_info->GetAllOutputFormats();
  auto output_device_type = kernel_build_info->GetAllOutputDeviceTypes();
  auto output_obj_type = kernel_build_info->GetAllOutputKernelObjectTypes();
}

/// Feature: Dynamic shape.
/// Description: Test TupleUnfold to TupleUnfold type transforming pass.
/// Expectation: After InsertTypeTransformOp pass, the graph is identical to the expected graph
/// expressed by python.
TEST_F(TestInsertTypeTransformOp, test_tuple_unfold_to_tuple_unfold_transform) {
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_tuple_unfold_to_tuple_unfold_transform", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int64_t> shp_x{2, 4};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto func_graph = GetFuncGraph(g, args_spec_list);
  ASSERT_TRUE(func_graph != nullptr);

  AnfNodePtr split1, addn1, split2, addn2;
  SetTupleUnfoldToTupleUnfoldKernelBuildInfo(func_graph, &split1, &addn1, &split2, &addn2);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::InsertTypeTransformOp>());
  optimizer->AddPassManager(pm);
  optimizer->Optimize(func_graph);

  CheckOutputKernelInfo(split1, kSizeTwo, kSizeTwo, kSizeOne);
  CheckInputKernelInfo(addn1, kSizeTwo, kSizeTwo, kSizeOne);
  CheckOutputKernelInfo(split1, kSizeTwo, kSizeTwo, kSizeOne);
  CheckInputKernelInfo(addn2, kSizeTwo, kSizeTwo, kSizeOne);

  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_tuple_unfold_to_tuple_unfold_transform", "after");
  ASSERT_TRUE(g_after != nullptr);
  EXPECT_TRUE(CheckEqualGraph(func_graph, g_after));
}

/// Feature: Dynamic shape.
/// Description: Test TupleUnfold to Tuple type transforming pass.
/// Expectation: After InsertTypeTransformOp pass, the graph is identical to the expected graph
/// expressed by python.
TEST_F(TestInsertTypeTransformOp, test_tuple_unfold_to_tuple_transform) {
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_tuple_unfold_to_tuple_transform", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int64_t> shp_x{2, 4};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  std::vector<int64_t> shp_y{2, 4};
  auto y_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_y);
  AbstractBasePtrList args_spec_list{x_abstract, y_abstract};
  auto func_graph = GetFuncGraph(g, args_spec_list);
  ASSERT_TRUE(func_graph != nullptr);
  AnfNodePtr make_tuple, split, tuple_add1, tuple_add2;
  SetTupleUnfoldToTupleKernelBuildInfo(func_graph, &make_tuple, &split, &tuple_add1, &tuple_add2);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::InsertTypeTransformOp>());
  optimizer->AddPassManager(pm);
  optimizer->Optimize(func_graph);

  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_tuple_unfold_to_tuple_transform", "after");
  ASSERT_TRUE(g_after != nullptr);
  EXPECT_TRUE(CheckEqualGraph(func_graph, g_after));
}

/// Feature: Dynamic shape.
/// Description: Test TupleUnfold to Tensor type transforming pass.
/// Expectation: After InsertTypeTransformOp pass, the graph is identical to the expected graph
/// expressed by python.
TEST_F(TestInsertTypeTransformOp, test_tuple_unfold_to_tensor_transform) {
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_tuple_unfold_to_tensor_transform", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int64_t> shp_x{4};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  std::vector<int64_t> shp_y{2};
  auto y_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_y);
  std::vector<int64_t> shp_z{2, 4};
  auto z_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_z);
  AbstractBasePtrList args_spec_list{x_abstract, y_abstract, z_abstract};
  auto func_graph = GetFuncGraph(g, args_spec_list);
  ASSERT_TRUE(func_graph != nullptr);
  AnfNodePtr make_tuple, reshape;
  SetTupleUnfoldToTensorKernelBuildInfo(func_graph, &make_tuple, &reshape);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::InsertTypeTransformOp>());
  optimizer->AddPassManager(pm);
  optimizer->Optimize(func_graph);

  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_tuple_unfold_to_tensor_transform", "after");
  ASSERT_TRUE(g_after != nullptr);
  EXPECT_TRUE(CheckEqualGraph(func_graph, g_after));
}

/// Description: Test Tuple to Tensor type transforming pass.
/// Expectation: After InsertTypeTransformOp pass, the graph is identical to the expected graph expressed by python.
TEST_F(TestInsertTypeTransformOp, test_tuple_to_tensor_transform) {
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_tuple_to_tensor_transform", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int64_t> shp_x{4, 2};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  std::vector<int64_t> shp_y{2};
  auto y_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_y);
  AbstractBasePtrList args_spec_list{x_abstract, y_abstract};
  auto func_graph = GetFuncGraph(g, args_spec_list);
  ASSERT_TRUE(func_graph != nullptr);
  AnfNodePtr reshape;
  SetTupleToTensorKernelBuildInfo(func_graph, &reshape);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::InsertTypeTransformOp>());
  optimizer->AddPassManager(pm);
  optimizer->Optimize(func_graph);

  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_tuple_to_tensor_transform", "after");
  ASSERT_TRUE(g_after != nullptr);
  EXPECT_TRUE(CheckEqualGraph(func_graph, g_after));
}
}  // namespace opt
}  // namespace mindspore
