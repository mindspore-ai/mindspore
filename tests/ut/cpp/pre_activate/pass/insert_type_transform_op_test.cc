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
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/pass_manager.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/kernel_build_info.h"

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
  void SetTupleUnfoldToTupleKernelBuildInfo(const FuncGraphPtr &g, AnfNodePtr *make_tuple_ptr,
                                            AnfNodePtr *seq_add1_ptr);
  void SetTupleUnfoldToTensorKernelBuildInfo(const FuncGraphPtr &g, AnfNodePtr *make_tuple, AnfNodePtr *reshape_ptr);
  void SetTupleToTupleUnfoldKernelBuildInfo(const FuncGraphPtr &g, AnfNodePtr *shape_ptr);
  void SetTupleToTensorKernelBuildInfo(const FuncGraphPtr &g, AnfNodePtr *reshape_ptr);
  void SetScalarToTensorKernelBuildInfo(const FuncGraphPtr &g, AnfNodePtr *add_ptr);
  void SetTensorToTupleKernelBuildInfo(const FuncGraphPtr &g, AnfNodePtr *seq_add_ptr);
  void SetTensorToScalarKernelBuildInfo(const FuncGraphPtr &g, AnfNodePtr *scalar_to_tensor_ptr);

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
                                                                     AnfNodePtr *seq_add1_ptr) {
  auto ret = g->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  auto seq_add1 = ret->input(1)->cast<CNodePtr>();
  *seq_add1_ptr = seq_add1;
  SetKernelBuildInfo(seq_add1, {"NCHW", "NCHW"}, {kNumberTypeFloat32, kNumberTypeFloat32}, {"NCHW"},
                     {kNumberTypeFloat32}, {KernelObjectType::TUPLE, KernelObjectType::TUPLE},
                     {KernelObjectType::TENSOR});

  auto input_x = seq_add1->input(2);
  SetKernelBuildInfo(input_x, {"NCHW"}, {kNumberTypeFloat32}, {"NCHW"}, {kNumberTypeFloat32}, {KernelObjectType::TUPLE},
                     {KernelObjectType::TUPLE});

  auto make_tuple = seq_add1->input(1)->cast<CNodePtr>();
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

void TestInsertTypeTransformOp::SetTupleToTupleUnfoldKernelBuildInfo(const FuncGraphPtr &g, AnfNodePtr *shape_ptr) {
  auto ret = g->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  auto tuple_get_item = ret->input(1)->cast<CNodePtr>();
  SetKernelBuildInfo(tuple_get_item, {"NCHW", "NCHW"}, {kNumberTypeFloat32, kNumberTypeInt64}, {"NCHW"},
                     {kNumberTypeFloat32}, {KernelObjectType::TUPLE_UNFOLD, KernelObjectType::SCALAR},
                     {KernelObjectType::SCALAR});

  auto shape = tuple_get_item->input(1)->cast<CNodePtr>();
  *shape_ptr = shape;
  SetKernelBuildInfo(shape, {"NCHW"}, {kNumberTypeFloat32}, {"NCHW"}, {kNumberTypeInt64}, {KernelObjectType::SCALAR},
                     {KernelObjectType::TUPLE});
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

void TestInsertTypeTransformOp::SetScalarToTensorKernelBuildInfo(const FuncGraphPtr &g, AnfNodePtr *add_ptr) {
  auto ret = g->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  auto add = ret->input(1)->cast<CNodePtr>();
  *add_ptr = add;
  SetKernelBuildInfo(add, {"NCHW", "NCHW"}, {kNumberTypeFloat32, kNumberTypeFloat32}, {"NCHW"}, {kNumberTypeFloat32},
                     {KernelObjectType::TENSOR, KernelObjectType::TENSOR}, {KernelObjectType::TENSOR});
}

void TestInsertTypeTransformOp::SetTensorToTupleKernelBuildInfo(const FuncGraphPtr &g, AnfNodePtr *seq_add_ptr) {
  auto ret = g->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  auto seq_add = ret->input(1)->cast<CNodePtr>();
  *seq_add_ptr = seq_add;
  SetKernelBuildInfo(seq_add, {"NCHW", "NCHW"}, {kNumberTypeFloat32, kNumberTypeFloat32}, {"NCHW"},
                     {kNumberTypeFloat32}, {KernelObjectType::TUPLE, KernelObjectType::TUPLE},
                     {KernelObjectType::TUPLE});
  auto input2 = seq_add->input(2);
  SetKernelBuildInfo(input2, {"NCHW"}, {kNumberTypeFloat32}, {"NCHW"}, {kNumberTypeFloat32}, {KernelObjectType::TENSOR},
                     {KernelObjectType::TENSOR});

  auto input1 = seq_add->input(1);
  SetKernelBuildInfo(input1, {"NCHW"}, {kNumberTypeFloat32}, {"NCHW"}, {kNumberTypeFloat32}, {KernelObjectType::TENSOR},
                     {KernelObjectType::TENSOR});
}

void TestInsertTypeTransformOp::SetTensorToScalarKernelBuildInfo(const FuncGraphPtr &g,
                                                                 AnfNodePtr *scalar_to_tensor_ptr) {
  auto ret = g->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  auto scalar_to_tensor = ret->input(1)->cast<CNodePtr>();
  *scalar_to_tensor_ptr = scalar_to_tensor;
  SetKernelBuildInfo(scalar_to_tensor, {"NCHW"}, {kNumberTypeFloat32}, {"NCHW"}, {kNumberTypeFloat32},
                     {KernelObjectType::SCALAR}, {KernelObjectType::TENSOR});
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
  std::vector<int64_t> shp_1{};
  auto abstract_1 = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_1);
  std::vector<int64_t> shp_2{};
  auto abstract_2 = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_2);
  std::vector<int64_t> shp_x{1, 3};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList abstract_list = {x_abstract};
  auto x_tuple_abs = std::make_shared<abstract::AbstractTuple>(abstract_list);
  AbstractBasePtrList args_spec_list{abstract_1, abstract_2, x_tuple_abs};
  auto func_graph = GetFuncGraph(g, args_spec_list);
  ASSERT_TRUE(func_graph != nullptr);
  AnfNodePtr make_tuple, seq_add1;
  SetTupleUnfoldToTupleKernelBuildInfo(func_graph, &make_tuple, &seq_add1);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::InsertTypeTransformOp>());
  optimizer->AddPassManager(pm);
  optimizer->Optimize(func_graph);

  auto real_make_tuple2 = func_graph->get_return()->input(1)->cast<CNodePtr>()->input(1)->cast<CNodePtr>();
  ASSERT_TRUE(IsPrimitiveCNode(real_make_tuple2, prim::kPrimRealMakeTuple));
  ASSERT_TRUE(real_make_tuple2->abstract()->isa<abstract::AbstractTuple>());
  auto obj_type = AnfAlgo::GetOutputKernelObjectType(real_make_tuple2, 0);
  ASSERT_TRUE(obj_type == KernelObjectType::TUPLE);

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
  std::vector<int64_t> shp_x{};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  std::vector<int64_t> shp_y{};
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

  auto tuple_to_tensor = func_graph->get_return()->input(1)->cast<CNodePtr>()->input(2)->cast<CNodePtr>();
  ASSERT_TRUE(IsPrimitiveCNode(tuple_to_tensor, prim::kPrimTupleToTensor));
  auto real_make_tuple = tuple_to_tensor->input(1);
  ASSERT_TRUE(IsPrimitiveCNode(real_make_tuple, prim::kPrimRealMakeTuple));
  ASSERT_TRUE(real_make_tuple->abstract()->isa<abstract::AbstractTuple>());
  auto obj_type = AnfAlgo::GetOutputKernelObjectType(real_make_tuple, 0);
  ASSERT_TRUE(obj_type == KernelObjectType::TUPLE);

  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_tuple_unfold_to_tensor_transform", "after");
  ASSERT_TRUE(g_after != nullptr);
  EXPECT_TRUE(CheckEqualGraph(func_graph, g_after));
}

/// Feature: Dynamic shape.
/// Description: Test Tuple to TupleUnfold type transforming pass.
/// Expectation: After InsertTypeTransformOp pass, the graph is identical to the expected graph expressed by python.
TEST_F(TestInsertTypeTransformOp, DISABLED_test_tuple_to_tuple_unfold_transform) {
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_tuple_to_tuple_unfold_transform", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int64_t> shp_x{4, 2};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto func_graph = GetFuncGraph(g, args_spec_list);
  ASSERT_TRUE(func_graph != nullptr);
  AnfNodePtr shape;
  SetTupleToTupleUnfoldKernelBuildInfo(func_graph, &shape);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::InsertTypeTransformOp>());
  optimizer->AddPassManager(pm);
  optimizer->Optimize(func_graph);

  auto ret = func_graph->get_return();
  auto real_tuple_get_item = ret->input(1)->cast<CNodePtr>();
  ASSERT_TRUE(IsPrimitiveCNode(real_tuple_get_item, prim::kPrimRealTupleGetItem));
  ASSERT_TRUE(real_tuple_get_item->abstract()->isa<abstract::AbstractScalar>());
  auto obj_type = AnfAlgo::GetOutputKernelObjectType(real_tuple_get_item, 0);
  ASSERT_TRUE(obj_type == KernelObjectType::SCALAR);

  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_tuple_to_tuple_unfold_transform", "after");
  ASSERT_TRUE(g_after != nullptr);
  EXPECT_TRUE(CheckEqualGraph(func_graph, g_after));
}

/// Feature: Dynamic shape.
/// Description: Test Tuple to Tensor type transforming pass.
/// Expectation: After InsertTypeTransformOp pass, the graph is identical to the expected graph expressed by python.
TEST_F(TestInsertTypeTransformOp, test_tuple_to_tensor_transform) {
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_tuple_to_tensor_transform", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int64_t> shp_x{4, 2};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  std::vector<int64_t> shp_y{1, 3};
  auto y_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_y);
  AbstractBasePtrList abstract_list = {y_abstract};
  auto y_tuple_abs = std::make_shared<abstract::AbstractTuple>(abstract_list);
  AbstractBasePtrList args_spec_list{x_abstract, y_tuple_abs};
  auto func_graph = GetFuncGraph(g, args_spec_list);
  ASSERT_TRUE(func_graph != nullptr);
  AnfNodePtr reshape;
  SetTupleToTensorKernelBuildInfo(func_graph, &reshape);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::InsertTypeTransformOp>());
  optimizer->AddPassManager(pm);
  optimizer->Optimize(func_graph);

  auto tuple_to_tensor = func_graph->get_return()->input(1)->cast<CNodePtr>()->input(2)->cast<CNodePtr>();
  ASSERT_TRUE(IsPrimitiveCNode(tuple_to_tensor, prim::kPrimTupleToTensor));
  auto dtype = common::AnfAlgo::GetNodeAttr<TypePtr>(tuple_to_tensor, kAttrDType);
  ASSERT_TRUE(dtype->type_id() == kNumberTypeFloat32);
  ASSERT_TRUE(tuple_to_tensor->abstract()->isa<abstract::AbstractTensor>());
  auto obj_type = AnfAlgo::GetOutputKernelObjectType(tuple_to_tensor, 0);
  ASSERT_TRUE(obj_type == KernelObjectType::TENSOR);

  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_tuple_to_tensor_transform", "after");
  ASSERT_TRUE(g_after != nullptr);
  EXPECT_TRUE(CheckEqualGraph(func_graph, g_after));
}

/// Feature: Dynamic shape.
/// Description: Test Scalar to Tensor type transforming pass.
/// Expectation: After InsertTypeTransformOp pass, the graph is identical to the expected graph expressed by python.
TEST_F(TestInsertTypeTransformOp, DISABLED_test_scalar_to_tensor_transform) {
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_scalar_to_tensor_transform", "before");
  ASSERT_TRUE(g != nullptr);
  auto x_abstract = std::make_shared<abstract::AbstractScalar>(3);
  auto y_abstract = std::make_shared<abstract::AbstractScalar>(4);
  AbstractBasePtrList args_spec_list{x_abstract, y_abstract};
  auto func_graph = GetFuncGraph(g, args_spec_list);
  ASSERT_TRUE(func_graph != nullptr);
  AnfNodePtr add;
  SetScalarToTensorKernelBuildInfo(func_graph, &add);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::InsertTypeTransformOp>());
  optimizer->AddPassManager(pm);
  optimizer->Optimize(func_graph);

  auto scalar_to_tensor1 = func_graph->get_return()->input(1)->cast<CNodePtr>()->input(1)->cast<CNodePtr>();
  ASSERT_TRUE(IsPrimitiveCNode(scalar_to_tensor1, prim::kPrimScalarToTensor));
  auto dtype = common::AnfAlgo::GetNodeAttr<TypePtr>(scalar_to_tensor1, kAttrDType);
  ASSERT_TRUE(dtype->type_id() == kNumberTypeInt64);
  ASSERT_TRUE(scalar_to_tensor1->abstract()->isa<abstract::AbstractTensor>());
  auto obj_type = AnfAlgo::GetOutputKernelObjectType(scalar_to_tensor1, 0);
  ASSERT_TRUE(obj_type == KernelObjectType::TENSOR);

  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_scalar_to_tensor_transform", "after");
  ASSERT_TRUE(g_after != nullptr);
  EXPECT_TRUE(CheckEqualGraph(func_graph, g_after));
}

/// Feature: Dynamic shape.
/// Description: Test Tensor to Tuple type transforming pass.
/// Expectation: After InsertTypeTransformOp pass, the graph is identical to the expected graph expressed by python.
TEST_F(TestInsertTypeTransformOp, test_tensor_to_tuple_transform) {
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_tensor_to_tuple_transform", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int64_t> shp_x{4};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  std::vector<int64_t> shp_y{4};
  auto y_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_y);
  AbstractBasePtrList args_spec_list{x_abstract, y_abstract};
  auto func_graph = GetFuncGraph(g, args_spec_list);
  ASSERT_TRUE(func_graph != nullptr);
  AnfNodePtr seq_add;
  SetTensorToTupleKernelBuildInfo(func_graph, &seq_add);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::InsertTypeTransformOp>());
  optimizer->AddPassManager(pm);
  optimizer->Optimize(func_graph);

  auto tensor_to_tuple1 = func_graph->get_return()->input(1)->cast<CNodePtr>()->input(1)->cast<CNodePtr>();
  ASSERT_TRUE(IsPrimitiveCNode(tensor_to_tuple1, prim::kPrimTensorToTuple));
  ASSERT_TRUE(tensor_to_tuple1->abstract()->isa<abstract::AbstractTuple>());
  auto obj_type = AnfAlgo::GetOutputKernelObjectType(tensor_to_tuple1, 0);
  ASSERT_TRUE(obj_type == KernelObjectType::TUPLE);

  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_tensor_to_tuple_transform", "after");
  ASSERT_TRUE(g_after != nullptr);
  EXPECT_TRUE(CheckEqualGraph(func_graph, g_after));
}

/// Feature: Dynamic shape.
/// Description: Test Tensor to Scalar type transforming pass.
/// Expectation: After InsertTypeTransformOp pass, the graph is identical to the expected graph expressed by python.
TEST_F(TestInsertTypeTransformOp, DISABLED_test_tensor_to_scalar_transform) {
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_tensor_to_scalar_transform", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int64_t> shp_x{4};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto func_graph = GetFuncGraph(g, args_spec_list);
  ASSERT_TRUE(func_graph != nullptr);
  AnfNodePtr scalar_to_tensor;
  SetTensorToScalarKernelBuildInfo(func_graph, &scalar_to_tensor);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::InsertTypeTransformOp>());
  optimizer->AddPassManager(pm);
  optimizer->Optimize(func_graph);

  auto tensor_to_scalar = func_graph->get_return()->input(1)->cast<CNodePtr>()->input(1)->cast<CNodePtr>();
  ASSERT_TRUE(IsPrimitiveCNode(tensor_to_scalar, prim::kPrimTensorToScalar));
  ASSERT_TRUE(tensor_to_scalar->abstract()->isa<abstract::AbstractScalar>());
  auto obj_type = AnfAlgo::GetOutputKernelObjectType(tensor_to_scalar, 0);
  ASSERT_TRUE(obj_type == KernelObjectType::SCALAR);

  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_tensor_to_scalar_transform", "after");
  ASSERT_TRUE(g_after != nullptr);
  EXPECT_TRUE(CheckEqualGraph(func_graph, g_after));
}
}  // namespace opt
}  // namespace mindspore
