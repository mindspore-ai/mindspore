/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "common/common_test.h"
#include "ir/param_info.h"
#include "frontend/operator/ops.h"
#include "include/backend/kernel_graph.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "mindspore/ccsrc/include/backend/kernel_info.h"
#include "mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_device_address.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace session {
using device::KernelInfo;
using KernelBuildInfoBuilder = kernel::KernelBuildInfo::KernelBuildInfoBuilder;
using AscendDeviceAddress = device::ascend::AscendDeviceAddress;

class AnfRuntimeAlgorithmTest : public UT::Common {
 public:
  AnfRuntimeAlgorithmTest() = default;
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(AnfRuntimeAlgorithmTest, VisitKernel) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  KernelWithIndex kernel_with_index;
  // test nullptr as input
  EXPECT_THROW(common::AnfAlgo::VisitKernel(nullptr, 0), std::runtime_error);
  // test value node as input
  ValueNodePtr value_node = NewValueNode(prim::kPrimAdd);
  kernel_with_index = common::AnfAlgo::VisitKernel(value_node, 0);
  EXPECT_NE(kernel_with_index.first->cast<ValueNodePtr>(), nullptr);
  EXPECT_EQ((kernel_with_index.first->cast<ValueNodePtr>()).get(), value_node.get());
  EXPECT_EQ(kernel_with_index.second, 0);
  // test parameter node as input
  ParameterPtr parameter_node = kernel_graph->add_parameter();
  kernel_with_index = common::AnfAlgo::VisitKernel(parameter_node, 0);
  EXPECT_NE(kernel_with_index.first->cast<ParameterPtr>(), nullptr);
  EXPECT_EQ((kernel_with_index.first->cast<ParameterPtr>()).get(), parameter_node.get());
  EXPECT_EQ(kernel_with_index.second, 0);
  // test cnode as input
  std::vector<AnfNodePtr> inputs{value_node};
  auto add = kernel_graph->NewCNode(inputs);
  kernel_with_index = common::AnfAlgo::VisitKernel(add, 0);
  EXPECT_NE(kernel_with_index.first->cast<CNodePtr>(), nullptr);
  EXPECT_EQ((kernel_with_index.first->cast<CNodePtr>()).get(), add.get());
  EXPECT_EQ(kernel_with_index.second, 0);
  // test maketuple node as input
  std::vector<AnfNodePtr> add_inputs{NewValueNode(prim::kPrimAdd)};
  auto add_second = kernel_graph->NewCNode(add_inputs);
  std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(prim::kPrimMakeTuple), add, add_second};
  auto make_tuple = kernel_graph->NewCNode(make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(make_tuple);
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract};
  make_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(args_spec_list));
  kernel_with_index = common::AnfAlgo::VisitKernel(make_tuple, 0);
  EXPECT_NE(kernel_with_index.first->cast<CNodePtr>(), nullptr);
  EXPECT_EQ((kernel_with_index.first->cast<CNodePtr>()).get(), add.get());
  EXPECT_EQ(kernel_with_index.second, 0);
  kernel_with_index = common::AnfAlgo::VisitKernel(make_tuple, 1);
  EXPECT_NE(kernel_with_index.first->cast<CNodePtr>(), nullptr);
  EXPECT_EQ((kernel_with_index.first->cast<CNodePtr>()).get(), add_second.get());
  EXPECT_EQ(kernel_with_index.second, 0);
  // test tuple get item node as input
  std::vector<AnfNodePtr> tuple_get_item_inputs{NewValueNode(prim::kPrimTupleGetItem), make_tuple,
                                                NewValueNode(static_cast<int64_t>(1))};
  auto tuple_get_item = kernel_graph->NewCNode(tuple_get_item_inputs);
  kernel_with_index = common::AnfAlgo::VisitKernel(tuple_get_item, 0);
  EXPECT_NE(kernel_with_index.first->cast<CNodePtr>(), nullptr);
  EXPECT_EQ((kernel_with_index.first->cast<CNodePtr>()).get(), add_second.get());
  EXPECT_EQ(kernel_with_index.second, 0);
  // test depend node as input
  std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), add, add_second};
  auto depend = kernel_graph->NewCNode(depend_inputs);
  kernel_with_index = common::AnfAlgo::VisitKernel(depend, 0);
  EXPECT_NE(kernel_with_index.first->cast<CNodePtr>(), nullptr);
  EXPECT_EQ((kernel_with_index.first->cast<CNodePtr>()).get(), add.get());
  EXPECT_EQ(kernel_with_index.second, 0);
}

TEST_F(AnfRuntimeAlgorithmTest, GetCNodePrimitive) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  // test cnode node
  PrimitivePtr add_primitive = prim::kPrimAdd;
  std::vector<AnfNodePtr> inputs{NewValueNode(add_primitive)};
  auto add = kernel_graph->NewCNode(inputs);
  EXPECT_NE(common::AnfAlgo::GetCNodePrimitive(add), nullptr);
  EXPECT_EQ(common::AnfAlgo::GetCNodePrimitive(add).get(), add_primitive.get());
  EXPECT_THROW(common::AnfAlgo::GetCNodePrimitive(nullptr), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, GetCNodeName) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  // test cnode node
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimAdd)};
  auto add = kernel_graph->NewCNode(inputs);
  EXPECT_EQ(common::AnfAlgo::GetCNodeName(add), prim::kPrimAdd->name());
  EXPECT_THROW(common::AnfAlgo::GetCNodeName(nullptr), std::runtime_error);
  // test parameter
  auto parameter_node = kernel_graph->add_parameter();
  EXPECT_THROW(common::AnfAlgo::GetCNodeName(parameter_node), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, GetNodeDebugString) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  // test cnode node
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimAdd)};
  auto add = kernel_graph->NewCNode(inputs);
  EXPECT_EQ(common::AnfAlgo::GetNodeDebugString(add), add->DebugString());
  EXPECT_THROW(common::AnfAlgo::GetNodeDebugString(nullptr), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, SetNodeAttr) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  // test cnode node
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimAdd)};
  auto add = kernel_graph->NewCNode(inputs);
  common::AnfAlgo::SetNodeAttr("test_set_attr", MakeValue("test_value"), add);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(add);
  MS_EXCEPTION_IF_NULL(primitive);
  EXPECT_EQ(GetValue<std::string>(primitive->GetAttr("test_set_attr")), "test_value");
  // test parameter node
  auto parameter = kernel_graph->add_parameter();
  EXPECT_THROW(common::AnfAlgo::SetNodeAttr("test_set_attr", MakeValue("test_value"), parameter), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, CopyNodeAttr) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  // test cnode node
  std::vector<AnfNodePtr> add_inputs{NewValueNode(prim::kPrimAdd)};
  auto add = kernel_graph->NewCNode(add_inputs);
  common::AnfAlgo::SetNodeAttr("test_set_attr", MakeValue("test_value"), add);

  std::vector<AnfNodePtr> mul_inputs{NewValueNode(prim::kPrimMul)};
  auto mul = kernel_graph->NewCNode(mul_inputs);
  common::AnfAlgo::SetNodeAttr("test_set_attr", MakeValue("test_value_v2"), mul);
  common::AnfAlgo::CopyNodeAttr("test_set_attr", mul, add);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(add);
  MS_EXCEPTION_IF_NULL(primitive);
  EXPECT_EQ(GetValue<std::string>(primitive->GetAttr("test_set_attr")), "test_value_v2");
  // test parameter node
  auto parameter = kernel_graph->add_parameter();
  EXPECT_THROW(common::AnfAlgo::CopyNodeAttr("test_set_attr", parameter, add), std::runtime_error);
  EXPECT_THROW(common::AnfAlgo::CopyNodeAttr("test_set_attr", mul, parameter), std::runtime_error);
  EXPECT_THROW(common::AnfAlgo::CopyNodeAttr("test_set_attr", parameter, parameter), std::runtime_error);
  EXPECT_THROW(common::AnfAlgo::CopyNodeAttr("test_set_attr", nullptr, add), std::runtime_error);
  EXPECT_THROW(common::AnfAlgo::CopyNodeAttr("test_set_attr", mul, nullptr), std::runtime_error);
  EXPECT_THROW(common::AnfAlgo::CopyNodeAttr("test_set_attr", nullptr, nullptr), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, CopyNodeAttrs) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  // test cnode node
  std::vector<AnfNodePtr> add_inputs{NewValueNode(prim::kPrimAdd)};
  auto add = kernel_graph->NewCNode(add_inputs);
  common::AnfAlgo::SetNodeAttr("test_set_attr", MakeValue("test_value"), add);

  std::vector<AnfNodePtr> mul_inputs{NewValueNode(prim::kPrimMul)};
  auto mul = kernel_graph->NewCNode(mul_inputs);
  common::AnfAlgo::SetNodeAttr("test_set_attr", MakeValue("test_value_v2"), mul);
  common::AnfAlgo::CopyNodeAttrs(mul, add);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(add);
  MS_EXCEPTION_IF_NULL(primitive);
  EXPECT_EQ(GetValue<std::string>(primitive->GetAttr("test_set_attr")), "test_value_v2");
  // test parameter node
  auto parameter = kernel_graph->add_parameter();
  EXPECT_THROW(common::AnfAlgo::CopyNodeAttrs(parameter, add), std::runtime_error);
  EXPECT_THROW(common::AnfAlgo::CopyNodeAttrs(mul, parameter), std::runtime_error);
  EXPECT_THROW(common::AnfAlgo::CopyNodeAttrs(parameter, parameter), std::runtime_error);
  EXPECT_THROW(common::AnfAlgo::CopyNodeAttrs(nullptr, add), std::runtime_error);
  EXPECT_THROW(common::AnfAlgo::CopyNodeAttrs(mul, nullptr), std::runtime_error);
  EXPECT_THROW(common::AnfAlgo::CopyNodeAttrs(nullptr, nullptr), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, EraseNodeAttr) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  // test cnode node
  std::vector<AnfNodePtr> add_inputs{NewValueNode(prim::kPrimAdd)};
  auto add = kernel_graph->NewCNode(add_inputs);
  common::AnfAlgo::SetNodeAttr("test_set_attr", MakeValue("test_value"), add);
  common::AnfAlgo::SetNodeAttr("test_set_attr_v2", MakeValue("test_value_v2"), add);
  common::AnfAlgo::EraseNodeAttr("test_set_attr_v2", add);
  EXPECT_THROW(common::AnfAlgo::GetNodeAttr<std::string>(add, "test_set_attr_v2"), std::runtime_error);
  EXPECT_THROW(common::AnfAlgo::EraseNodeAttr("test_set_attr_v2", nullptr), std::runtime_error);
  // test parameter node
  auto parameter = kernel_graph->add_parameter();
  EXPECT_THROW(common::AnfAlgo::EraseNodeAttr("test_set_attr_v2", parameter), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, GetInputTensorNum) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  // test cnode node
  auto parameter_one = kernel_graph->NewParameter();
  auto parameter_two = kernel_graph->NewParameter();
  std::vector<AnfNodePtr> add_inputs{NewValueNode(prim::kPrimAdd), parameter_one, parameter_two};
  auto add = kernel_graph->NewCNode(add_inputs);
  EXPECT_EQ(common::AnfAlgo::GetInputTensorNum(add), 2);
  EXPECT_THROW(common::AnfAlgo::GetInputTensorNum(nullptr), std::runtime_error);
  // test parameter node
  EXPECT_THROW(common::AnfAlgo::GetInputTensorNum(parameter_one), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, GetOutputTensorNum) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;
  // test fused batch norm as input
  inputs.push_back(NewValueNode(prim::kPrimBatchNorm));
  auto bn = kernel_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(bn);
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract, x_abstract, x_abstract, x_abstract};
  bn->set_abstract(std::make_shared<abstract::AbstractTuple>(args_spec_list));
  KernelBuildInfoBuilder builder;
  builder.SetOutputsFormat(
    {kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT});
  builder.SetOutputsDeviceType(
    {kFloat32->type_id(), kFloat32->type_id(), kFloat32->type_id(), kFloat32->type_id(), kFloat32->type_id()});
  builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TUPLE_UNFOLD});
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), bn.get());
  EXPECT_EQ(AnfAlgo::GetOutputTensorNum(bn), 5);
  EXPECT_THROW(AnfAlgo::GetOutputTensorNum(nullptr), std::runtime_error);
  // test add as input
  inputs.clear();
  inputs.push_back(NewValueNode(prim::kPrimAdd));
  auto add = kernel_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(add);
  add->set_abstract(std::make_shared<abstract::AbstractNone>());
  EXPECT_EQ(AnfAlgo::GetOutputTensorNum(add), 0);
  add->set_abstract(x_abstract);
  EXPECT_EQ(AnfAlgo::GetOutputTensorNum(add), 1);
}

TEST_F(AnfRuntimeAlgorithmTest, GetOutputFormat) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim::kPrimAdd), kernel_graph->NewParameter(),
                                    kernel_graph->NewParameter()};
  auto add = kernel_graph->NewCNode(inputs);
  ShapeVector shape = {1, 2, 3, 4};
  common::AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat32, kNumberTypeFloat32}, {shape, shape}, add.get());
  MS_EXCEPTION_IF_NULL(add);
  add->set_kernel_info(std::make_shared<KernelInfo>());
  auto d_kernel_info = dynamic_cast<KernelInfo *>(add->kernel_info());
  MS_EXCEPTION_IF_NULL(d_kernel_info);
  KernelBuildInfoBuilder builder;
  builder.SetOutputsDeviceType({kFloat32->type_id(), kFloat16->type_id()});
  builder.SetOutputsFormat({kOpFormat_NCHW, kOpFormat_NC1HWC0});
  d_kernel_info->set_select_kernel_build_info(builder.Build());
  EXPECT_EQ(AnfAlgo::GetOutputFormat(add, 0), kOpFormat_NCHW);
  EXPECT_EQ(AnfAlgo::GetOutputFormat(add, 1), kOpFormat_NC1HWC0);
  EXPECT_THROW(AnfAlgo::GetOutputFormat(add, 2), std::runtime_error);
  EXPECT_THROW(AnfAlgo::GetOutputFormat(nullptr, 0), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, GetInputFormat) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim::kPrimAdd), kernel_graph->NewParameter(),
                                    kernel_graph->NewParameter()};
  auto add = kernel_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(add);
  add->set_kernel_info(std::make_shared<KernelInfo>());
  auto d_kernel_info = dynamic_cast<KernelInfo *>(add->kernel_info());
  MS_EXCEPTION_IF_NULL(d_kernel_info);
  KernelBuildInfoBuilder builder;
  builder.SetInputsDeviceType({kFloat32->type_id(), kFloat16->type_id()});
  builder.SetInputsFormat({kOpFormat_NCHW, kOpFormat_NC1HWC0});
  d_kernel_info->set_select_kernel_build_info(builder.Build());
  EXPECT_EQ(AnfAlgo::GetInputFormat(add, 0), kOpFormat_NCHW);
  EXPECT_EQ(AnfAlgo::GetInputFormat(add, 1), kOpFormat_NC1HWC0);
  EXPECT_THROW(AnfAlgo::GetInputFormat(add, 2), std::runtime_error);
  EXPECT_THROW(AnfAlgo::GetInputFormat(nullptr, 0), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, GetPrevNodeOutputFormat) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> pre_node_inputs;
  pre_node_inputs.push_back(NewValueNode(prim::kPrimAdd));
  auto pre_add = kernel_graph->NewCNode(pre_node_inputs);
  MS_EXCEPTION_IF_NULL(pre_add);
  pre_add->set_kernel_info(std::make_shared<KernelInfo>());
  auto d_kernel_info = dynamic_cast<KernelInfo *>(pre_add->kernel_info());
  MS_EXCEPTION_IF_NULL(d_kernel_info);
  KernelBuildInfoBuilder builder;
  builder.SetOutputsDeviceType({kFloat32->type_id()});
  builder.SetOutputsFormat({kOpFormat_NCHW});
  d_kernel_info->set_select_kernel_build_info(builder.Build());
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimAdd), pre_add};
  auto add = kernel_graph->NewCNode(inputs);
  EXPECT_EQ(AnfAlgo::GetPrevNodeOutputFormat(add, 0), kOpFormat_NCHW);
  EXPECT_THROW(AnfAlgo::GetPrevNodeOutputFormat(nullptr, 0), std::runtime_error);
  // test parameter node as input
  auto parameter_node = kernel_graph->add_parameter();
  EXPECT_THROW(AnfAlgo::GetPrevNodeOutputFormat(parameter_node, 0), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, GetOutputInferShape) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto none_abstract = std::make_shared<abstract::AbstractNone>();
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list{x_abstract, none_abstract, x_abstract};
  auto tuple_abstract = std::make_shared<abstract::AbstractTuple>(args_spec_list);
  // test value node as input
  auto value_node = NewValueNode(prim::kPrimAdd);
  MS_EXCEPTION_IF_NULL(value_node);
  value_node->set_abstract(x_abstract);
  EXPECT_EQ(common::AnfAlgo::GetOutputInferShape(value_node, 0)[1], 32);
  EXPECT_THROW(common::AnfAlgo::GetOutputInferShape(nullptr, 0), std::runtime_error);
  // test parameter node as input
  auto parameter_node = kernel_graph->add_parameter();
  MS_EXCEPTION_IF_NULL(parameter_node);
  parameter_node->set_abstract(x_abstract);
  EXPECT_EQ(common::AnfAlgo::GetOutputInferShape(parameter_node, 0)[2], 224);
  // test cnode as input
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimAdd));
  auto add = kernel_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(add);
  add->set_abstract(std::make_shared<abstract::AbstractNone>());
  EXPECT_TRUE(common::AnfAlgo::GetOutputInferShape(add, 0).empty());
  add->set_abstract(x_abstract);
  EXPECT_EQ(common::AnfAlgo::GetOutputInferShape(add, 0)[3], 224);
  EXPECT_THROW(common::AnfAlgo::GetOutputInferShape(add, 1), std::runtime_error);
  add->set_abstract(tuple_abstract);
  EXPECT_EQ(common::AnfAlgo::GetOutputInferShape(add, 0)[0], 2);
  EXPECT_TRUE(common::AnfAlgo::GetOutputInferShape(add, 1).empty());
  EXPECT_EQ(common::AnfAlgo::GetOutputInferShape(add, 2)[1], 32);
  EXPECT_THROW(common::AnfAlgo::GetOutputInferShape(add, 3), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, GetPrevNodeOutputInferShape) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  // test parameter node as input
  auto parameter_node = kernel_graph->NewParameter();
  MS_EXCEPTION_IF_NULL(parameter_node);
  parameter_node->set_abstract(x_abstract);
  EXPECT_THROW(common::AnfAlgo::GetPrevNodeOutputInferShape(parameter_node, 0), std::runtime_error);
  // test cnode as input
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimAdd), parameter_node};
  auto add = kernel_graph->NewCNode(inputs);
  EXPECT_EQ(common::AnfAlgo::GetPrevNodeOutputInferShape(add, 0)[1], 32);
  EXPECT_THROW(common::AnfAlgo::GetPrevNodeOutputInferShape(add, 1), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, GetOutputDeviceShape) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract, x_abstract};
  args_spec_list.emplace_back(std::make_shared<abstract::AbstractTensor>(kFloat32, std::vector<int64_t>{1, 2, 3, 4}));
  auto tuple_abstract = std::make_shared<abstract::AbstractTuple>(args_spec_list);
  // test cnode as input
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimAdd));
  auto add = kernel_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(add);
  add->set_abstract(tuple_abstract);
  add->set_kernel_info(std::make_shared<KernelInfo>());
  auto d_kernel_info = dynamic_cast<KernelInfo *>(add->kernel_info());
  MS_EXCEPTION_IF_NULL(d_kernel_info);
  KernelBuildInfoBuilder builder;
  builder.SetOutputsFormat({kOpFormat_NCHW, kOpFormat_NCHW, kOpFormat_NHWC, kOpFormat_FRAC_NZ});
  builder.SetOutputsDeviceType({kFloat32->type_id(), kFloat32->type_id(), kFloat32->type_id(), kFloat32->type_id()});
  d_kernel_info->set_select_kernel_build_info(builder.Build());
  EXPECT_EQ(AnfAlgo::GetOutputDeviceShape(add, 0)[2], 224);
  EXPECT_EQ(AnfAlgo::GetOutputDeviceShape(add, 1)[0], 2);
  ShapeVector expect_shape{2, 224, 224, 32};
  EXPECT_EQ(AnfAlgo::GetOutputDeviceShape(add, 2), expect_shape);
  ShapeVector nz_expect_shape{1, 2, 1, 1, 16, 16};
  EXPECT_EQ(AnfAlgo::GetOutputDeviceShape(add, 3), nz_expect_shape);
}

TEST_F(AnfRuntimeAlgorithmTest, GetInputDeviceShape) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  auto parameter_one = kernel_graph->NewParameter();
  MS_EXCEPTION_IF_NULL(parameter_one);
  parameter_one->set_abstract(x_abstract);
  auto parameter_two = kernel_graph->NewParameter();
  MS_EXCEPTION_IF_NULL(parameter_two);
  parameter_two->set_abstract(x_abstract);
  auto parameter_third = kernel_graph->NewParameter();
  MS_EXCEPTION_IF_NULL(parameter_third);
  parameter_third->set_abstract(x_abstract);
  // test cnode as input
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimAdd), parameter_one, parameter_two, parameter_third};
  auto add = kernel_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(add);
  add->set_kernel_info(std::make_shared<KernelInfo>());
  auto d_kernel_info = dynamic_cast<KernelInfo *>(add->kernel_info());
  MS_EXCEPTION_IF_NULL(d_kernel_info);
  KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({kOpFormat_NCHW, kOpFormat_NCHW, kOpFormat_NHWC});
  builder.SetInputsDeviceType({kFloat32->type_id(), kFloat32->type_id(), kFloat32->type_id()});
  d_kernel_info->set_select_kernel_build_info(builder.Build());
  EXPECT_EQ(AnfAlgo::GetInputDeviceShape(add, 0)[2], 224);
  EXPECT_EQ(AnfAlgo::GetInputDeviceShape(add, 1)[1], 32);
  ShapeVector expect_shape{2, 224, 224, 32};
  EXPECT_EQ(AnfAlgo::GetInputDeviceShape(add, 2), expect_shape);
  EXPECT_THROW(common::AnfAlgo::GetPrevNodeOutputInferShape(nullptr, 0), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, GetOutputInferDataTypeTest) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimBatchNorm));
  auto bn = kernel_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(bn);
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract, x_abstract, x_abstract, x_abstract};
  bn->set_abstract(std::make_shared<abstract::AbstractTuple>(args_spec_list));
  EXPECT_EQ(common::AnfAlgo::GetOutputInferDataType(bn, 0), kFloat32->type_id());
  EXPECT_EQ(common::AnfAlgo::GetOutputInferDataType(bn, 4), kFloat32->type_id());
  EXPECT_THROW(common::AnfAlgo::GetOutputInferDataType(bn, 5), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, GetPrevNodeOutputInferDataType) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> pre_node_inputs;
  pre_node_inputs.push_back(NewValueNode(prim::kPrimAdd));
  auto pre_add = kernel_graph->NewCNode(pre_node_inputs);
  MS_EXCEPTION_IF_NULL(pre_add);
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  pre_add->set_abstract(x_abstract);
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimAdd), pre_add};
  auto add = kernel_graph->NewCNode(inputs);
  EXPECT_EQ(common::AnfAlgo::GetPrevNodeOutputInferDataType(add, 0), kFloat32->type_id());
  EXPECT_THROW(common::AnfAlgo::GetPrevNodeOutputInferDataType(add, 1), std::runtime_error);
  EXPECT_THROW(common::AnfAlgo::GetPrevNodeOutputInferDataType(nullptr, 0), std::runtime_error);
  // test parameter as input
  auto parameter_node = kernel_graph->add_parameter();
  EXPECT_THROW(common::AnfAlgo::GetPrevNodeOutputInferDataType(parameter_node, 0), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, GetOutputDeviceDataTypeTest) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimAdd));
  auto add = kernel_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(add);
  add->set_kernel_info(std::make_shared<KernelInfo>());
  auto d_kernel_info = dynamic_cast<KernelInfo *>(add->kernel_info());
  MS_EXCEPTION_IF_NULL(d_kernel_info);
  KernelBuildInfoBuilder builder;
  builder.SetOutputsDeviceType({kFloat32->type_id()});
  builder.SetOutputsFormat({kOpFormat_NCHW});
  d_kernel_info->set_select_kernel_build_info(builder.Build());
  EXPECT_EQ(AnfAlgo::GetOutputDeviceDataType(add, 0), kFloat32->type_id());
  EXPECT_THROW(AnfAlgo::GetOutputDeviceDataType(add, 1), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, GetInputDeviceDataTypeTest) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim::kPrimAdd), kernel_graph->NewParameter(),
                                    kernel_graph->NewParameter()};
  auto add = kernel_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(add);
  add->set_kernel_info(std::make_shared<KernelInfo>());
  auto d_kernel_info = dynamic_cast<KernelInfo *>(add->kernel_info());
  MS_EXCEPTION_IF_NULL(d_kernel_info);
  KernelBuildInfoBuilder builder;
  builder.SetInputsDeviceType({kFloat32->type_id(), kFloat16->type_id()});
  builder.SetInputsFormat({kOpFormat_NCHW, kOpFormat_NC1HWC0});
  d_kernel_info->set_select_kernel_build_info(builder.Build());
  EXPECT_EQ(AnfAlgo::GetInputDeviceDataType(add, 0), kFloat32->type_id());
  EXPECT_EQ(AnfAlgo::GetInputDeviceDataType(add, 1), kFloat16->type_id());
  EXPECT_THROW(AnfAlgo::GetInputDeviceDataType(add, 2), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, GetPrevNodeOutputDeviceDataType) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> pre_add_inputs;
  pre_add_inputs.push_back(NewValueNode(prim::kPrimAdd));
  auto pre_add = kernel_graph->NewCNode(pre_add_inputs);
  MS_EXCEPTION_IF_NULL(pre_add);
  pre_add->set_kernel_info(std::make_shared<KernelInfo>());
  auto d_kernel_info = dynamic_cast<KernelInfo *>(pre_add->kernel_info());
  MS_EXCEPTION_IF_NULL(d_kernel_info);
  KernelBuildInfoBuilder builder;
  builder.SetOutputsDeviceType({kFloat32->type_id()});
  d_kernel_info->set_select_kernel_build_info(builder.Build());
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimAdd), pre_add};
  auto add = kernel_graph->NewCNode(inputs);
  EXPECT_EQ(AnfAlgo::GetPrevNodeOutputDeviceDataType(add, 0), kFloat32->type_id());
  EXPECT_THROW(AnfAlgo::GetPrevNodeOutputDeviceDataType(add, 1), std::runtime_error);
  // test parameter as input
  auto parameter_node = kernel_graph->add_parameter();
  EXPECT_THROW(AnfAlgo::GetPrevNodeOutputDeviceDataType(parameter_node, 0), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, GetOutputAddr) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimAdd));
  auto add = kernel_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(add);
  add->set_kernel_info(std::make_shared<KernelInfo>());
  auto d_kernel_info = dynamic_cast<KernelInfo *>(add->kernel_info());
  MS_EXCEPTION_IF_NULL(d_kernel_info);
  int *addr = nullptr;
  auto device_address = std::make_shared<AscendDeviceAddress>(addr, 1);
  d_kernel_info->SetOutputAddr(device_address, 0);
  EXPECT_EQ(AnfAlgo::GetOutputAddr(add, 0), device_address.get());
}

TEST_F(AnfRuntimeAlgorithmTest, GetPrevNodeOutputAddr) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> pre_add_inputs;
  pre_add_inputs.push_back(NewValueNode(prim::kPrimAdd));
  auto pre_add = kernel_graph->NewCNode(pre_add_inputs);
  MS_EXCEPTION_IF_NULL(pre_add);
  pre_add->set_kernel_info(std::make_shared<KernelInfo>());
  auto d_kernel_info = dynamic_cast<KernelInfo *>(pre_add->kernel_info());
  MS_EXCEPTION_IF_NULL(d_kernel_info);
  int *addr = nullptr;
  auto device_address = std::make_shared<AscendDeviceAddress>(addr, 1);
  d_kernel_info->SetOutputAddr(device_address, 0);
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimAdd), pre_add};
  auto add = kernel_graph->NewCNode(inputs);
  EXPECT_EQ(AnfAlgo::GetPrevNodeOutputAddr(add, 0), device_address.get());
  EXPECT_THROW(AnfAlgo::GetPrevNodeOutputAddr(add, 1), std::runtime_error);
  // test parameter as input
  auto parameter_node = kernel_graph->add_parameter();
  EXPECT_THROW(AnfAlgo::GetPrevNodeOutputAddr(parameter_node, 0), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, SetOutputAddr) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimAdd));
  auto add = kernel_graph->NewCNode(inputs);
  int *addr = nullptr;
  auto device_address = std::make_shared<AscendDeviceAddress>(addr, 1);
  EXPECT_THROW(AnfAlgo::SetOutputAddr(device_address, 0, nullptr), std::runtime_error);
  AnfAlgo::SetOutputAddr(device_address, 0, add.get());
  EXPECT_EQ(AnfAlgo::GetOutputAddr(add, 0), device_address.get());
}

TEST_F(AnfRuntimeAlgorithmTest, GetWorkspaceAddr) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimAdd));
  auto add = kernel_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(add);
  add->set_kernel_info(std::make_shared<KernelInfo>());
  auto d_kernel_info = dynamic_cast<KernelInfo *>(add->kernel_info());
  MS_EXCEPTION_IF_NULL(d_kernel_info);
  int *addr = nullptr;
  auto device_address = std::make_shared<AscendDeviceAddress>(addr, 1);
  d_kernel_info->SetWorkspaceAddr(device_address, 0);
  EXPECT_EQ(AnfAlgo::GetWorkspaceAddr(add, 0), device_address.get());
}

TEST_F(AnfRuntimeAlgorithmTest, SetWorkspaceAddr) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimAdd));
  auto add = kernel_graph->NewCNode(inputs);
  int *addr = nullptr;
  auto device_address = std::make_shared<AscendDeviceAddress>(addr, 1);
  EXPECT_THROW(AnfAlgo::SetWorkspaceAddr(device_address, 0, nullptr), std::runtime_error);
  AnfAlgo::SetWorkspaceAddr(device_address, 0, add.get());
  EXPECT_EQ(AnfAlgo::GetWorkspaceAddr(add, 0), device_address.get());
}

TEST_F(AnfRuntimeAlgorithmTest, SetOutputInferTypeAndShape) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimAdd));
  auto add = kernel_graph->NewCNode(inputs);
  // set none abstract
  std::vector<TypeId> none_types = {};
  std::vector<ShapeVector> none_shapes = {};
  EXPECT_THROW(common::AnfAlgo::SetOutputInferTypeAndShape(none_types, none_shapes, nullptr), std::runtime_error);
  common::AnfAlgo::SetOutputInferTypeAndShape(none_types, none_shapes, add.get());
  EXPECT_EQ((*add->abstract()), abstract::AbstractNone());
  // set single input
  std::vector<TypeId> single_types = {kFloat32->type_id()};
  std::vector<ShapeVector> single_shapes = {{2, 32, 224, 224}};
  EXPECT_THROW(common::AnfAlgo::SetOutputInferTypeAndShape(none_types, single_shapes, add.get()), std::runtime_error);
  common::AnfAlgo::SetOutputInferTypeAndShape(single_types, single_shapes, add.get());
  EXPECT_EQ(common::AnfAlgo::GetOutputInferDataType(add, 0), kFloat32->type_id());
  EXPECT_EQ(common::AnfAlgo::GetOutputInferShape(add, 0).size(), 4);
  // set multiple input
  std::vector<TypeId> mutiple_types = {kFloat16->type_id(), kFloat32->type_id(), kFloat64->type_id()};
  std::vector<ShapeVector> mutiple_shapes = {{2, 32, 224, 224}, {2, 32, 224, 224}, {2, 32, 224, 224}};
  common::AnfAlgo::SetOutputInferTypeAndShape(mutiple_types, mutiple_shapes, add.get());
  EXPECT_EQ(common::AnfAlgo::GetOutputInferDataType(add, 0), kFloat16->type_id());
  EXPECT_EQ(common::AnfAlgo::GetOutputInferDataType(add, 1), kFloat32->type_id());
  EXPECT_EQ(common::AnfAlgo::GetOutputInferDataType(add, 2), kFloat64->type_id());
  EXPECT_EQ(common::AnfAlgo::GetOutputInferShape(add, 0).size(), 4);
  EXPECT_EQ(common::AnfAlgo::GetOutputInferShape(add, 1).size(), 4);
  EXPECT_EQ(common::AnfAlgo::GetOutputInferShape(add, 2).size(), 4);
}

TEST_F(AnfRuntimeAlgorithmTest, CopyAbstract) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> first_inputs;
  first_inputs.push_back(NewValueNode(prim::kPrimAdd));
  auto first_add = kernel_graph->NewCNode(first_inputs);
  // set single input
  std::vector<TypeId> single_types = {kFloat32->type_id()};
  std::vector<ShapeVector> single_shapes = {{2, 32, 224, 224}};
  common::AnfAlgo::SetOutputInferTypeAndShape(single_types, single_shapes, first_add.get());
  // set multiple input
  std::vector<AnfNodePtr> second_inputs;
  second_inputs.push_back(NewValueNode(prim::kPrimAdd));
  auto second_add = kernel_graph->NewCNode(second_inputs);
  std::vector<TypeId> mutiple_types = {kFloat16->type_id(), kFloat32->type_id(), kFloat64->type_id()};
  std::vector<ShapeVector> mutiple_shapes = {{2, 32, 224, 224}, {2, 32, 224, 224}, {2, 32, 224, 224}};
  common::AnfAlgo::SetOutputInferTypeAndShape(mutiple_types, mutiple_shapes, second_add.get());
  common::AnfAlgo::CopyAbstract(second_add, first_add.get());
  EXPECT_EQ(common::AnfAlgo::GetOutputInferDataType(first_add, 0), kFloat16->type_id());
  EXPECT_EQ(common::AnfAlgo::GetOutputInferDataType(first_add, 1), kFloat32->type_id());
  EXPECT_EQ(common::AnfAlgo::GetOutputInferDataType(first_add, 2), kFloat64->type_id());
  EXPECT_EQ(common::AnfAlgo::GetOutputInferShape(first_add, 0).size(), 4);
  EXPECT_EQ(common::AnfAlgo::GetOutputInferShape(first_add, 1).size(), 4);
  EXPECT_EQ(common::AnfAlgo::GetOutputInferShape(first_add, 2).size(), 4);
}

TEST_F(AnfRuntimeAlgorithmTest, GetKernelType) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimAdd));
  auto add = kernel_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(add);
  add->set_kernel_info(std::make_shared<KernelInfo>());
  auto d_kernel_info = dynamic_cast<KernelInfo *>(add->kernel_info());
  MS_EXCEPTION_IF_NULL(d_kernel_info);
  KernelBuildInfoBuilder builder;
  builder.SetKernelType(AKG_KERNEL);
  d_kernel_info->set_select_kernel_build_info(builder.Build());
  EXPECT_EQ(AnfAlgo::GetKernelType(add), AKG_KERNEL);
  EXPECT_THROW(AnfAlgo::GetKernelType(nullptr), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, GetProcessor) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimAdd));
  auto add = kernel_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(add);
  add->set_kernel_info(std::make_shared<KernelInfo>());
  auto d_kernel_info = dynamic_cast<KernelInfo *>(add->kernel_info());
  MS_EXCEPTION_IF_NULL(d_kernel_info);
  KernelBuildInfoBuilder builder;
  builder.SetProcessor(kernel::AICORE);
  d_kernel_info->set_select_kernel_build_info(builder.Build());
  EXPECT_EQ(AnfAlgo::GetProcessor(add), kernel::AICORE);
  EXPECT_THROW(AnfAlgo::GetProcessor(nullptr), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, GetFusionType) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimAdd));
  auto add = kernel_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(add);
  add->set_kernel_info(std::make_shared<KernelInfo>());
  auto d_kernel_info = dynamic_cast<KernelInfo *>(add->kernel_info());
  MS_EXCEPTION_IF_NULL(d_kernel_info);
  KernelBuildInfoBuilder builder;
  builder.SetFusionType(kernel::kPatternConvolution);
  d_kernel_info->set_select_kernel_build_info(builder.Build());
  EXPECT_EQ(AnfAlgo::GetFusionType(add), kernel::kPatternConvolution);
  EXPECT_THROW(AnfAlgo::GetFusionType(nullptr), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, SetSelectKernelBuildInfo) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimAdd));
  auto add = kernel_graph->NewCNode(inputs);
  std::shared_ptr<KernelBuildInfoBuilder> builder = std::make_shared<KernelBuildInfoBuilder>();
  builder->SetFusionType(kernel::kPatternConvolution);
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), add.get());
  EXPECT_THROW(AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), nullptr), std::runtime_error);
  EXPECT_EQ(AnfAlgo::GetFusionType(add), kernel::kPatternConvolution);
}

TEST_F(AnfRuntimeAlgorithmTest, GetKernelMod) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimAdd));
  auto add = kernel_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(add);
  add->set_kernel_info(std::make_shared<KernelInfo>());
  auto d_kernel_info = dynamic_cast<KernelInfo *>(add->kernel_info());
  MS_EXCEPTION_IF_NULL(d_kernel_info);
  d_kernel_info->set_kernel_mod(nullptr);
  EXPECT_EQ(AnfAlgo::GetKernelMod(add), nullptr);
  EXPECT_THROW(AnfAlgo::GetKernelMod(nullptr), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, SetKernelMod) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimAdd));
  auto add = kernel_graph->NewCNode(inputs);
  AnfAlgo::SetKernelMod(nullptr, add.get());
  EXPECT_THROW(AnfAlgo::SetKernelMod(nullptr, nullptr), std::runtime_error);
  EXPECT_EQ(AnfAlgo::GetKernelMod(add), nullptr);
}

TEST_F(AnfRuntimeAlgorithmTest, IsRealKernel) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  // test value node as input
  auto value_node = NewValueNode(prim::kPrimAdd);
  EXPECT_TRUE(AnfUtils::IsRealKernel(value_node));
  EXPECT_THROW(AnfUtils::IsRealKernel(nullptr), std::runtime_error);
  // test parameter as input
  auto parameter_node = kernel_graph->add_parameter();
  EXPECT_TRUE(AnfUtils::IsRealKernel(parameter_node));
  // test add as input
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimAdd));
  auto add = kernel_graph->NewCNode(inputs);
  EXPECT_TRUE(AnfUtils::IsRealKernel(add));
  // test Depend as input
  inputs.clear();
  inputs.push_back(NewValueNode(prim::kPrimDepend));
  auto depend_node = kernel_graph->NewCNode(inputs);
  EXPECT_FALSE(AnfUtils::IsRealKernel(depend_node));
}

TEST_F(AnfRuntimeAlgorithmTest, IsRealCNodeKernel) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  // test value node as input
  auto value_node = NewValueNode(prim::kPrimAdd);
  EXPECT_FALSE(AnfUtils::IsRealCNodeKernel(value_node));
  EXPECT_THROW(AnfUtils::IsRealCNodeKernel(nullptr), std::runtime_error);
  // test parameter as input
  auto parameter_node = kernel_graph->add_parameter();
  EXPECT_FALSE(AnfUtils::IsRealCNodeKernel(parameter_node));
  // test add as input
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimAdd));
  auto add = kernel_graph->NewCNode(inputs);
  EXPECT_TRUE(AnfUtils::IsRealCNodeKernel(add));
  // test ImageSummary as input
  inputs.clear();
  inputs.push_back(NewValueNode(prim::kPrimDepend));
  auto depend = kernel_graph->NewCNode(inputs);
  EXPECT_FALSE(AnfUtils::IsRealCNodeKernel(depend));
}

TEST_F(AnfRuntimeAlgorithmTest, IsParameterWeight) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  auto parameter_node = kernel_graph->add_parameter();
  MS_EXCEPTION_IF_NULL(parameter_node);
  auto param_value_new = std::make_shared<tensor::Tensor>(int64_t(0), kInt32);
  parameter_node->set_default_param(param_value_new);
  EXPECT_TRUE(common::AnfAlgo::IsParameterWeight(parameter_node));
  EXPECT_THROW(common::AnfAlgo::IsParameterWeight(nullptr), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, GetStreamId) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimAdd));
  auto add = kernel_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(add);
  add->set_kernel_info(std::make_shared<KernelInfo>());
  auto d_kernel_info = dynamic_cast<KernelInfo *>(add->kernel_info());
  MS_EXCEPTION_IF_NULL(d_kernel_info);
  d_kernel_info->set_stream_id(0);
  EXPECT_EQ(AnfAlgo::GetStreamId(add), 0);
  EXPECT_THROW(AnfAlgo::GetStreamId(nullptr), std::runtime_error);
}

TEST_F(AnfRuntimeAlgorithmTest, SetStreamId) {
  auto kernel_graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimAdd));
  auto add = kernel_graph->NewCNode(inputs);
  AnfAlgo::SetStreamId(0, add.get());
  EXPECT_THROW(AnfAlgo::SetStreamId(0, nullptr), std::runtime_error);
  EXPECT_EQ(AnfAlgo::GetStreamId(add), 0);
}

/// Feature: fix get item to tuple in tuple.
/// Description: check the function between tuple and getitem.
/// Expectation: As expected.
TEST_F(AnfRuntimeAlgorithmTest, GetAllOutputWithIndex) {
  std::vector<int64_t> shp{1};
  auto graph = std::make_shared<FuncGraph>();
  std::vector<KernelWithIndex> results;

  auto value_node_0 = NewValueNode(static_cast<int64_t>(0));
  auto value_node_1 = NewValueNode(static_cast<int64_t>(1));
  auto value_node_2 = NewValueNode(static_cast<int64_t>(2));
  auto value_node_3 = NewValueNode(static_cast<int64_t>(3));
  auto abstract_single = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList abstruct_list_1{abstract_single, abstract_single};
  auto abstruct_tuple = std::make_shared<abstract::AbstractTuple>(abstruct_list_1);
  AbstractBasePtrList abstruct_list_2{abstract_single, abstruct_tuple, abstract_single};
  auto abstract_tuple_in_tuple = std::make_shared<abstract::AbstractTuple>(abstruct_list_2);
  value_node_1->set_abstract(abstract_single);

  // Make tuple node: (1, 2)
  std::vector<AnfNodePtr> tuple_input1{NewValueNode(prim::kPrimMakeTuple), value_node_1, value_node_2};
  auto tuple_1 = graph->NewCNode(tuple_input1);
  tuple_1->set_abstract(abstruct_tuple);
  results = common::AnfAlgo::GetAllOutputWithIndex(tuple_1);
  EXPECT_EQ(results.size(), 2);

  // Make tuple node: (0, (1, 2), 3)
  std::vector<AnfNodePtr> tuple_input2{NewValueNode(prim::kPrimMakeTuple), value_node_0, tuple_1, value_node_3};
  auto tuple_2 = graph->NewCNode(tuple_input2);
  tuple_2->set_abstract(abstract_tuple_in_tuple);
  results = common::AnfAlgo::GetAllOutputWithIndex(tuple_2);
  EXPECT_EQ(results.size(), 4);

  // Get item node: (0, (1, 2), 3)[1] = (1, 2)
  std::vector<AnfNodePtr> getitem_input1{NewValueNode(prim::kPrimTupleGetItem), tuple_2, value_node_1};
  auto getitem_1 = graph->NewCNode(getitem_input1);
  getitem_1->set_abstract(abstruct_tuple);
  results = common::AnfAlgo::GetAllOutputWithIndex(getitem_1);
  EXPECT_EQ(results.size(), 2);
  EXPECT_EQ(GetValue<int64_t>(results[0].first->cast<ValueNodePtr>()->value()), 1);
  EXPECT_EQ(GetValue<int64_t>(results[1].first->cast<ValueNodePtr>()->value()), 2);

  // Get item node: (0, (1, 2), 3)[1][1] = (1, 2)[1] = (1)
  std::vector<AnfNodePtr> getitem_input2{NewValueNode(prim::kPrimTupleGetItem), getitem_1, value_node_1};
  auto getitem_2 = graph->NewCNode(getitem_input2);
  getitem_2->set_abstract(abstract_single);
  results = common::AnfAlgo::GetAllOutputWithIndex(getitem_2);
  EXPECT_EQ(results.size(), 1);

  // Call node with output construct: (x, (x, x), x)
  std::vector<AnfNodePtr> call_input{NewValueNode(graph)};
  auto call = graph->NewCNode(call_input);
  call->set_abstract(abstract_tuple_in_tuple);
  results = common::AnfAlgo::GetAllOutputWithIndex(call);
  EXPECT_EQ(results.size(), 4);

  // Get item node: (x, (x, x), x)[1] = (x, x)
  std::vector<AnfNodePtr> getitem_input3{NewValueNode(prim::kPrimTupleGetItem), call, value_node_1};
  auto getitem_3 = graph->NewCNode(getitem_input3);
  getitem_3->set_abstract(abstruct_tuple);
  results = common::AnfAlgo::GetAllOutputWithIndex(getitem_3);
  EXPECT_EQ(results.size(), 2);
  EXPECT_EQ(results[0].second, 1);
  EXPECT_EQ(results[1].second, 2);

  // Get item node: (0, (1, 2), 3)[2] = (3)
  std::vector<AnfNodePtr> getitem_input4{NewValueNode(prim::kPrimTupleGetItem), tuple_2, value_node_2};
  auto getitem_4 = graph->NewCNode(getitem_input4);
  getitem_4->set_abstract(abstract_single);
  results = common::AnfAlgo::GetAllOutputWithIndex(getitem_4);
  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(GetValue<int64_t>(results[0].first->cast<ValueNodePtr>()->value()), 3);

  // Get item node: (x, (x, x), x)[2] = (x)
  std::vector<AnfNodePtr> getitem_input5{NewValueNode(prim::kPrimTupleGetItem), call, value_node_2};
  auto getitem_5 = graph->NewCNode(getitem_input5);
  getitem_5->set_abstract(abstract_single);
  results = common::AnfAlgo::GetAllOutputWithIndex(getitem_5);
  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].second, 3);

  // Make tuple node: (3, (x, (x, x), x)[2], 1) = (3, x, 1)
  std::vector<AnfNodePtr> tuple_input3{NewValueNode(prim::kPrimMakeTuple), value_node_3, getitem_5, value_node_1};
  auto tuple_3 = graph->NewCNode(tuple_input3);
  tuple_3->set_abstract(abstract_tuple_in_tuple);
  results = common::AnfAlgo::GetAllOutputWithIndex(tuple_3);
  EXPECT_EQ(results.size(), 3);

  // Get item node: (3, (x, (x, x), x)[2], 1)[1] = (3, x, 1)[1] = (x)
  std::vector<AnfNodePtr> getitem_input6{NewValueNode(prim::kPrimTupleGetItem), tuple_3, value_node_1};
  auto getitem_6 = graph->NewCNode(getitem_input5);
  getitem_6->set_abstract(abstract_single);
  results = common::AnfAlgo::GetAllOutputWithIndex(getitem_6);
  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].second, 3);
}
}  // namespace session
}  // namespace mindspore
