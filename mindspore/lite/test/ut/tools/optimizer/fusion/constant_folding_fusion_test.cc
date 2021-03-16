/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <memory>
#include "schema/inner/model_generated.h"
#include "include/model.h"
#include "common/common_test.h"
#include "include/lite_session.h"
#include "include/context.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "tools/converter/model_parser.h"
#include "tools/converter/anf_transform.h"
#include "tools/optimizer/fusion/constant_folding_fusion.h"
#include "tools/anf_exporter/anf_exporter.h"
#include "test/common/import_from_meta_graphT.h"

namespace mindspore {
class ConstantFoldingFusionTest : public mindspore::CommonTest {
 public:
  ConstantFoldingFusionTest() = default;
};
using MetaGraphTptr = std::shared_ptr<schema::MetaGraphT>;
using CNodeTptr = std::unique_ptr<schema::CNodeT>;

namespace {

MetaGraphTptr BuildGraph(schema::PrimitiveType op_type, void *op_node) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  // biasadd node
  auto example_node = std::make_unique<schema::CNodeT>();
  example_node->inputIndex = {0, 1};
  example_node->outputIndex = {2};
  example_node->primitive = std::make_unique<schema::PrimitiveT>();
  example_node->primitive->value.type = op_type;
  example_node->primitive->value.value = op_node;
  example_node->name = "example";
  meta_graph->nodes.emplace_back(std::move(example_node));

  meta_graph->inputIndex = {0, 1};
  meta_graph->outputIndex = {2};

  // input 0: data1
  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_ValueNode;
  input0->format = schema::Format_NHWC;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {1, 2, 2, 3};
  input0->offset = -1;
  auto input0_data = new (std::nothrow) float[2 * 2 * 3];
  for (auto i = 0; i < 2 * 2 * 3; i++) {
    input0_data[i] = i;
  }
  input0->data.resize(sizeof(float) * 2 * 2 * 3);
  memcpy(input0->data.data(), input0_data, 2 * 2 * 3 * sizeof(float));
  delete[] input0_data;
  meta_graph->allTensors.emplace_back(std::move(input0));

  // input 1: data2
  auto input1 = std::make_unique<schema::TensorT>();
  input1->nodeType = lite::NodeType_ValueNode;
  input1->format = schema::Format_NHWC;
  input1->dataType = TypeId::kNumberTypeFloat32;
  input1->dims = {1, 2, 2, 3};
  input1->offset = -1;
  input1->data.resize(sizeof(float) * 2 * 2 * 3);
  auto input1_data = new (std::nothrow) float[2 * 2 * 3];
  for (auto i = 0; i < 2 * 2 * 3; i++) {
    input1_data[i] = i;
  }
  memcpy(input1->data.data(), input1_data, 2 * 2 * 3 * sizeof(float));
  delete[] input1_data;
  meta_graph->allTensors.emplace_back(std::move(input1));

  // final add output
  auto add_output = std::make_unique<schema::TensorT>();
  add_output->nodeType = lite::NodeType_Parameter;
  add_output->format = schema::Format_NHWC;
  add_output->dataType = TypeId::kNumberTypeFloat32;
  add_output->dims = {1, 2, 2, 3};
  meta_graph->allTensors.emplace_back(std::move(add_output));
  // final output
  return meta_graph;
}

MetaGraphTptr BuildGraphForOneInput(schema::PrimitiveType op_type, void *op_node) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  // biasadd node
  auto example_node = std::make_unique<schema::CNodeT>();
  example_node->inputIndex = {0};
  example_node->outputIndex = {1};
  example_node->primitive = std::make_unique<schema::PrimitiveT>();
  example_node->primitive->value.type = op_type;
  example_node->primitive->value.value = op_node;
  example_node->name = "example";
  meta_graph->nodes.emplace_back(std::move(example_node));

  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {1};

  // input 0: data1
  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_ValueNode;
  input0->format = schema::Format_NHWC;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {1, 2, 2, 3};
  input0->offset = -1;
  auto input0_data = new (std::nothrow) float[2 * 2 * 3];
  for (auto i = 0; i < 2 * 2 * 3; i++) {
    input0_data[i] = i + 1;
  }
  input0->data.resize(sizeof(float) * 2 * 2 * 3);
  memcpy(input0->data.data(), input0_data, 2 * 2 * 3 * sizeof(float));
  delete[] input0_data;
  meta_graph->allTensors.emplace_back(std::move(input0));

  // final add output
  auto add_output = std::make_unique<schema::TensorT>();
  add_output->nodeType = lite::NodeType_Parameter;
  add_output->format = schema::Format_NHWC;
  add_output->dataType = TypeId::kNumberTypeFloat32;
  add_output->dims = {1, 2, 2, 3};
  meta_graph->allTensors.emplace_back(std::move(add_output));

  // final output
  return meta_graph;
}

MetaGraphTptr BuildMixGraph() {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  // add node
  auto add_node = std::make_unique<schema::CNodeT>();
  add_node->inputIndex = {0, 1};
  add_node->outputIndex = {2};
  add_node->primitive = std::make_unique<schema::PrimitiveT>();
  add_node->primitive->value.type = schema::PrimitiveType_AddFusion;
  add_node->primitive->value.value = new schema::AddFusionT;
  add_node->name = "add";
  meta_graph->nodes.emplace_back(std::move(add_node));

  meta_graph->inputIndex = {0, 1, 2};
  meta_graph->outputIndex = {4};

  auto mul_node = std::make_unique<schema::CNodeT>();
  mul_node->inputIndex = {2, 3};
  mul_node->outputIndex = {4};
  mul_node->primitive = std::make_unique<schema::PrimitiveT>();
  mul_node->primitive->value.type = schema::PrimitiveType_MulFusion;
  mul_node->primitive->value.value = new schema::MulFusionT;
  mul_node->name = "mul";
  meta_graph->nodes.emplace_back(std::move(mul_node));

  // input 0: data1
  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_ValueNode;
  input0->format = schema::Format_NHWC;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {1, 2, 2, 3};
  input0->offset = -1;
  auto input0_data = new (std::nothrow) float[2 * 2 * 3];
  for (auto i = 0; i < 2 * 2 * 3; i++) {
    input0_data[i] = i;
  }
  input0->data.resize(sizeof(float) * 2 * 2 * 3);
  memcpy(input0->data.data(), input0_data, 2 * 2 * 3 * sizeof(float));
  delete[] input0_data;
  meta_graph->allTensors.emplace_back(std::move(input0));

  // input 1: data2
  auto input1 = std::make_unique<schema::TensorT>();
  input1->nodeType = lite::NodeType_ValueNode;
  input1->format = schema::Format_NHWC;
  input1->dataType = TypeId::kNumberTypeFloat32;
  input1->dims = {1, 2, 2, 3};
  input1->offset = -1;
  input1->data.resize(sizeof(float) * 2 * 2 * 3);
  auto input1_data = new (std::nothrow) float[2 * 2 * 3];
  for (auto i = 0; i < 2 * 2 * 3; i++) {
    input1_data[i] = i;
  }
  memcpy(input1->data.data(), input1_data, 2 * 2 * 3 * sizeof(float));
  delete[] input1_data;
  meta_graph->allTensors.emplace_back(std::move(input1));

  // addoutput
  auto add_output = std::make_unique<schema::TensorT>();
  add_output->nodeType = lite::NodeType_Parameter;
  add_output->format = schema::Format_NHWC;
  add_output->dataType = TypeId::kNumberTypeFloat32;
  add_output->dims = {1, 2, 2, 3};
  add_output->offset = -1;
  add_output->data.resize(sizeof(float) * 2 * 2 * 3);
  auto add_output_data = new (std::nothrow) float[2 * 2 * 3];
  memcpy(add_output->data.data(), add_output_data, 2 * 2 * 3 * sizeof(float));
  delete[] add_output_data;
  meta_graph->allTensors.emplace_back(std::move(add_output));

  // input 2: data3
  auto input2 = std::make_unique<schema::TensorT>();
  input2->nodeType = lite::NodeType_ValueNode;
  input2->format = schema::Format_NHWC;
  input2->dataType = TypeId::kNumberTypeFloat32;
  input2->dims = {1, 2, 2, 3};
  input2->offset = -1;
  input2->data.resize(sizeof(float) * 2 * 2 * 3);
  auto input2_data = new (std::nothrow) float[2 * 2 * 3];
  for (auto i = 0; i < 2 * 2 * 3; i++) {
    input2_data[i] = 10;
  }
  memcpy(input2->data.data(), input2_data, 2 * 2 * 3 * sizeof(float));
  delete[] input2_data;
  meta_graph->allTensors.emplace_back(std::move(input2));

  // final mul output
  auto mul_output = std::make_unique<schema::TensorT>();
  mul_output->nodeType = lite::NodeType_Parameter;
  mul_output->format = schema::Format_NHWC;
  mul_output->dataType = TypeId::kNumberTypeFloat32;
  mul_output->dims = {1, 2, 2, 3};
  meta_graph->allTensors.emplace_back(std::move(mul_output));
  // final output
  return meta_graph;
}
MetaGraphTptr BuildSplitGraph() {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  // slice node
  auto split_node = std::make_unique<schema::CNodeT>();
  split_node->inputIndex = {0};
  split_node->outputIndex = {1, 2};
  split_node->primitive = std::make_unique<schema::PrimitiveT>();
  split_node->primitive->value.type = schema::PrimitiveType_Split;
  std::unique_ptr<schema::SplitT> attr = std::make_unique<schema::SplitT>();
  attr->output_num = 2;
  attr->axis = 1;
  split_node->primitive->value.value = attr.release();
  split_node->name = "split";
  meta_graph->nodes.emplace_back(std::move(split_node));

  meta_graph->inputIndex = {0, 3, 4};
  meta_graph->outputIndex = {5, 6};

  auto mul_node1 = std::make_unique<schema::CNodeT>();
  mul_node1->inputIndex = {1, 3};
  mul_node1->outputIndex = {5};
  mul_node1->primitive = std::make_unique<schema::PrimitiveT>();
  mul_node1->primitive->value.type = schema::PrimitiveType_MulFusion;
  std::unique_ptr<schema::MulFusionT> mul_attr = std::make_unique<schema::MulFusionT>();
  mul_node1->primitive->value.value = mul_attr.release();
  mul_node1->name = "mul1";
  meta_graph->nodes.emplace_back(std::move(mul_node1));

  auto mul_node2 = std::make_unique<schema::CNodeT>();
  mul_node2->inputIndex = {2, 4};
  mul_node2->outputIndex = {6};
  mul_node2->primitive = std::make_unique<schema::PrimitiveT>();
  mul_node2->primitive->value.type = schema::PrimitiveType_MulFusion;
  std::unique_ptr<schema::MulFusionT> mul2_attr = std::make_unique<schema::MulFusionT>();
  mul_node2->primitive->value.value = mul2_attr.release();
  mul_node2->name = "mul2";
  meta_graph->nodes.emplace_back(std::move(mul_node2));

  // input 0: data1
  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_ValueNode;
  input0->format = schema::Format_NHWC;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {1, 2, 2, 3};
  input0->offset = -1;
  auto input0_data = new (std::nothrow) float[2 * 2 * 3];
  for (auto i = 0; i < 2 * 2 * 3; i++) {
    input0_data[i] = i;
  }
  input0->data.resize(sizeof(float) * 2 * 2 * 3);
  memcpy(input0->data.data(), input0_data, 2 * 2 * 3 * sizeof(float));
  delete[] input0_data;
  meta_graph->allTensors.emplace_back(std::move(input0));

  // split output1
  auto split_output1 = std::make_unique<schema::TensorT>();
  split_output1->nodeType = lite::NodeType_Parameter;
  split_output1->format = schema::Format_NHWC;
  split_output1->dataType = TypeId::kNumberTypeFloat32;
  split_output1->dims = {1, 1, 2, 3};
  split_output1->offset = -1;
  split_output1->data.resize(sizeof(float) * 1 * 2 * 3);
  auto split_output_data1 = new (std::nothrow) float[1 * 2 * 3];
  memcpy(split_output1->data.data(), split_output_data1, 1 * 2 * 3 * sizeof(float));
  delete[] split_output_data1;
  meta_graph->allTensors.emplace_back(std::move(split_output1));

  // split output2
  auto split_output2 = std::make_unique<schema::TensorT>();
  split_output2->nodeType = lite::NodeType_Parameter;
  split_output2->format = schema::Format_NHWC;
  split_output2->dataType = TypeId::kNumberTypeFloat32;
  split_output2->dims = {1, 1, 2, 3};
  split_output2->offset = -1;
  split_output2->data.resize(sizeof(float) * 1 * 2 * 3);
  auto split_output_data2 = new (std::nothrow) float[1 * 2 * 3];
  memcpy(split_output2->data.data(), split_output_data2, 1 * 2 * 3 * sizeof(float));
  delete[] split_output_data2;
  meta_graph->allTensors.emplace_back(std::move(split_output2));

  // input 1: data2
  auto input1 = std::make_unique<schema::TensorT>();
  input1->nodeType = lite::NodeType_ValueNode;
  input1->format = schema::Format_NHWC;
  input1->dataType = TypeId::kNumberTypeFloat32;
  input1->dims = {1, 1, 2, 3};
  input1->offset = -1;
  input1->data.resize(sizeof(float) * 2 * 3);
  auto input1_data = new (std::nothrow) float[2 * 3];
  for (auto i = 0; i < 2 * 3; i++) {
    input1_data[i] = i;
  }
  memcpy(input1->data.data(), input1_data, 2 * 3 * sizeof(float));
  delete[] input1_data;
  meta_graph->allTensors.emplace_back(std::move(input1));

  // input 2: data3
  auto input2 = std::make_unique<schema::TensorT>();
  input2->nodeType = lite::NodeType_ValueNode;
  input2->format = schema::Format_NHWC;
  input2->dataType = TypeId::kNumberTypeFloat32;
  input2->dims = {1, 1, 2, 3};
  input2->offset = -1;
  input2->data.resize(sizeof(float) * 2 * 3);
  auto input2_data = new (std::nothrow) float[2 * 3];
  for (auto i = 0; i < 2 * 3; i++) {
    input2_data[i] = 10;
  }
  memcpy(input2->data.data(), input2_data, 2 * 3 * sizeof(float));
  delete[] input2_data;
  meta_graph->allTensors.emplace_back(std::move(input2));

  // final mul output1
  auto mul_output = std::make_unique<schema::TensorT>();
  mul_output->nodeType = lite::NodeType_Parameter;
  mul_output->format = schema::Format_NHWC;
  mul_output->dataType = TypeId::kNumberTypeFloat32;
  mul_output->dims = {1, 1, 2, 3};
  meta_graph->allTensors.emplace_back(std::move(mul_output));

  // final mul output2
  auto mul_output2 = std::make_unique<schema::TensorT>();
  mul_output2->nodeType = lite::NodeType_Parameter;
  mul_output2->format = schema::Format_NHWC;
  mul_output2->dataType = TypeId::kNumberTypeFloat32;
  mul_output2->dims = {1, 1, 2, 3};
  meta_graph->allTensors.emplace_back(std::move(mul_output2));
  return meta_graph;
}
}  //  namespace
TEST_F(ConstantFoldingFusionTest, TestADDConstantFold) {
  auto meta_graph = BuildGraph(schema::PrimitiveType_AddFusion, new schema::AddFusionT);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ConstFoldPass>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(func_graph);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 0);
}

TEST_F(ConstantFoldingFusionTest, TestMixedConstantFold) {
  auto meta_graph = BuildMixGraph();
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ConstFoldPass>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(func_graph);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 0);
}

TEST_F(ConstantFoldingFusionTest, TestSubConstantFold) {
  auto meta_graph = BuildGraph(schema::PrimitiveType_SubFusion, new schema::SubFusionT);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ConstFoldPass>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(func_graph);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 0);
}

TEST_F(ConstantFoldingFusionTest, TestMulConstantFold) {
  auto meta_graph = BuildGraph(schema::PrimitiveType_MulFusion, new schema::MulFusionT);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ConstFoldPass>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(func_graph);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 0);
}

TEST_F(ConstantFoldingFusionTest, TestTransposeConstantFold) {
  auto transposeT = new schema::TransposeT;
  auto meta_graph = BuildGraph(schema::PrimitiveType_Transpose, transposeT);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ConstFoldPass>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(func_graph);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 0);
}

TEST_F(ConstantFoldingFusionTest, TestStackConstantFold) {
  auto stackT = new schema::StackT;
  stackT->axis = 1;
  auto meta_graph = BuildGraph(schema::PrimitiveType_Stack, stackT);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ConstFoldPass>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(func_graph);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 0);
}

TEST_F(ConstantFoldingFusionTest, TestSliceConstantFold) {
  auto sliceT = new schema::SliceFusionT;
  auto meta_graph = BuildGraph(schema::PrimitiveType_SliceFusion, sliceT);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ConstFoldPass>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(func_graph);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 0);
}

TEST_F(ConstantFoldingFusionTest, TestShapeConstantFold) {
  auto shapeT = new schema::ShapeT;
  auto meta_graph = BuildGraphForOneInput(schema::PrimitiveType_Shape, shapeT);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ConstFoldPass>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(func_graph);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 0);
}

TEST_F(ConstantFoldingFusionTest, TestRsqrtConstantFold) {
  auto rsqrtT = new schema::RsqrtT;
  auto meta_graph = BuildGraphForOneInput(schema::PrimitiveType_Rsqrt, rsqrtT);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ConstFoldPass>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(func_graph);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 0);
}

TEST_F(ConstantFoldingFusionTest, TestReshapeConstantFold) {
  auto reshapeT = new schema::ReshapeT;
  auto meta_graph = BuildGraphForOneInput(schema::PrimitiveType_Reshape, reshapeT);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ConstFoldPass>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(func_graph);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 0);
}

TEST_F(ConstantFoldingFusionTest, TestRangeConstantFold) {
  auto rangeT = new schema::RangeT;
  rangeT->limit = 10;
  rangeT->start = 1;
  rangeT->delta = 1;
  auto meta_graph = BuildGraphForOneInput(schema::PrimitiveType_Range, rangeT);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ConstFoldPass>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(func_graph);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 0);
}
TEST_F(ConstantFoldingFusionTest, TestMatmulConstantFold) {
  auto matmulT = new schema::MatMulT;
  auto meta_graph = BuildGraph(schema::PrimitiveType_MatMul, matmulT);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ConstFoldPass>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(func_graph);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 0);
}

TEST_F(ConstantFoldingFusionTest, TestExpandDimsConstantFold) {
  auto expandDimsT = new schema::ExpandDimsT;
  auto meta_graph = BuildGraphForOneInput(schema::PrimitiveType_ExpandDims, expandDimsT);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ConstFoldPass>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(func_graph);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 0);
}

TEST_F(ConstantFoldingFusionTest, TestConcatDimsConstantFold) {
  auto concatT = new schema::ConcatT;
  auto meta_graph = BuildGraph(schema::PrimitiveType_Concat, concatT);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ConstFoldPass>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(func_graph);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 0);
}

TEST_F(ConstantFoldingFusionTest, TestCastDimsConstantFold) {
  auto castT = new schema::CastT;
  auto meta_graph = BuildGraphForOneInput(schema::PrimitiveType_Cast, castT);
  auto input_tensor = meta_graph->allTensors.at(0).get();
  input_tensor->dataType = kNumberTypeUInt8;
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ConstFoldPass>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(func_graph);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 0);
}

TEST_F(ConstantFoldingFusionTest, TestSplitConstantFold) {
  auto meta_graph = BuildSplitGraph();
  auto input_tensor = meta_graph->allTensors.at(0).get();
  input_tensor->dataType = kNumberTypeFloat32;
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>("test", false);
  pm->AddPass(std::make_shared<opt::ConstFoldPass>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(func_graph);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 0);
}
}  // namespace mindspore
