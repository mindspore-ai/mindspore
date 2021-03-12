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
#include "tools/anf_exporter/anf_exporter.h"
#include "test/common/import_from_meta_graphT.h"

namespace mindspore {
class ConvBiasAddFusionTest : public mindspore::CommonTest {
 public:
  ConvBiasAddFusionTest() = default;
};
using MetaGraphTptr = std::shared_ptr<schema::MetaGraphT>;
using CNodeTptr = std::unique_ptr<schema::CNodeT>;

namespace {
CNodeTptr BuildConv2D() {
  auto convNode = std::make_unique<schema::CNodeT>();
  convNode->inputIndex = {0, 1};
  convNode->outputIndex = {2};
  convNode->primitive = std::make_unique<schema::PrimitiveT>();
  convNode->primitive->value.type = schema::PrimitiveType_Conv2DFusion;
  auto prim1 = new schema::Conv2DFusionT;
  prim1->pad_mode = schema::PadMode_SAME;
  prim1->format = schema::Format_NHWC;
  prim1->stride = {1, 1};
  prim1->kernel_size = {3, 3};
  prim1->dilation = {1, 1};
  prim1->out_channel = 3;
  convNode->primitive->value.value = prim1;
  convNode->name = "Conv2D";
  return convNode;
}
CNodeTptr BuildDepthwiseConv2D() {
  auto convNode = std::make_unique<schema::CNodeT>();
  convNode->inputIndex = {0, 1};
  convNode->outputIndex = {2};
  convNode->primitive = std::make_unique<schema::PrimitiveT>();
  convNode->primitive->value.type = schema::PrimitiveType_Conv2DFusion;
  auto prim1 = new schema::Conv2DFusionT;
  prim1->pad_mode = schema::PadMode_SAME;
  prim1->format = schema::Format_NHWC;
  prim1->stride = {1, 1};
  prim1->kernel_size = {3, 3};
  prim1->dilation = {1, 1};
  prim1->in_channel = 1;
  convNode->primitive->value.value = prim1;
  convNode->name = "Conv2D";
  return convNode;
}

MetaGraphTptr BuildGraph(schema::PrimitiveType conv_type, schema::PrimitiveType add_type) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  // conv node
  CNodeTptr convNode;
  if (conv_type == schema::PrimitiveType_Conv2DFusion) {
    convNode = BuildConv2D();
  } else {
    convNode = BuildDepthwiseConv2D();
  }

  meta_graph->nodes.emplace_back(std::move(convNode));

  // biasadd node
  auto biasadd_node = std::make_unique<schema::CNodeT>();
  biasadd_node->inputIndex = {2, 3};
  biasadd_node->outputIndex = {4};
  biasadd_node->primitive = std::make_unique<schema::PrimitiveT>();
  biasadd_node->primitive->value.type = add_type;
  auto prim2 = new schema::BiasAddT;
  biasadd_node->primitive->value.value = prim2;
  biasadd_node->name = "BiasAdd";
  meta_graph->nodes.emplace_back(std::move(biasadd_node));

  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {4};

  // input 0: data
  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_ValueNode;
  input0->format = schema::Format_NHWC;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {1, 5, 5, 3};
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));

  // input 1: weight
  auto input1 = std::make_unique<schema::TensorT>();
  input1->nodeType = lite::NodeType_ValueNode;
  input1->format = schema::Format_KHWC;
  input1->dataType = TypeId::kNumberTypeFloat32;
  input1->dims = {8, 3, 3, 3};
  input1->data.resize(sizeof(float) * 8 * 3 * 3 * 3);
  meta_graph->allTensors.emplace_back(std::move(input1));

  // conv output
  auto conv_output = std::make_unique<schema::TensorT>();
  conv_output->nodeType = lite::NodeType_Parameter;
  conv_output->format = schema::Format_NHWC;
  conv_output->dataType = TypeId::kNumberTypeFloat32;
  conv_output->dims = {1, 5, 5, 8};
  meta_graph->allTensors.emplace_back(std::move(conv_output));

  // input2: bias
  auto input2 = std::make_unique<schema::TensorT>();
  input2->nodeType = lite::NodeType_ValueNode;
  input2->format = schema::Format_NHWC;
  input2->dataType = TypeId::kNumberTypeFloat32;
  input2->dims = {1, 5, 5, 8};
  input2->data.resize(sizeof(float) * 8 * 5 * 5);
  meta_graph->allTensors.emplace_back(std::move(input2));

  // final output
  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = lite::NodeType_Parameter;
  output->format = schema::Format_NHWC;
  output->dataType = TypeId::kNumberTypeFloat32;
  output->dims = {1, 5, 5, 8};
  meta_graph->allTensors.emplace_back(std::move(output));
  return meta_graph;
}
}  //  namespace
TEST_F(ConvBiasAddFusionTest, TestConvAddNode) {
  auto meta_graph = BuildGraph(schema::PrimitiveType_Conv2DFusion, schema::PrimitiveType_BiasAdd);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto anf_transform = new lite::AnfTransform();
  auto new_graph = anf_transform->Transform(func_graph);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 1);
  MS_LOG(INFO) << "Passed";
}

TEST_F(ConvBiasAddFusionTest, TestDeptiwiseConvAddNode) {
  auto meta_graph = BuildGraph(schema::PrimitiveType_Conv2DFusion, schema::PrimitiveType_AddFusion);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto anf_transform = new lite::AnfTransform();
  auto new_graph = anf_transform->Transform(func_graph);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 1);
}

TEST_F(ConvBiasAddFusionTest, TestBadCase_ConvAdd) {
  auto meta_graph = BuildGraph(schema::PrimitiveType_Conv2DFusion, schema::PrimitiveType_MatMul);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto anf_transform = new lite::AnfTransform();
  auto new_graph = anf_transform->Transform(func_graph);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 2);
}
}  // namespace mindspore
