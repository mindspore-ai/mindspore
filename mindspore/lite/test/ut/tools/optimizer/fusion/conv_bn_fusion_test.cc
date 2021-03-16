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
class ConvBNFusionTest : public mindspore::CommonTest {
 public:
  ConvBNFusionTest() = default;
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
  convNode->inputIndex = {0, 1, 2};
  convNode->outputIndex = {3};
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
// caffe bn op has 3 inputs
MetaGraphTptr BuildCaffeGraph(schema::PrimitiveType conv_type) {
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

  // bn_node
  auto bn_node = std::make_unique<schema::CNodeT>();
  bn_node->inputIndex = {2, 3, 4};
  bn_node->outputIndex = {5};
  bn_node->primitive = std::make_unique<schema::PrimitiveT>();
  bn_node->primitive->value.type = schema::PrimitiveType_BatchNorm;
  auto prim2 = new schema::BatchNormT;
  bn_node->primitive->value.value = prim2;
  bn_node->name = "bn";
  meta_graph->nodes.emplace_back(std::move(bn_node));

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

  // caffe bn : mean
  auto input2 = std::make_unique<schema::TensorT>();
  input2->nodeType = lite::NodeType_ValueNode;
  input2->format = schema::Format_NHWC;
  input2->dataType = TypeId::kNumberTypeFloat32;
  input2->dims = {1, 5, 5, 8};
  input2->data.resize(sizeof(float) * 8 * 5 * 5);
  meta_graph->allTensors.emplace_back(std::move(input2));

  // caffe bn : var
  auto input3 = std::make_unique<schema::TensorT>();
  input3->nodeType = lite::NodeType_ValueNode;
  input3->format = schema::Format_NHWC;
  input3->dataType = TypeId::kNumberTypeFloat32;
  input3->dims = {1, 5, 5, 8};
  input3->data.resize(sizeof(float) * 8 * 5 * 5);
  meta_graph->allTensors.emplace_back(std::move(input3));

  // final bn output
  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = lite::NodeType_Parameter;
  output->format = schema::Format_NHWC;
  output->dataType = TypeId::kNumberTypeFloat32;
  output->dims = {1, 5, 5, 8};
  meta_graph->allTensors.emplace_back(std::move(output));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {5};
  return meta_graph;
}

// tf bn op has 4 inputs
MetaGraphTptr BuildTFGraph(schema::PrimitiveType conv_type) {
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

  // bn_node
  auto bn_node = std::make_unique<schema::CNodeT>();
  bn_node->inputIndex = {3, 4, 5, 6, 7};
  bn_node->outputIndex = {8};
  bn_node->primitive = std::make_unique<schema::PrimitiveT>();
  bn_node->primitive->value.type = schema::PrimitiveType_FusedBatchNorm;
  auto prim2 = new schema::FusedBatchNormT;
  bn_node->primitive->value.value = prim2;
  bn_node->name = "bn";
  meta_graph->nodes.emplace_back(std::move(bn_node));

  // input 0: data
  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_ValueNode;
  input0->format = schema::Format_NHWC;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {1, 5, 5, 3};
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));

  // input 1: conv_bias
  auto input11 = std::make_unique<schema::TensorT>();
  input11->nodeType = lite::NodeType_ValueNode;
  input11->format = schema::Format_KHWC;
  input11->dataType = TypeId::kNumberTypeFloat32;
  input11->dims = {8, 3, 3, 3};
  input11->data.resize(sizeof(float) * 8 * 3 * 3 * 3);
  meta_graph->allTensors.emplace_back(std::move(input11));

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

  // tflite bn : scale
  auto input2 = std::make_unique<schema::TensorT>();
  input2->nodeType = lite::NodeType_ValueNode;
  input2->format = schema::Format_NHWC;
  input2->dataType = TypeId::kNumberTypeFloat32;
  input2->dims = {1, 5, 5, 8};
  input2->data.resize(sizeof(float) * 8 * 5 * 5);
  meta_graph->allTensors.emplace_back(std::move(input2));

  // tflite bn : bias
  auto input3 = std::make_unique<schema::TensorT>();
  input3->nodeType = lite::NodeType_ValueNode;
  input3->format = schema::Format_NHWC;
  input3->dataType = TypeId::kNumberTypeFloat32;
  input3->dims = {1, 5, 5, 8};
  input3->data.resize(sizeof(float) * 8 * 5 * 5);
  meta_graph->allTensors.emplace_back(std::move(input3));

  // tflite bn : mean
  auto input4 = std::make_unique<schema::TensorT>();
  input4->nodeType = lite::NodeType_ValueNode;
  input4->format = schema::Format_NHWC;
  input4->dataType = TypeId::kNumberTypeFloat32;
  input4->dims = {1, 5, 5, 8};
  input4->data.resize(sizeof(float) * 8 * 5 * 5);
  meta_graph->allTensors.emplace_back(std::move(input4));

  // tflite bn : var
  auto input5 = std::make_unique<schema::TensorT>();
  input5->nodeType = lite::NodeType_ValueNode;
  input5->format = schema::Format_NHWC;
  input5->dataType = TypeId::kNumberTypeFloat32;
  input5->dims = {1, 5, 5, 8};
  input5->data.resize(sizeof(float) * 8 * 5 * 5);
  meta_graph->allTensors.emplace_back(std::move(input5));

  // final output
  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = lite::NodeType_Parameter;
  output->format = schema::Format_NHWC;
  output->dataType = TypeId::kNumberTypeFloat32;
  output->dims = {1, 5, 5, 8};
  meta_graph->allTensors.emplace_back(std::move(output));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {8};
  return meta_graph;
}
}  //  namespace
TEST_F(ConvBNFusionTest, TestConvAddNode) {
  auto meta_graph = BuildCaffeGraph(schema::PrimitiveType_Conv2DFusion);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto anf_transform = new lite::AnfTransform();
  auto new_graph = anf_transform->Transform(func_graph);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 1);
}

TEST_F(ConvBNFusionTest, TestDeptiwiseConvAddNode) {
  auto meta_graph = BuildTFGraph(schema::PrimitiveType_Conv2DFusion);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto anf_transform = new lite::AnfTransform();
  auto new_graph = anf_transform->Transform(func_graph);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 1);
}
}  // namespace mindspore
