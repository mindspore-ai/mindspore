/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <cmath>
#include <memory>
#include "schema/inner/model_generated.h"
#include "mindspore/lite/include/model.h"
#include "common/common_test.h"
#include "include/lite_session.h"
#include "include/context.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/lite_session.h"
#include "include/version.h"

namespace mindspore {
class ControlFlowTest : public mindspore::CommonTest {
 public:
  ControlFlowTest() {}
};

TEST_F(ControlFlowTest, TestMergeWhileModel) {
  // make graph
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  MS_LOG(DEBUG) << "make subgraph";
  meta_graph->name = "graph";
  meta_graph->version = lite::Version();
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {9};
  //  subgraph 0 : main graph
  auto sub_graph_0 = std::make_unique<schema::SubGraphT>();
  sub_graph_0->name = "main_graph";

  //  subgraph 1 : cond graph
  auto sub_graph_1 = std::make_unique<schema::SubGraphT>();
  sub_graph_1->name = "cond_graph";

  //  subgraph 2: body graph
  auto sub_graph_2 = std::make_unique<schema::SubGraphT>();
  sub_graph_2->name = "body_graph";

  MS_LOG(DEBUG) << "make subgraph";

  // subgraph 0:  node 0    before-add-1
  auto sub_graph_0_node_0 = std::make_unique<schema::CNodeT>();
  sub_graph_0_node_0->inputIndex = {0, 1};
  sub_graph_0_node_0->outputIndex = {2};
  sub_graph_0_node_0->primitive = std::make_unique<schema::PrimitiveT>();
  sub_graph_0_node_0->primitive->value.type = schema::PrimitiveType_AddFusion;
  auto primitive_sub_graph_0_node_0 = new schema::AddFusionT;
  primitive_sub_graph_0_node_0->activation_type = schema::ActivationType_NO_ACTIVATION;
  sub_graph_0_node_0->primitive->value.value = primitive_sub_graph_0_node_0;
  sub_graph_0_node_0->name = "before_Add_1";
  meta_graph->nodes.emplace_back(std::move(sub_graph_0_node_0));
  sub_graph_0->nodeIndices.push_back(0);
  MS_LOG(DEBUG) << "node 0";

  // subgraph 0:  node 1    before-add-1
  auto sub_graph_0_node_1 = std::make_unique<schema::CNodeT>();
  sub_graph_0_node_1->inputIndex = {2, 3};
  sub_graph_0_node_1->outputIndex = {4};
  sub_graph_0_node_1->primitive = std::make_unique<schema::PrimitiveT>();
  sub_graph_0_node_1->primitive->value.type = schema::PrimitiveType_AddFusion;
  auto primitive_sub_graph_0_node_1 = new schema::AddFusionT;
  primitive_sub_graph_0_node_1->activation_type = schema::ActivationType_NO_ACTIVATION;
  sub_graph_0_node_1->primitive->value.value = primitive_sub_graph_0_node_1;
  sub_graph_0_node_1->name = "before_Add_2";
  meta_graph->nodes.emplace_back(std::move(sub_graph_0_node_1));
  sub_graph_0->nodeIndices.push_back(1);
  MS_LOG(DEBUG) << "node 1";

  // subgraph 0:  node 2    merge
  auto sub_graph_0_node_2 = std::make_unique<schema::CNodeT>();
  sub_graph_0_node_2->inputIndex = {4, 17};
  sub_graph_0_node_2->outputIndex = {16};
  sub_graph_0_node_2->primitive = std::make_unique<schema::PrimitiveT>();
  sub_graph_0_node_2->primitive->value.type = schema::PrimitiveType_Merge;
  auto primitive_sub_graph_0_node_2 = new schema::MergeT;
  sub_graph_0_node_2->primitive->value.value = primitive_sub_graph_0_node_2;
  sub_graph_0_node_2->name = "merge";
  meta_graph->nodes.emplace_back(std::move(sub_graph_0_node_2));
  sub_graph_0->nodeIndices.push_back(2);
  MS_LOG(DEBUG) << "node 2";

  // subgraph 0:  node 3   partial   cond subGraph
  auto sub_graph_0_node_3 = std::make_unique<schema::CNodeT>();
  sub_graph_0_node_3->inputIndex = {16};
  sub_graph_0_node_3->outputIndex = {5};  // 5 : bool
  sub_graph_0_node_3->primitive = std::make_unique<schema::PrimitiveT>();
  sub_graph_0_node_3->primitive->value.type = schema::PrimitiveType_PartialFusion;
  auto primitive_sub_graph_0_node_3 = new schema::PartialFusionT;
  primitive_sub_graph_0_node_3->sub_graph_index = 1;
  sub_graph_0_node_3->primitive->value.value = primitive_sub_graph_0_node_3;
  sub_graph_0_node_3->name = "Partial_cond";
  meta_graph->nodes.emplace_back(std::move(sub_graph_0_node_3));
  sub_graph_0->nodeIndices.push_back(3);
  MS_LOG(DEBUG) << "node 2";

  // subgraph 0:  node 4   switch
  auto sub_graph_0_node_4 = std::make_unique<schema::CNodeT>();
  sub_graph_0_node_4->inputIndex = {5, 16};  // 5 : bool; 16 data
  sub_graph_0_node_4->outputIndex = {6, 7};
  sub_graph_0_node_4->primitive = std::make_unique<schema::PrimitiveT>();
  sub_graph_0_node_4->primitive->value.type = schema::PrimitiveType_Switch;
  auto primitive_sub_graph_0_node_4 = new schema::SwitchT;
  sub_graph_0_node_4->primitive->value.value = primitive_sub_graph_0_node_4;
  sub_graph_0_node_4->name = "Switch";
  meta_graph->nodes.emplace_back(std::move(sub_graph_0_node_4));
  sub_graph_0->nodeIndices.push_back(4);
  MS_LOG(DEBUG) << "node 4";

  // subgraph 0:  node 5    partial  body subgraph
  auto sub_graph_0_node_5 = std::make_unique<schema::CNodeT>();
  sub_graph_0_node_5->inputIndex = {6};
  sub_graph_0_node_5->outputIndex = {17};
  sub_graph_0_node_5->primitive = std::make_unique<schema::PrimitiveT>();
  sub_graph_0_node_5->primitive->value.type = schema::PrimitiveType_PartialFusion;
  auto primitive_sub_graph_0_node_5 = new schema::PartialFusionT;
  primitive_sub_graph_0_node_5->sub_graph_index = 2;
  sub_graph_0_node_5->primitive->value.value = primitive_sub_graph_0_node_5;
  sub_graph_0_node_5->name = "Partial_body";
  meta_graph->nodes.emplace_back(std::move(sub_graph_0_node_5));
  sub_graph_0->nodeIndices.push_back(5);
  MS_LOG(DEBUG) << "node 5";

  // subgraph 0:  node 6     add-after
  auto sub_graph_0_node_6 = std::make_unique<schema::CNodeT>();
  sub_graph_0_node_6->inputIndex = {7, 8};
  sub_graph_0_node_6->outputIndex = {9};
  sub_graph_0_node_6->primitive = std::make_unique<schema::PrimitiveT>();
  sub_graph_0_node_6->primitive->value.type = schema::PrimitiveType_AddFusion;
  auto primitive_sub_graph_0_node_6 = new schema::AddFusionT;
  sub_graph_0_node_6->primitive->value.value = primitive_sub_graph_0_node_6;
  sub_graph_0_node_6->name = "Add-after";
  meta_graph->nodes.emplace_back(std::move(sub_graph_0_node_6));
  sub_graph_0->nodeIndices.push_back(6);
  MS_LOG(DEBUG) << "node 6";

  sub_graph_0->inputIndices = {0};
  sub_graph_0->outputIndices = {9};
  sub_graph_0->tensorIndices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17};

  meta_graph->subGraph.push_back(std::move(sub_graph_0));

  // subgraph 1 ;  node:0   add   cond
  auto sub_graph_1_node_0 = std::make_unique<schema::CNodeT>();
  sub_graph_1_node_0->inputIndex = {16, 10};
  sub_graph_1_node_0->outputIndex = {11};
  sub_graph_1_node_0->primitive = std::make_unique<schema::PrimitiveT>();
  sub_graph_1_node_0->primitive->value.type = schema::PrimitiveType_AddFusion;
  auto primitive_sub_graph_1_node_0 = new schema::AddFusionT;
  sub_graph_1_node_0->primitive->value.value = primitive_sub_graph_1_node_0;
  sub_graph_1_node_0->name = "cond_add";
  meta_graph->nodes.emplace_back(std::move(sub_graph_1_node_0));
  sub_graph_1->nodeIndices.push_back(7);
  MS_LOG(DEBUG) << "node 6";

  // subgraph 1 ;  node:1   Less   cond
  auto sub_graph_1_node_1 = std::make_unique<schema::CNodeT>();
  sub_graph_1_node_1->inputIndex = {11, 12};
  sub_graph_1_node_1->outputIndex = {5};
  sub_graph_1_node_1->primitive = std::make_unique<schema::PrimitiveT>();
  sub_graph_1_node_1->primitive->value.type = schema::PrimitiveType_Less;
  auto primitive_sub_graph_1_node_1 = new schema::LessT;
  sub_graph_1_node_1->primitive->value.value = primitive_sub_graph_1_node_1;
  sub_graph_1_node_1->name = "cond_Less";
  meta_graph->nodes.emplace_back(std::move(sub_graph_1_node_1));
  sub_graph_1->nodeIndices.push_back(8);
  MS_LOG(DEBUG) << "node 7";

  sub_graph_1->inputIndices = {16};
  sub_graph_1->outputIndices = {5};
  sub_graph_1->tensorIndices = {16, 10, 11, 12, 5};
  meta_graph->subGraph.push_back(std::move(sub_graph_1));

  // subgraph 2 ;  node:0   body add-1
  auto sub_graph_2_node_0 = std::make_unique<schema::CNodeT>();
  sub_graph_2_node_0->inputIndex = {6, 13};
  sub_graph_2_node_0->outputIndex = {14};
  sub_graph_2_node_0->primitive = std::make_unique<schema::PrimitiveT>();
  sub_graph_2_node_0->primitive->value.type = schema::PrimitiveType_AddFusion;
  auto primitive_sub_graph_2_node_0 = new schema::AddFusionT;
  sub_graph_2_node_0->primitive->value.value = primitive_sub_graph_2_node_0;
  sub_graph_2_node_0->name = "body_add_1";
  meta_graph->nodes.emplace_back(std::move(sub_graph_2_node_0));
  sub_graph_2->nodeIndices.push_back(9);
  MS_LOG(DEBUG) << "node 8";

  // subgraph 2 ;  node:1   body add-2
  auto sub_graph_2_node_1 = std::make_unique<schema::CNodeT>();
  sub_graph_2_node_1->inputIndex = {14, 15};
  sub_graph_2_node_1->outputIndex = {17};
  sub_graph_2_node_1->primitive = std::make_unique<schema::PrimitiveT>();
  sub_graph_2_node_1->primitive->value.type = schema::PrimitiveType_AddFusion;
  auto primitive_sub_graph_2_node_1 = new schema::AddFusionT;
  sub_graph_2_node_1->primitive->value.value = primitive_sub_graph_2_node_1;
  sub_graph_2_node_1->name = "body_add_2";
  meta_graph->nodes.emplace_back(std::move(sub_graph_2_node_1));
  sub_graph_2->nodeIndices.push_back(10);
  MS_LOG(DEBUG) << "node 9";

  sub_graph_2->inputIndices = {6};
  sub_graph_2->outputIndices = {17};
  sub_graph_2->tensorIndices = {13, 14, 15, 6, 17};

  meta_graph->subGraph.push_back(std::move(sub_graph_2));

  //  -------   tensor    ---------
  //  tensor: 0   before-add input0 <main graph input>
  auto tensor_0 = std::make_unique<schema::TensorT>();
  tensor_0->nodeType = lite::NodeType_ValueNode;
  tensor_0->format = schema::Format_NHWC;
  tensor_0->dataType = TypeId::kNumberTypeFloat32;
  tensor_0->dims = {1};
  tensor_0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(tensor_0));
  MS_LOG(DEBUG) << "tensor 0";

  //  tensor: 1    before-add input1 <const>
  auto tensor_1 = std::make_unique<schema::TensorT>();
  tensor_1->nodeType = lite::NodeType_ValueNode;
  tensor_1->format = schema::Format_NHWC;
  tensor_1->dataType = TypeId::kNumberTypeFloat32;
  tensor_1->dims = {1};
  tensor_1->data.resize(sizeof(float) * 1);
  float input1_data[] = {1};
  memcpy(tensor_1->data.data(), input1_data, sizeof(float) * 1);
  tensor_1->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(tensor_1));
  MS_LOG(DEBUG) << "tensor 1";

  //  tensor: 2  before-add output/partial input
  auto tensor_2 = std::make_unique<schema::TensorT>();
  tensor_2->nodeType = lite::NodeType_Parameter;
  tensor_2->format = schema::Format_NHWC;
  tensor_2->dataType = TypeId::kNumberTypeFloat32;
  tensor_2->dims = {1};
  tensor_2->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(tensor_2));
  MS_LOG(DEBUG) << "tensor 2";

  //  tensor: 3    before-add input1 <const>
  auto tensor_3 = std::make_unique<schema::TensorT>();
  tensor_3->nodeType = lite::NodeType_ValueNode;
  tensor_3->format = schema::Format_NHWC;
  tensor_3->dataType = TypeId::kNumberTypeFloat32;
  tensor_3->dims = {1};
  tensor_3->data.resize(sizeof(float) * 1);
  float tensor_3_data[] = {1};
  memcpy(tensor_3->data.data(), tensor_3_data, sizeof(float) * 1);
  tensor_3->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(tensor_3));
  MS_LOG(DEBUG) << "tensor 3";

  auto tensor_4 = std::make_unique<schema::TensorT>();
  tensor_4->nodeType = lite::NodeType_Parameter;
  tensor_4->format = schema::Format_NHWC;
  tensor_4->dataType = TypeId::kNumberTypeFloat32;
  tensor_4->dims = {1};
  tensor_4->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(tensor_4));
  MS_LOG(DEBUG) << "tensor 4";

  //  tensor :5   partial output <bool>
  auto tensor_5 = std::make_unique<schema::TensorT>();
  tensor_5->nodeType = lite::NodeType_Parameter;
  tensor_5->format = schema::Format_NHWC;
  tensor_5->dataType = TypeId::kNumberTypeBool;
  tensor_5->dims = {1};
  tensor_5->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(tensor_5));
  MS_LOG(DEBUG) << "tensor_4";

  //  tensor: 6 switch true output
  auto tensor_6 = std::make_unique<schema::TensorT>();
  tensor_6->nodeType = lite::NodeType_Parameter;
  tensor_6->format = schema::Format_NHWC;
  tensor_6->dataType = TypeId::kNumberTypeFloat32;
  tensor_6->dims = {1};
  tensor_6->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(tensor_6));
  MS_LOG(DEBUG) << "tensor 6";

  //  tensor: 5  switch False output
  auto tensor_7 = std::make_unique<schema::TensorT>();
  tensor_7->nodeType = lite::NodeType_Parameter;
  tensor_7->format = schema::Format_NHWC;
  tensor_7->dataType = TypeId::kNumberTypeFloat32;
  tensor_7->dims = {1};
  tensor_7->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(tensor_7));
  MS_LOG(DEBUG) << "tensor_7";

  //  tensor: 6  body-add input ,other input is switch true output
  auto tensor_8 = std::make_unique<schema::TensorT>();
  tensor_8->nodeType = lite::NodeType_ValueNode;
  tensor_8->format = schema::Format_NHWC;
  tensor_8->dataType = TypeId::kNumberTypeFloat32;
  tensor_8->dims = {1};
  tensor_8->data.resize(sizeof(float) * 1);
  float tensor_8_data[] = {10};
  memcpy(tensor_8->data.data(), tensor_8_data, sizeof(float) * 1);
  tensor_8->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(tensor_8));
  MS_LOG(DEBUG) << "tensor_8";

  auto tensor_9 = std::make_unique<schema::TensorT>();
  tensor_9->nodeType = lite::NodeType_Parameter;
  tensor_9->format = schema::Format_NHWC;
  tensor_9->dataType = TypeId::kNumberTypeFloat32;
  tensor_9->dims = {1};
  tensor_9->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(tensor_9));
  MS_LOG(DEBUG) << "tensor_9";

  //  tensor: 7  after-add input ,other input is switch false output
  auto tensor_10 = std::make_unique<schema::TensorT>();
  tensor_10->nodeType = lite::NodeType_ValueNode;
  tensor_10->format = schema::Format_NHWC;
  tensor_10->dataType = TypeId::kNumberTypeFloat32;
  tensor_10->dims = {1};
  tensor_10->data.resize(sizeof(float) * 1);
  float tensor_10_data[] = {1};
  memcpy(tensor_10->data.data(), tensor_10_data, sizeof(float) * 1);
  tensor_10->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(tensor_10));
  MS_LOG(DEBUG) << "tensor_10";

  //  tensor: 8   main graph output
  auto tensor_11 = std::make_unique<schema::TensorT>();
  tensor_11->nodeType = lite::NodeType_Parameter;
  tensor_11->format = schema::Format_NHWC;
  tensor_11->dataType = TypeId::kNumberTypeFloat32;
  tensor_11->dims = {1};
  tensor_11->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(tensor_11));
  MS_LOG(DEBUG) << "tensor 11";

  //  tensor: 9  cond-Less  input, other input is tensor 2
  auto tensor_12 = std::make_unique<schema::TensorT>();
  tensor_12->nodeType = lite::NodeType_ValueNode;
  tensor_12->format = schema::Format_NHWC;
  tensor_12->dataType = TypeId::kNumberTypeFloat32;
  tensor_12->dims = {1};
  tensor_12->data.resize(sizeof(float) * 1);
  float tensor_12_data[] = {10};
  memcpy(tensor_12->data.data(), tensor_12_data, sizeof(float) * 1);
  tensor_12->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(tensor_12));
  MS_LOG(DEBUG) << "tensor_12";

  auto tensor_13 = std::make_unique<schema::TensorT>();
  tensor_13->nodeType = lite::NodeType_ValueNode;
  tensor_13->format = schema::Format_NHWC;
  tensor_13->dataType = TypeId::kNumberTypeFloat32;
  tensor_13->dims = {1};
  tensor_13->data.resize(sizeof(float) * 1);
  float tensor_13_data[] = {1};
  memcpy(tensor_13->data.data(), tensor_13_data, sizeof(float) * 1);
  tensor_13->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(tensor_13));
  MS_LOG(DEBUG) << "tensor_13";

  auto tensor_14 = std::make_unique<schema::TensorT>();
  tensor_14->nodeType = lite::NodeType_Parameter;
  tensor_14->format = schema::Format_NHWC;
  tensor_14->dataType = TypeId::kNumberTypeFloat32;
  tensor_14->dims = {1};
  tensor_14->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(tensor_14));
  MS_LOG(DEBUG) << "tensor 14";

  auto tensor_15 = std::make_unique<schema::TensorT>();
  tensor_15->nodeType = lite::NodeType_ValueNode;
  tensor_15->format = schema::Format_NHWC;
  tensor_15->dataType = TypeId::kNumberTypeFloat32;
  tensor_15->dims = {1};
  tensor_15->data.resize(sizeof(float) * 1);
  float tensor_15_data[] = {1};
  memcpy(tensor_15->data.data(), tensor_15_data, sizeof(float) * 1);
  tensor_15->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(tensor_15));
  MS_LOG(DEBUG) << "tensor_15";

  auto tensor_16 = std::make_unique<schema::TensorT>();
  tensor_16->nodeType = lite::NodeType_Parameter;
  tensor_16->format = schema::Format_NHWC;
  tensor_16->dataType = TypeId::kNumberTypeFloat32;
  tensor_16->dims = {1};
  tensor_16->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(tensor_16));
  MS_LOG(DEBUG) << "tensor_16";

  auto tensor_17 = std::make_unique<schema::TensorT>();
  tensor_17->nodeType = lite::NodeType_Parameter;
  tensor_17->format = schema::Format_NHWC;
  tensor_17->dataType = TypeId::kNumberTypeFloat32;
  tensor_17->dims = {1};
  tensor_17->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(tensor_17));
  MS_LOG(DEBUG) << "tensor_17";
  //  -----------------------------------------------------------------------

  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = schema::MetaGraph::Pack(builder, meta_graph.get());
  builder.Finish(offset);
  schema::FinishMetaGraphBuffer(builder, offset);
  size_t size = builder.GetSize();
  const char *content = reinterpret_cast<char *>(builder.GetBufferPointer());

  auto model = std::shared_ptr<lite::Model>(lite::Model::Import(content, size));
  ASSERT_NE(model, nullptr);
  lite::Context context;
  context.thread_num_ = 2;
  auto &cpu_device_ctx = context.device_list_[0];
  cpu_device_ctx.device_info_.cpu_device_info_.cpu_bind_mode_ = lite::MID_CPU;
  cpu_device_ctx.device_info_.cpu_device_info_.enable_float16_ = false;
  auto session = std::shared_ptr<session::LiteSession>(session::LiteSession::CreateSession(&context));
  ASSERT_NE(session, nullptr);
  auto ret = session->CompileGraph(model.get());
  ASSERT_EQ(ret, lite::RET_OK);
  model->Free();
  auto inputs = session->GetInputs();
  ASSERT_EQ(inputs.size(), 1);
  auto input = inputs.front();
  ASSERT_NE(input, nullptr);
  ASSERT_EQ(input->data_type(), kNumberTypeFloat32);
  ASSERT_EQ(input->shape().size(), 1);
  ASSERT_EQ(input->shape().at(0), 1);
  auto in_data = reinterpret_cast<float *>(input->MutableData());
  ASSERT_NE(in_data, nullptr);
  in_data[0] = 1;
  ret = session->RunGraph();
  ASSERT_EQ(ret, lite::RET_OK);
  auto outputs = session->GetOutputs();
  ASSERT_EQ(outputs.size(), 1);
  auto output = outputs.begin()->second;
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(output->data_type(), kNumberTypeFloat32);
  ASSERT_EQ(output->shape().size(), 1);
  ASSERT_EQ(output->shape().at(0), 1);
  auto out_data = reinterpret_cast<float *>(output->MutableData());
  ASSERT_NE(out_data, nullptr);
  ASSERT_EQ(out_data[0], 19);
}
}  // namespace mindspore
