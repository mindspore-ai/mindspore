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
#include "include/model.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/lite_session.h"
#include "src/runtime/parallel_executor.h"
#include "tools/common/storage.h"
#include "include/version.h"

namespace mindspore {
class SubGraphTest : public mindspore::CommonTest {
 public:
  SubGraphTest() = default;
};

TEST_F(SubGraphTest, RecursiveSubGraphTest) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->allTensors.resize(16);
  {    // subgraph-0
    {  // add-0
      auto add_0 = std::make_unique<schema::CNodeT>();
      add_0->inputIndex = {0, 1};
      add_0->outputIndex = {2};
      add_0->primitive = std::make_unique<schema::PrimitiveT>();
      add_0->primitive->value.type = schema::PrimitiveType_AddFusion;
      auto add_0_prim = new schema::AddFusionT;
      add_0_prim->activation_type = schema::ActivationType_NO_ACTIVATION;
      add_0->primitive->value.value = add_0_prim;
      add_0->name = "Add0";
      auto tensor_0 = std::make_unique<schema::TensorT>();
      tensor_0->nodeType = lite::NodeType_ValueNode;
      tensor_0->format = schema::Format_NHWC;
      tensor_0->dataType = TypeId::kNumberTypeFloat32;
      tensor_0->dims = {1};
      auto tensor_1 = std::make_unique<schema::TensorT>();
      tensor_1->nodeType = lite::NodeType_ValueNode;
      tensor_1->format = schema::Format_NHWC;
      tensor_1->dataType = TypeId::kNumberTypeFloat32;
      tensor_1->dims = {1};
      tensor_1->data.resize(sizeof(float));
      auto data1 = reinterpret_cast<float *>(tensor_1->data.data());
      ASSERT_NE(data1, nullptr);
      data1[0] = 1;
      auto tensor_2 = std::make_unique<schema::TensorT>();
      tensor_2->nodeType = lite::NodeType_Parameter;
      tensor_2->format = schema::Format_NHWC;
      tensor_2->dataType = TypeId::kNumberTypeFloat32;
      meta_graph->nodes.emplace_back(std::move(add_0));
      meta_graph->allTensors[0] = std::move(tensor_0);
      meta_graph->allTensors[1] = std::move(tensor_1);
      meta_graph->allTensors[2] = std::move(tensor_2);
    }
    {  // add-1
      auto add_1 = std::make_unique<schema::CNodeT>();
      add_1->inputIndex = {2, 3};
      add_1->outputIndex = {4};
      add_1->primitive = std::make_unique<schema::PrimitiveT>();
      add_1->primitive->value.type = schema::PrimitiveType_AddFusion;
      auto add_1_prim = new schema::AddFusionT;
      add_1_prim->activation_type = schema::ActivationType_NO_ACTIVATION;
      add_1->primitive->value.value = add_1_prim;
      add_1->name = "Add1";
      auto tensor_3 = std::make_unique<schema::TensorT>();
      tensor_3->nodeType = lite::NodeType_ValueNode;
      tensor_3->format = schema::Format_NHWC;
      tensor_3->dataType = TypeId::kNumberTypeFloat32;
      tensor_3->dims = {1};
      tensor_3->data.resize(sizeof(float));
      auto data3 = reinterpret_cast<float *>(tensor_3->data.data());
      ASSERT_NE(data3, nullptr);
      data3[0] = 1;
      auto tensor_4 = std::make_unique<schema::TensorT>();
      tensor_4->nodeType = lite::NodeType_Parameter;
      tensor_4->format = schema::Format_NHWC;
      tensor_4->dataType = TypeId::kNumberTypeFloat32;
      meta_graph->nodes.emplace_back(std::move(add_1));
      meta_graph->allTensors[3] = std::move(tensor_3);
      meta_graph->allTensors[4] = std::move(tensor_4);
    }
    {  // partial cond
      auto partial_cond = std::make_unique<schema::CNodeT>();
      partial_cond->inputIndex = {4};
      partial_cond->outputIndex = {9};
      partial_cond->primitive = std::make_unique<schema::PrimitiveT>();
      partial_cond->primitive->value.type = schema::PrimitiveType_PartialFusion;
      auto partial_cond_prim = new schema::PartialFusionT;
      partial_cond_prim->sub_graph_index = 1;
      partial_cond->primitive->value.value = partial_cond_prim;
      partial_cond->name = "partial_cond";
      meta_graph->nodes.emplace_back(std::move(partial_cond));
    }
    {  // add-5
      auto add_5 = std::make_unique<schema::CNodeT>();
      add_5->inputIndex = {9, 13};
      add_5->outputIndex = {14};
      add_5->primitive = std::make_unique<schema::PrimitiveT>();
      add_5->primitive->value.type = schema::PrimitiveType_AddFusion;
      auto add_5_prim = new schema::AddFusionT;
      add_5_prim->activation_type = schema::ActivationType_NO_ACTIVATION;
      add_5->primitive->value.value = add_5_prim;
      add_5->name = "Add5";
      auto tensor_13 = std::make_unique<schema::TensorT>();
      tensor_13->nodeType = lite::NodeType_ValueNode;
      tensor_13->format = schema::Format_NHWC;
      tensor_13->dataType = TypeId::kNumberTypeFloat32;
      tensor_13->dims = {1};
      tensor_13->data.resize(sizeof(float));
      auto data13 = reinterpret_cast<float *>(tensor_13->data.data());
      ASSERT_NE(data13, nullptr);
      data13[0] = 1;
      auto tensor_14 = std::make_unique<schema::TensorT>();
      tensor_14->nodeType = lite::NodeType_Parameter;
      tensor_14->format = schema::Format_NHWC;
      tensor_14->dataType = TypeId::kNumberTypeFloat32;
      meta_graph->nodes.emplace_back(std::move(add_5));
      meta_graph->allTensors[13] = std::move(tensor_13);
      meta_graph->allTensors[14] = std::move(tensor_14);
    }
    auto sub_graph_0 = std::make_unique<schema::SubGraphT>();
    sub_graph_0->name = "main_graph";
    sub_graph_0->inputIndices = {0};
    sub_graph_0->outputIndices = {14};
    sub_graph_0->nodeIndices = {0, 1, 2, 3};
    sub_graph_0->tensorIndices = {0, 1, 2, 3, 4, 9, 13, 14};
    meta_graph->subGraph.emplace_back(std::move(sub_graph_0));
  }
  {    // subgraph-1
    {  // add-2
      auto add_2 = std::make_unique<schema::CNodeT>();
      add_2->inputIndex = {4, 5};
      add_2->outputIndex = {6};
      add_2->primitive = std::make_unique<schema::PrimitiveT>();
      add_2->primitive->value.type = schema::PrimitiveType_AddFusion;
      auto add_2_prim = new schema::AddFusionT;
      add_2_prim->activation_type = schema::ActivationType_NO_ACTIVATION;
      add_2->primitive->value.value = add_2_prim;
      add_2->name = "Add2";
      auto tensor_5 = std::make_unique<schema::TensorT>();
      tensor_5->nodeType = lite::NodeType_ValueNode;
      tensor_5->format = schema::Format_NHWC;
      tensor_5->dataType = TypeId::kNumberTypeFloat32;
      tensor_5->dims = {1};
      tensor_5->data.resize(sizeof(float));
      auto data5 = reinterpret_cast<float *>(tensor_5->data.data());
      ASSERT_NE(data5, nullptr);
      data5[0] = 1;
      auto tensor_6 = std::make_unique<schema::TensorT>();
      tensor_6->nodeType = lite::NodeType_Parameter;
      tensor_6->format = schema::Format_NHWC;
      tensor_6->dataType = TypeId::kNumberTypeFloat32;
      meta_graph->nodes.emplace_back(std::move(add_2));
      meta_graph->allTensors[5] = std::move(tensor_5);
      meta_graph->allTensors[6] = std::move(tensor_6);
    }
    {  // less
      auto less = std::make_unique<schema::CNodeT>();
      less->inputIndex = {6, 15};
      less->outputIndex = {7};
      less->primitive = std::make_unique<schema::PrimitiveT>();
      less->primitive->value.type = schema::PrimitiveType_Less;
      auto less_prim = new schema::LessT;
      less->primitive->value.value = less_prim;
      less->name = "less";
      auto tensor_15 = std::make_unique<schema::TensorT>();
      tensor_15->nodeType = lite::NodeType_ValueNode;
      tensor_15->format = schema::Format_NHWC;
      tensor_15->dataType = TypeId::kNumberTypeFloat32;
      tensor_15->dims = {1};
      tensor_15->data.resize(sizeof(float));
      auto data15 = reinterpret_cast<float *>(tensor_15->data.data());
      ASSERT_NE(data15, nullptr);
      data15[0] = 1;
      auto tensor_7 = std::make_unique<schema::TensorT>();
      tensor_7->nodeType = lite::NodeType_Parameter;
      tensor_7->format = schema::Format_NHWC;
      tensor_7->dataType = TypeId::kNumberTypeFloat32;
      meta_graph->nodes.emplace_back(std::move(less));
      meta_graph->allTensors[7] = std::move(tensor_7);
      meta_graph->allTensors[15] = std::move(tensor_15);
    }
    {  // switch
      auto switchop = std::make_unique<schema::CNodeT>();
      switchop->inputIndex = {7, 4};
      switchop->outputIndex = {8, 9};
      switchop->primitive = std::make_unique<schema::PrimitiveT>();
      switchop->primitive->value.type = schema::PrimitiveType_Switch;
      auto switch_prim = new schema::SwitchT;
      switchop->primitive->value.value = switch_prim;
      switchop->name = "switch";
      auto tensor_8 = std::make_unique<schema::TensorT>();
      tensor_8->nodeType = lite::NodeType_Parameter;
      tensor_8->format = schema::Format_NHWC;
      tensor_8->dataType = TypeId::kNumberTypeFloat32;
      auto tensor_9 = std::make_unique<schema::TensorT>();
      tensor_9->nodeType = lite::NodeType_Parameter;
      tensor_9->format = schema::Format_NHWC;
      tensor_9->dataType = TypeId::kNumberTypeFloat32;
      meta_graph->nodes.emplace_back(std::move(switchop));
      meta_graph->allTensors[8] = std::move(tensor_8);
      meta_graph->allTensors[9] = std::move(tensor_9);
    }
    {  // partial body
      auto partial_body = std::make_unique<schema::CNodeT>();
      partial_body->inputIndex = {8};
      partial_body->outputIndex = {4};
      partial_body->primitive = std::make_unique<schema::PrimitiveT>();
      partial_body->primitive->value.type = schema::PrimitiveType_PartialFusion;
      auto partial_body_prim = new schema::PartialFusionT;
      partial_body_prim->sub_graph_index = 2;
      partial_body->primitive->value.value = partial_body_prim;
      partial_body->name = "partial_body";
      meta_graph->nodes.emplace_back(std::move(partial_body));
    }
    auto sub_graph_1 = std::make_unique<schema::SubGraphT>();
    sub_graph_1->name = "while_cond";
    sub_graph_1->inputIndices = {4};
    sub_graph_1->outputIndices = {9};
    sub_graph_1->nodeIndices = {4, 5, 6, 7};
    sub_graph_1->tensorIndices = {4, 5, 6, 7, 8, 9, 15};
    meta_graph->subGraph.emplace_back(std::move(sub_graph_1));
  }
  {    // subgraph-2
    {  // add-3
      auto add_3 = std::make_unique<schema::CNodeT>();
      add_3->inputIndex = {8, 10};
      add_3->outputIndex = {11};
      add_3->primitive = std::make_unique<schema::PrimitiveT>();
      add_3->primitive->value.type = schema::PrimitiveType_AddFusion;
      auto add_3_prim = new schema::AddFusionT;
      add_3_prim->activation_type = schema::ActivationType_NO_ACTIVATION;
      add_3->primitive->value.value = add_3_prim;
      add_3->name = "Add3";
      auto tensor_10 = std::make_unique<schema::TensorT>();
      tensor_10->nodeType = lite::NodeType_ValueNode;
      tensor_10->format = schema::Format_NHWC;
      tensor_10->dataType = TypeId::kNumberTypeFloat32;
      tensor_10->dims = {1};
      tensor_10->data.resize(sizeof(float));
      auto data10 = reinterpret_cast<float *>(tensor_10->data.data());
      ASSERT_NE(data10, nullptr);
      data10[0] = 1;
      auto tensor_11 = std::make_unique<schema::TensorT>();
      tensor_11->nodeType = lite::NodeType_Parameter;
      tensor_11->format = schema::Format_NHWC;
      tensor_11->dataType = TypeId::kNumberTypeFloat32;
      meta_graph->nodes.emplace_back(std::move(add_3));
      meta_graph->allTensors[10] = std::move(tensor_10);
      meta_graph->allTensors[11] = std::move(tensor_11);
    }
    {  // add-4
      auto add_4 = std::make_unique<schema::CNodeT>();
      add_4->inputIndex = {11, 12};
      add_4->outputIndex = {4};
      add_4->primitive = std::make_unique<schema::PrimitiveT>();
      add_4->primitive->value.type = schema::PrimitiveType_AddFusion;
      auto add_4_prim = new schema::AddFusionT;
      add_4_prim->activation_type = schema::ActivationType_NO_ACTIVATION;
      add_4->primitive->value.value = add_4_prim;
      add_4->name = "Add4";
      auto tensor_12 = std::make_unique<schema::TensorT>();
      tensor_12->nodeType = lite::NodeType_ValueNode;
      tensor_12->format = schema::Format_NHWC;
      tensor_12->dataType = TypeId::kNumberTypeFloat32;
      tensor_12->dims = {1};
      tensor_12->data.resize(sizeof(float));
      auto data12 = reinterpret_cast<float *>(tensor_12->data.data());
      ASSERT_NE(data12, nullptr);
      data12[0] = 1;
      meta_graph->nodes.emplace_back(std::move(add_4));
      meta_graph->allTensors[12] = std::move(tensor_12);
    }
    {  // partial cond
      auto partial_cond = std::make_unique<schema::CNodeT>();
      partial_cond->inputIndex = {4};
      partial_cond->outputIndex = {9};
      partial_cond->primitive = std::make_unique<schema::PrimitiveT>();
      partial_cond->primitive->value.type = schema::PrimitiveType_PartialFusion;
      auto partial_cond_prim = new schema::PartialFusionT;
      partial_cond_prim->sub_graph_index = 1;
      partial_cond->primitive->value.value = partial_cond_prim;
      partial_cond->name = "partial_cond1";
      meta_graph->nodes.emplace_back(std::move(partial_cond));
    }
    auto sub_graph_2 = std::make_unique<schema::SubGraphT>();
    sub_graph_2->name = "while_body";
    sub_graph_2->inputIndices = {8};
    sub_graph_2->outputIndices = {9};
    sub_graph_2->nodeIndices = {8, 9, 10};
    sub_graph_2->tensorIndices = {8, 10, 11, 12, 4, 9};
    meta_graph->subGraph.emplace_back(std::move(sub_graph_2));
  }
  meta_graph->name = "graph";
  meta_graph->version = lite::Version();
  //  -----------------------------------------------------------------------
  lite::Storage::Save(*meta_graph,
                      "/mnt/data/workspace/OpenAI/Huawei/mindspore/mindspore/lite/my_test/models/recursive_subgraph");
  //  -----------------------------------------------------------------------
  size_t size = 0;
  char *graph_buf = lite::ReadFile(
    "/mnt/data/workspace/OpenAI/Huawei/mindspore/mindspore/lite/my_test/models/recursive_subgraph.ms", &size);
  ASSERT_NE(graph_buf, nullptr);

  auto model = std::shared_ptr<lite::Model>(lite::Model::Import(graph_buf, size));
  ASSERT_NE(model, nullptr);
  delete[](graph_buf);
  lite::Context context;
  auto &cpu_device_ctx = context.device_list_[0];
  cpu_device_ctx.device_info_.cpu_device_info_.cpu_bind_mode_ = lite::MID_CPU;
  context.thread_num_ = 2;
  auto session = std::shared_ptr<session::LiteSession>(lite::LiteSession::CreateSession(&context));
  ASSERT_NE(session, nullptr);
  auto ret = session->CompileGraph(model.get());
  ASSERT_EQ(ret, lite::RET_OK);
  auto inputs = session->GetInputs();
  for (auto *input : inputs) {
    (void)input->MutableData();
  }
  ret = session->RunGraph();
  ASSERT_EQ(ret, lite::RET_OK);
}
}  // namespace mindspore
