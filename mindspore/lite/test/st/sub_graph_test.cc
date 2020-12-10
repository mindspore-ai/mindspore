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
  SubGraphTest() {}
};

TEST_F(SubGraphTest, RecursiveSubGraphTest) {
  // add0 partial1 2 3 tensor0 1 2
  auto add_0 = std::make_unique<schema::CNodeT>();
  add_0->inputIndex = {0, 1};
  add_0->outputIndex = {2};
  add_0->primitive = std::make_unique<schema::PrimitiveT>();
  add_0->primitive->value.type = schema::PrimitiveType_Add;
  auto add_0_prim = new schema::AddT;
  add_0_prim->activationType = schema::ActivationType_NO_ACTIVATION;
  add_0->primitive->value.value = add_0_prim;
  add_0->name = "Add0";
  auto partial_1 = std::make_unique<schema::CNodeT>();
  partial_1->inputIndex = {2};
  partial_1->outputIndex = {7};
  partial_1->primitive = std::make_unique<schema::PrimitiveT>();
  partial_1->primitive->value.type = schema::PrimitiveType_Partial;
  auto partial_1_prim = new schema::PartialT;
  partial_1_prim->subGraphIndex = 1;
  partial_1->primitive->value.value = partial_1_prim;
  partial_1->name = "Partial1";
  auto partial_2 = std::make_unique<schema::CNodeT>();
  partial_2->inputIndex = {2};
  partial_2->outputIndex = {7};
  partial_2->primitive = std::make_unique<schema::PrimitiveT>();
  partial_2->primitive->value.type = schema::PrimitiveType_Partial;
  auto partial_2_prim = new schema::PartialT;
  partial_2_prim->subGraphIndex = 2;
  partial_2->primitive->value.value = partial_2_prim;
  partial_2->name = "Partial2";
  auto partial_3 = std::make_unique<schema::CNodeT>();
  partial_3->inputIndex = {4, 6};
  partial_3->outputIndex = {7};
  partial_3->primitive = std::make_unique<schema::PrimitiveT>();
  partial_3->primitive->value.type = schema::PrimitiveType_Partial;
  auto partial_3_prim = new schema::PartialT;
  partial_3_prim->subGraphIndex = 3;
  partial_3->primitive->value.value = partial_3_prim;
  partial_3->name = "Partial3";
  auto tensor_0 = std::make_unique<schema::TensorT>();
  tensor_0->nodeType = schema::NodeType::NodeType_Parameter;
  tensor_0->format = schema::Format_NHWC;
  tensor_0->dataType = TypeId::kNumberTypeFloat32;
  tensor_0->dims = {1, 2};
  auto tensor_1 = std::make_unique<schema::TensorT>();
  tensor_1->nodeType = schema::NodeType::NodeType_ValueNode;
  tensor_1->format = schema::Format_NHWC;
  tensor_1->dataType = TypeId::kNumberTypeFloat32;
  tensor_1->dims = {1, 2};
  auto tensor_2 = std::make_unique<schema::TensorT>();
  tensor_2->nodeType = schema::NodeType::NodeType_Parameter;
  tensor_2->format = schema::Format_NHWC;
  tensor_2->dataType = TypeId::kNumberTypeFloat32;
  auto sub_graph_0 = std::make_unique<schema::SubGraphT>();
  sub_graph_0->name = "main_graph";
  sub_graph_0->inputIndices = {0};
  sub_graph_0->outputIndices = {7};
  sub_graph_0->nodeIndices = {0, 1, 2};
  sub_graph_0->tensorIndices = {0, 1, 2, 7};
  // add1 tensor3 4
  auto add_1 = std::make_unique<schema::CNodeT>();
  add_1->inputIndex = {2, 3};
  add_1->outputIndex = {4};
  add_1->primitive = std::make_unique<schema::PrimitiveT>();
  add_1->primitive->value.type = schema::PrimitiveType_Add;
  auto add_1_prim = new schema::AddT;
  add_1_prim->activationType = schema::ActivationType_NO_ACTIVATION;
  add_1->primitive->value.value = add_1_prim;
  add_1->name = "Add1";
  auto tensor_3 = std::make_unique<schema::TensorT>();
  tensor_3->nodeType = schema::NodeType::NodeType_ValueNode;
  tensor_3->format = schema::Format_NHWC;
  tensor_3->dataType = TypeId::kNumberTypeFloat32;
  tensor_3->dims = {1, 2};
  auto tensor_4 = std::make_unique<schema::TensorT>();
  tensor_4->nodeType = schema::NodeType::NodeType_Parameter;
  tensor_4->format = schema::Format_NHWC;
  tensor_4->dataType = TypeId::kNumberTypeFloat32;
  auto sub_graph_1 = std::make_unique<schema::SubGraphT>();
  sub_graph_1->name = "sub_graph_1";
  sub_graph_1->inputIndices = {2};
  sub_graph_1->outputIndices = {7};
  sub_graph_1->nodeIndices = {4, 3};
  sub_graph_1->tensorIndices = {2, 3, 4, 7};
  // add2 tensor5 6
  auto add_2 = std::make_unique<schema::CNodeT>();
  add_2->inputIndex = {2, 5};
  add_2->outputIndex = {6};
  add_2->primitive = std::make_unique<schema::PrimitiveT>();
  add_2->primitive->value.type = schema::PrimitiveType_Add;
  auto add_2_prim = new schema::AddT;
  add_2_prim->activationType = schema::ActivationType_NO_ACTIVATION;
  add_2->primitive->value.value = add_2_prim;
  add_2->name = "Add2";
  auto tensor_5 = std::make_unique<schema::TensorT>();
  tensor_5->nodeType = schema::NodeType::NodeType_ValueNode;
  tensor_5->format = schema::Format_NHWC;
  tensor_5->dataType = TypeId::kNumberTypeFloat32;
  tensor_5->dims = {1, 2};
  auto tensor_6 = std::make_unique<schema::TensorT>();
  tensor_6->nodeType = schema::NodeType::NodeType_Parameter;
  tensor_6->format = schema::Format_NHWC;
  tensor_6->dataType = TypeId::kNumberTypeFloat32;
  auto sub_graph_2 = std::make_unique<schema::SubGraphT>();
  sub_graph_2->name = "sub_graph_2";
  sub_graph_2->inputIndices = {2};
  sub_graph_2->outputIndices = {7};
  sub_graph_2->nodeIndices = {5, 3};
  sub_graph_2->tensorIndices = {2, 5, 6, 7};
  // add3 tensor7
  auto add_3 = std::make_unique<schema::CNodeT>();
  add_3->inputIndex = {4, 6};
  add_3->outputIndex = {7};
  add_3->primitive = std::make_unique<schema::PrimitiveT>();
  add_3->primitive->value.type = schema::PrimitiveType_Add;
  auto add_3_prim = new schema::AddT;
  add_3_prim->activationType = schema::ActivationType_NO_ACTIVATION;
  add_3->primitive->value.value = add_3_prim;
  add_3->name = "Add3";
  auto tensor_7 = std::make_unique<schema::TensorT>();
  tensor_7->nodeType = schema::NodeType::NodeType_Parameter;
  tensor_7->format = schema::Format_NHWC;
  tensor_7->dataType = TypeId::kNumberTypeFloat32;
  auto sub_graph_3 = std::make_unique<schema::SubGraphT>();
  sub_graph_3->name = "sub_graph_3";
  sub_graph_3->inputIndices = {4, 6};
  sub_graph_3->outputIndices = {7};
  sub_graph_3->nodeIndices = {6};
  sub_graph_3->tensorIndices = {4, 6, 7};

  // make graph
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  meta_graph->nodes.emplace_back(std::move(add_0));
  meta_graph->nodes.emplace_back(std::move(partial_1));
  meta_graph->nodes.emplace_back(std::move(partial_2));
  meta_graph->nodes.emplace_back(std::move(partial_3));
  meta_graph->nodes.emplace_back(std::move(add_1));
  meta_graph->nodes.emplace_back(std::move(add_2));
  meta_graph->nodes.emplace_back(std::move(add_3));
  meta_graph->allTensors.emplace_back(std::move(tensor_0));
  meta_graph->allTensors.emplace_back(std::move(tensor_1));
  meta_graph->allTensors.emplace_back(std::move(tensor_2));
  meta_graph->allTensors.emplace_back(std::move(tensor_3));
  meta_graph->allTensors.emplace_back(std::move(tensor_4));
  meta_graph->allTensors.emplace_back(std::move(tensor_5));
  meta_graph->allTensors.emplace_back(std::move(tensor_6));
  meta_graph->allTensors.emplace_back(std::move(tensor_7));
  meta_graph->subGraph.emplace_back(std::move(sub_graph_0));
  meta_graph->subGraph.emplace_back(std::move(sub_graph_1));
  meta_graph->subGraph.emplace_back(std::move(sub_graph_2));
  meta_graph->subGraph.emplace_back(std::move(sub_graph_3));
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
