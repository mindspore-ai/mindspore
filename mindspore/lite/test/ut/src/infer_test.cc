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
#include "src/runtime/parallel_executor.h"

namespace mindspore {
class InferTest : public mindspore::CommonTest {
 public:
  InferTest() {}
};

TEST_F(InferTest, TestConvNode) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Conv2DFusion;
  auto primitive = new schema::Conv2DFusionT;
  primitive->pad_mode = schema::PadMode_SAME;
  primitive->in_channel = 3;
  primitive->out_channel = 32;
  primitive->format = schema::Format_NHWC;
  primitive->stride = std::vector<int64_t>{1, 1};
  primitive->kernel_size = std::vector<int64_t>{3, 3};
  primitive->dilation = std::vector<int64_t>{1, 1};
  node->primitive->value.value = primitive;
  node->name = "Conv2D";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {2};

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_ValueNode;
  input0->format = schema::Format_NHWC;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {1, 28, 28, 3};
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto weight = std::make_unique<schema::TensorT>();
  weight->nodeType = lite::NodeType_ValueNode;
  weight->format = schema::Format_KHWC;
  weight->dataType = TypeId::kNumberTypeFloat32;
  weight->dims = {32, 3, 3, 3};

  auto buf = new char *[1];
  //================================================================
  size_t weight_size;
  std::string weight_path = "./test_data/conv/convfp32_weight_32_3_3_3.bin";
  ReadFile(weight_path.c_str(), &weight_size, buf);
  ASSERT_NE(nullptr, buf[0]);
  auto weight_data_temp = reinterpret_cast<float *>(buf[0]);
  ASSERT_NE(nullptr, weight_data_temp);
  weight->data.resize(sizeof(float) * 32 * 3 * 3 * 3);

  //================================================================
  memcpy(weight->data.data(), weight_data_temp, weight_size);
  weight->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(weight));

  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = lite::NodeType_Parameter;
  output->format = schema::Format_NHWC;
  output->dataType = TypeId::kNumberTypeFloat32;
  output->dims = {1, 28, 28, 32};
  output->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(output));

  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = schema::MetaGraph::Pack(builder, meta_graph.get());
  builder.Finish(offset);
  size_t size = builder.GetSize();
  const char *content = reinterpret_cast<char *>(builder.GetBufferPointer());

  auto model = lite::Model::Import(content, size);
  ASSERT_NE(nullptr, model);
  meta_graph.reset();
  content = nullptr;
  auto context = new lite::InnerContext;
  auto &device_list = context->device_list_;
  lite::DeviceContext device_ctx = {lite::DT_CPU, {false, lite::NO_BIND}};
  device_list.push_back(device_ctx);
  context->thread_num_ = 4;
  ASSERT_EQ(lite::RET_OK, context->Init());
  auto session = session::LiteSession::CreateSession(context);
  ASSERT_NE(nullptr, session);
  auto ret = session->CompileGraph(model);
  ASSERT_EQ(lite::RET_OK, ret);
  auto inputs = session->GetInputs();
  ASSERT_EQ(inputs.size(), 1);
  auto inTensor = inputs.front();
  ASSERT_NE(nullptr, inTensor);
  auto data = inTensor->MutableData();
  //===================================================
  size_t input_size;
  std::string input_path = "./test_data/conv/convfp32_input_1_28_28_3.bin";
  ReadFile(input_path.c_str(), &input_size, buf);
  ASSERT_NE(nullptr, buf[0]);
  auto input_data = reinterpret_cast<float *>(buf[0]);
  ASSERT_NE(nullptr, input_data);
  //===================================================
  ASSERT_EQ(input_size, inTensor->Size());
  memcpy(data, input_data, input_size);
  ret = session->RunGraph();
  ASSERT_EQ(lite::RET_OK, ret);
  auto outputs = session->GetOutputs();
  ASSERT_EQ(outputs.size(), 1);
  auto outTensor = outputs.begin()->second;
  ASSERT_NE(nullptr, outTensor);
  ASSERT_EQ(28 * 28 * 32, outTensor->ElementsNum());
  ASSERT_EQ(TypeId::kNumberTypeFloat32, outTensor->data_type());
  auto *outData = reinterpret_cast<float *>(outTensor->MutableData());
  ASSERT_NE(nullptr, outData);
  //===================================================
  size_t output_size;
  std::string output_path = "./test_data/conv/convfp32_out_1_28_28_32.bin";
  ReadFile(output_path.c_str(), &output_size, buf);
  ASSERT_NE(nullptr, buf[0]);
  auto output_data = reinterpret_cast<float *>(buf[0]);
  ASSERT_NE(nullptr, output_data);
  //===================================================
  ASSERT_EQ(output_size, outTensor->Size());
  for (int i = 0; i < outTensor->ElementsNum(); i++) {
    ASSERT_LE((output_data[i] - outData[i]), 0.001);
  }
  MS_LOG(INFO) << "Passed";
}

TEST_F(InferTest, TestAddNode) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_AddFusion;
  auto primitive = new schema::AddFusionT;
  node->primitive->value.value = primitive;
  node->name = "Add";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0, 1};
  meta_graph->outputIndex = {2};

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_ValueNode;
  input0->format = schema::Format_NHWC;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {1, 28, 28, 3};
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto weight = std::make_unique<schema::TensorT>();
  weight->nodeType = lite::NodeType_ValueNode;
  weight->format = schema::Format_KHWC;
  weight->dataType = TypeId::kNumberTypeFloat32;
  weight->dims = {1, 28, 28, 3};

  weight->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(weight));

  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = lite::NodeType_Parameter;
  output->format = schema::Format_NHWC;
  output->dataType = TypeId::kNumberTypeFloat32;
  output->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(output));

  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = schema::MetaGraph::Pack(builder, meta_graph.get());
  builder.Finish(offset);
  size_t size = builder.GetSize();
  const char *content = reinterpret_cast<char *>(builder.GetBufferPointer());

  auto model = lite::Model::Import(content, size);
  ASSERT_NE(nullptr, model);
  meta_graph.reset();
  content = nullptr;
  auto context = new lite::InnerContext;
  auto &device_list = context->device_list_;
  lite::DeviceContext device_ctx = {lite::DT_CPU, {false, lite::NO_BIND}};
  device_list.push_back(device_ctx);
  context->thread_num_ = 4;
  ASSERT_EQ(lite::RET_OK, context->Init());
  auto session = session::LiteSession::CreateSession(context);
  ASSERT_NE(nullptr, session);
  auto ret = session->CompileGraph(model);
  ASSERT_EQ(lite::RET_OK, ret);
  auto inputs = session->GetInputs();
  ASSERT_EQ(inputs.size(), 2);
  auto inTensor = inputs.front();
  ASSERT_NE(nullptr, inTensor);
  (void)inTensor->MutableData();
  auto inTensor1 = inputs.back();
  ASSERT_NE(nullptr, inTensor1);
  (void)inTensor1->MutableData();
  ret = session->RunGraph();
  ASSERT_EQ(lite::RET_OK, ret);
  auto outputs = session->GetOutputs();
  ASSERT_EQ(outputs.size(), 1);
  auto outTensor = outputs.begin()->second;
  ASSERT_NE(nullptr, outTensor);
  ASSERT_EQ(28 * 28 * 3, outTensor->ElementsNum());
  ASSERT_EQ(TypeId::kNumberTypeFloat32, outTensor->data_type());
  auto *outData = reinterpret_cast<float *>(outTensor->MutableData());
  ASSERT_NE(nullptr, outData);
  MS_LOG(INFO) << "Passed";
}

class SessionWithParallelExecutor : public lite::LiteSession {
 public:
  int Init(lite::InnerContext *context) {
    lite::LiteSession::Init(context);
    delete this->executor_;
    this->executor_ = new mindspore::lite::ParallelExecutor();
    return 0;
  }
};

TEST_F(InferTest, TestParallelExecutor) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_AddFusion;
  auto primitive = new schema::AddFusionT;
  node->primitive->value.value = primitive;
  node->name = "Add";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0, 1};
  meta_graph->outputIndex = {2};

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_ValueNode;
  input0->format = schema::Format_NHWC;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {1, 28, 28, 3};
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto weight = std::make_unique<schema::TensorT>();
  weight->nodeType = lite::NodeType_ValueNode;
  weight->format = schema::Format_NHWC;
  weight->dataType = TypeId::kNumberTypeFloat32;
  weight->dims = {1, 28, 28, 3};

  weight->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(weight));

  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = lite::NodeType_Parameter;
  output->format = schema::Format_NHWC;
  output->dataType = TypeId::kNumberTypeFloat32;
  output->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(output));

  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = schema::MetaGraph::Pack(builder, meta_graph.get());
  builder.Finish(offset);
  size_t size = builder.GetSize();
  const char *content = reinterpret_cast<char *>(builder.GetBufferPointer());

  auto model = lite::Model::Import(content, size);
  ASSERT_NE(nullptr, model);
  meta_graph.reset();
  content = nullptr;
  auto context = new lite::InnerContext;
  auto &device_list = context->device_list_;
  lite::DeviceContext device_ctx = {lite::DT_CPU, {false, lite::NO_BIND}};
  device_list.push_back(device_ctx);
  context->thread_num_ = 4;
  ASSERT_EQ(lite::RET_OK, context->Init());
  auto session = new SessionWithParallelExecutor();
  session->Init(context);
  ASSERT_NE(nullptr, session);
  auto ret = session->CompileGraph(model);
  ASSERT_EQ(lite::RET_OK, ret);
  auto inputs = session->GetInputs();
  ASSERT_EQ(inputs.size(), 2);
  auto inTensor = inputs.front();
  ASSERT_NE(nullptr, inTensor);
  (void)inTensor->MutableData();
  auto inTensor1 = inputs.back();
  ASSERT_NE(nullptr, inTensor1);
  (void)inTensor1->MutableData();
  ret = session->RunGraph();
  ASSERT_EQ(lite::RET_OK, ret);
  auto outputs = session->GetOutputs();
  ASSERT_EQ(outputs.size(), 1);
  auto outTensor = outputs.begin()->second;
  ASSERT_NE(nullptr, outTensor);
  ASSERT_EQ(28 * 28 * 3, outTensor->ElementsNum());
  ASSERT_EQ(TypeId::kNumberTypeFloat32, outTensor->data_type());
  auto *outData = reinterpret_cast<float *>(outTensor->MutableData());
  ASSERT_NE(nullptr, outData);
  MS_LOG(INFO) << "Passed";
}

TEST_F(InferTest, TestModel) {
  auto buf = new char *[1];
  size_t model_size;
  std::string model_path = "./models/model_hebing_3branch.ms";
  ReadFile(model_path.c_str(), &model_size, buf);
  ASSERT_NE(nullptr, buf[0]);

  auto model = lite::Model::Import(buf[0], model_size);
  ASSERT_NE(nullptr, model);
  delete[] buf[0];
  auto context = new lite::InnerContext;
  context->device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = lite::NO_BIND;
  context->thread_num_ = 4;
  ASSERT_EQ(lite::RET_OK, context->Init());
  auto session = session::LiteSession::CreateSession(context);
  ASSERT_NE(nullptr, session);
  auto ret = session->CompileGraph(model);
  ASSERT_EQ(lite::RET_OK, ret);
  auto inputs = session->GetInputs();
  ASSERT_EQ(inputs.size(), 1);
  auto inTensor = inputs.front();
  ASSERT_NE(nullptr, inTensor);
  (void)inTensor->MutableData();
  ret = session->RunGraph();
  ASSERT_EQ(lite::RET_OK, ret);
  auto outputs = session->GetOutputs();
  MS_LOG(INFO) << "Passed";
}
}  // namespace mindspore
