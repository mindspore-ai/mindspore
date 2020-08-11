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
#include "mindspore/lite/schema/inner/model_generated.h"
#include "mindspore/lite/include/model.h"
#include "common/common_test.h"
#include "include/lite_session.h"
#include "include/context.h"
#include "include/errorcode.h"
#include "mindspore/core/utils/log_adapter.h"

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
  node->primitive->value.type = schema::PrimitiveType_Conv2D;
  auto primitive = new schema::Conv2DT;
  primitive->padMode = schema::PadMode_SAME;
  primitive->channelIn = 3;
  primitive->channelOut = 32;
  primitive->format = schema::Format_NHWC;
  primitive->strideH = 1;
  primitive->strideW = 1;
  primitive->kernelH = 3;
  primitive->kernelW = 3;
  primitive->dilateH = 1;
  primitive->dilateW = 1;
  node->primitive->value.value = primitive;
  node->name = "Conv2D";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {2};

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = schema::NodeType::NodeType_ValueNode;
  input0->format = schema::Format_NHWC;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {1, 28, 28, 3};
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto weight = std::make_unique<schema::TensorT>();
  weight->nodeType = schema::NodeType::NodeType_ValueNode;
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
  output->nodeType = schema::NodeType::NodeType_Parameter;
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
  auto context = new lite::Context;
  context->cpu_bind_mode_ = lite::NO_BIND;
  context->device_ctx_.type = lite::DT_CPU;
  context->thread_num_ = 4;
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
  ASSERT_EQ(outputs.begin()->second.size(), 1);
  auto outTensor = outputs.begin()->second.front();
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
  for (size_t i = 0; i < outTensor->ElementsNum(); i++) {
    ASSERT_LE((output_data[i]- outData[i]), 0.001);
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
  node->primitive->value.type = schema::PrimitiveType_Add;
  auto primitive = new schema::AddT;
  node->primitive->value.value = primitive;
  node->name = "Add";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0, 1};
  meta_graph->outputIndex = {2};

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = schema::NodeType::NodeType_ValueNode;
  input0->format = schema::Format_NHWC;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {1, 28, 28, 3};
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto weight = std::make_unique<schema::TensorT>();
  weight->nodeType = schema::NodeType::NodeType_ValueNode;
  weight->format = schema::Format_KHWC;
  weight->dataType = TypeId::kNumberTypeFloat32;
  weight->dims = {1, 28, 28, 3};

  weight->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(weight));

  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = schema::NodeType::NodeType_Parameter;
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
  auto context = new lite::Context;
  context->cpu_bind_mode_ = lite::NO_BIND;
  context->device_ctx_.type = lite::DT_GPU;
  context->thread_num_ = 4;
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
  ASSERT_EQ(outputs.begin()->second.size(), 1);
  auto outTensor = outputs.begin()->second.front();
  ASSERT_NE(nullptr, outTensor);
  ASSERT_EQ(28 * 28 * 3, outTensor->ElementsNum());
  ASSERT_EQ(TypeId::kNumberTypeFloat32, outTensor->data_type());
  auto *outData = reinterpret_cast<float *>(outTensor->MutableData());
  ASSERT_NE(nullptr, outData);
  // //===================================================
  // size_t output_size;
  // std::string output_path = "./convfp32_out_1_28_28_32.bin";
  // ReadFile(output_path.c_str(), &output_size, buf);
  // ASSERT_NE(nullptr, buf[0]);
  // auto output_data = reinterpret_cast<float *>(buf[0]);
  // ASSERT_NE(nullptr, output_data);
  // //===================================================
  // ASSERT_EQ(output_size, outTensor->Size());
  // for (size_t i = 0; i < outTensor->ElementsNum(); i++) {
  //   ASSERT_EQ(output_data[i], outData[i]);
  // }
  MS_LOG(INFO) << "Passed";
}

TEST_F(InferTest, TestModel) {
  auto buf = new char *[1];
  size_t model_size;
  std::string model_path = "./model.ms";
  ReadFile(model_path.c_str(), &model_size, buf);
  ASSERT_NE(nullptr, buf[0]);

  auto model = lite::Model::Import(buf[0], model_size);
  ASSERT_NE(nullptr, model);
  delete[] buf[0];
  auto context = new lite::Context;
  context->cpu_bind_mode_ = lite::NO_BIND;
  context->device_ctx_.type = lite::DT_CPU;
  context->thread_num_ = 4;
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

// TEST_F(TrainTest, TestMultiNode) {
//  auto msGraph = std::make_shared<schema::GraphDefT>();
//  msGraph->name = "graph";
//  auto msSubgraph = std::make_unique<schema::SubGraphDefT>();
//  msSubgraph->name = "subGraph";
//
//  auto conv = std::make_unique<schema::OpDefT>();
//  conv->inputIndex = {0, 1};
//  conv->outputIndex = {2};
//  conv->attr.type = schema::OpT_Conv2D;
//  auto conv_attr = new schema::Conv2DT;
//  conv_attr->padMode = schema::PadMode_SAME;
//  conv_attr->format = schema::Format_NHWC;
//  conv_attr->strideH = 1;
//  conv_attr->strideW = 1;
//  conv_attr->kernelH = 3;
//  conv_attr->kernelW = 3;
//  conv_attr->dilateH = 1;
//  conv_attr->dilateW = 1;
//
//  conv->attr.value = conv_attr;
//  conv->name = "Conv2D";
//  conv->fmkType = schema::FmkType_CAFFE;
//  msSubgraph->nodes.emplace_back(std::move(conv));
//
//  auto matMul1 = std::make_unique<schema::OpDefT>();
//  matMul1->inputIndex = {2, 3};
//  matMul1->outputIndex = {4};
//  matMul1->attr.type = schema::OpT_MatMul;
//  auto matMul_attr1 = new schema::MatMulT;
//  matMul_attr1->transposeA = false;
//  matMul_attr1->transposeB = true;
//  matMul1->attr.value = matMul_attr1;
//  matMul1->name = "matmul1";
//  matMul1->fmkType = schema::FmkType_CAFFE;
//  msSubgraph->nodes.emplace_back(std::move(matMul1));
//
//  auto matMul2 = std::make_unique<schema::OpDefT>();
//  matMul2->inputIndex = {4, 5};
//  matMul2->outputIndex = {6};
//  matMul2->attr.type = schema::OpT_MatMul;
//  auto matMul_attr2 = new schema::MatMulT;
//  matMul_attr2->transposeA = false;
//  matMul_attr2->transposeB = true;
//  matMul2->attr.value = matMul_attr2;
//  matMul2->name = "matmul2";
//  matMul2->fmkType = schema::FmkType_CAFFE;
//  msSubgraph->nodes.emplace_back(std::move(matMul2));
//
//  msSubgraph->inputIndex = {0};
//  msSubgraph->outputIndex = {6};
//
//  auto input0 = std::make_unique<schema::TensorDefT>();
//  input0->refCount = schema::MSCONST_WEIGHT_REFCOUNT;
//  input0->format = schema::Format_NHWC;
//  input0->dataType = TypeId::kNumberTypeFloat32;
//  input0->dims = {1, 5, 5, 3};
//  input0->offset = -1;
//  msSubgraph->allTensors.emplace_back(std::move(input0));
//
//  auto conv_weight = std::make_unique<schema::TensorDefT>();
//  conv_weight->refCount = schema::MSCONST_WEIGHT_REFCOUNT;
//  conv_weight->format = schema::Format_KHWC;
//  conv_weight->dataType = TypeId::kNumberTypeFloat32;
//  conv_weight->dims = {8, 3, 3, 3};
//  conv_weight->data.resize(8*3*3*3*sizeof(float));
//  msSubgraph->allTensors.emplace_back(std::move(conv_weight));
//
//  auto conv_output = std::make_unique<schema::TensorDefT>();
//  conv_output->refCount = 0;
//  conv_output->format = schema::Format_NHWC;
//  conv_output->dataType = TypeId::kNumberTypeFloat32;
//  conv_output->dims = {1, 5, 5, 8};
//  msSubgraph->allTensors.emplace_back(std::move(conv_output));
//
//  auto add_weight = std::make_unique<schema::TensorDefT>();
//  add_weight->refCount = schema::MSCONST_WEIGHT_REFCOUNT;
//  add_weight->format = schema::Format_NHWC;
//  add_weight->dataType = TypeId::kNumberTypeFloat32;
//  add_weight->dims = {1, 5, 5, 8};
//  add_weight->data.resize(5*5*8*sizeof(float));
//  msSubgraph->allTensors.emplace_back(std::move(add_weight));
//
//  auto add_output = std::make_unique<schema::TensorDefT>();
//  add_output->refCount = 0;
//  add_output->format = schema::Format_NHWC;
//  add_output->dataType = TypeId::kNumberTypeFloat32;
//  add_output->dims = {1, 5, 5, 8};
//  msSubgraph->allTensors.emplace_back(std::move(add_output));
//
//  auto mul_weight = std::make_unique<schema::TensorDefT>();
//  mul_weight->refCount = schema::MSCONST_WEIGHT_REFCOUNT;
//  mul_weight->format = schema::Format_NHWC;
//  mul_weight->dataType = TypeId::kNumberTypeFloat32;
//  mul_weight->dims = {1, 5, 5, 8};
//  mul_weight->data.resize(5*5*8*sizeof(float));
//  msSubgraph->allTensors.emplace_back(std::move(mul_weight));
//
//  auto mul_output = std::make_unique<schema::TensorDefT>();
//  mul_output->refCount = 0;
//  mul_output->format = schema::Format_NHWC;
//  mul_output->dataType = TypeId::kNumberTypeFloat32;
//  mul_output->dims = {1, 5, 5, 8};
//  msSubgraph->allTensors.emplace_back(std::move(mul_output));
//  msGraph->subgraphs.emplace_back(std::move(msSubgraph));
//
//  flatbuffers::FlatBufferBuilder builder(1024);
//  auto offset = schema::GraphDef::Pack(builder, msGraph.get());
//  builder.Finish(offset);
//  size_t size = builder.GetSize();
//  const char *content = (char *)builder.GetBufferPointer();
//  const std::string strstub = "";
//
//  auto func_graph = inference::LoadModel(content, size, strstub);
//  ASSERT_NE(nullptr, func_graph);
//  auto session = inference::MSSession::CreateSession(kCPUDevice, 0);
//  ASSERT_NE(nullptr, session);
//  auto graphId = session->CompileGraph(func_graph);
//
//  auto inTensor =
//    std::shared_ptr<inference::MSTensor>(inference::MSTensor::CreateTensor(TypeId::kNumberTypeFloat32, {1, 5, 5, 3}));
//  ASSERT_NE(nullptr, inTensor);
//  ASSERT_EQ(sizeof(float) * (5 * 5 * 3), inTensor->Size());
//  (void)inTensor->MutableData();
//
//  std::vector<std::shared_ptr<inference::MSTensor>> inputs;
//  inputs.emplace_back(inTensor);
//  auto outputs = session->RunGraph(graphId, inputs);
//  ASSERT_EQ(1, outputs.size());
//  ASSERT_EQ(1, outputs.front().size());
//  auto runOutput = outputs.front().front();
//  ASSERT_NE(nullptr, runOutput);
//  ASSERT_EQ(5 * 5 * 8, runOutput->ElementsNum());
//  ASSERT_EQ(TypeId::kNumberTypeFloat32, runOutput->data_type());
//  MS_LOG(INFO) << "Passed";
//}
}  // namespace mindspore
