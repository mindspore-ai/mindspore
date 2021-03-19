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
#include <dirent.h>
#include <climits>
#include <cmath>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <functional>

#include "schema/inner/model_generated.h"
#include "common/common_test.h"
#include "include/train/train_session.h"
#include "include/context.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/common/file_utils.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/fp32_grad/convolution.h"

using mindspore::lite::RET_OK;
namespace mindspore {
class NetworkTest : public mindspore::CommonTest {
 public:
  NetworkTest() {}
};

int32_t runNet(mindspore::session::LiteSession *session, const std::string &in, const std::string &out,
               const char *tensor_name, bool debug = false);

//             INPUT(0)
//                V
//        +-------------+
//        |     ReLU    |
//        +-------------+
//  +---output(1) V
//  |             V   V weights(2) <----+
//  |     +-------------+               |
//  |     |    MatMul   |               |
//  |     +-------------+               |
//  |   output(3) V                     |
//  |             V   V weights(4)<-+   |
//  |     +-------------+           |   |
//  |     |    Bias     |           |   |
//  |     +-------------+           |   |
//  |   output(5) V                 |   |
//  |             V   V LABELS(6)   |   |
//  |     +-------------+           |   |
//  |     | CrossEntropy|           |   |
//  |     +-------------+           |   |
//  |  +-dy(7) V  V------------------------->Loss (14)
//  |  |       V                    |   |
//  |  |  +-------------+           |   |
//  |  |  |  BiasGrad   |           |   |
//  |  |  +-------------+           |   |
//  |  |          V db(8)           |   |
//  |  |          +--------Update---+   |
//  |  +-------+                        |
//  +------V   V                        |
//        +-------------+               |
//        |  MatMul     |               |
//        +-------------+               |
//               V dw(9)                |
//               +-----------Update-----+
TEST_F(NetworkTest, tuning_layer) {
  const int BATCH_SIZE = 32;
  const int NUM_CLASSES = 10;
  const int FEATURE_SIZE = 1000;
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  // define nodes
  {
    auto node = std::make_unique<schema::CNodeT>();
    node->inputIndex = {0};
    node->outputIndex = {1};
    node->primitive = std::make_unique<schema::PrimitiveT>();
    node->primitive->value.type = schema::PrimitiveType_Activation;
    auto primitive = new schema::ActivationT;
    ASSERT_NE(primitive, nullptr);
    primitive->activation_type = schema::ActivationType_RELU;
    node->primitive->value.value = primitive;
    node->name = "ReLU";
    meta_graph->nodes.emplace_back(std::move(node));
  }
  {
    auto node = std::make_unique<schema::CNodeT>();
    node->inputIndex = {1, 2};
    node->outputIndex = {3};
    node->primitive = std::make_unique<schema::PrimitiveT>();
    node->primitive->value.type = schema::PrimitiveType_MatMul;
    auto primitive = new schema::MatMulT;
    ASSERT_NE(primitive, nullptr);
    primitive->transpose_a = false;
    primitive->transpose_b = true;
    node->primitive->value.value = primitive;
    node->name = "MatMul1";
    meta_graph->nodes.emplace_back(std::move(node));
  }
  {
    auto node = std::make_unique<schema::CNodeT>();
    node->inputIndex = {3, 4};
    node->outputIndex = {5};
    node->primitive = std::make_unique<schema::PrimitiveT>();
    node->primitive->value.type = schema::PrimitiveType_BiasAdd;
    auto primitive = new schema::BiasAddT;
    ASSERT_NE(primitive, nullptr);
    node->primitive->value.value = primitive;
    node->name = "BiasAdd";
    meta_graph->nodes.emplace_back(std::move(node));
  }
  {
    auto node = std::make_unique<schema::CNodeT>();
    node->inputIndex = {5, 6};
    node->outputIndex = {14, 7};
    node->primitive = std::make_unique<schema::PrimitiveT>();
    node->primitive->value.type = schema::PrimitiveType_SoftmaxCrossEntropyWithLogits;
    auto primitive = new schema::SoftmaxCrossEntropyWithLogitsT;
    ASSERT_NE(primitive, nullptr);
    node->primitive->value.value = primitive;
    node->name = "SoftmaxCrossEntropyWithLogits";
    meta_graph->nodes.emplace_back(std::move(node));
  }
  {
    auto node = std::make_unique<schema::CNodeT>();
    node->inputIndex = {7};
    node->outputIndex = {8};
    node->primitive = std::make_unique<schema::PrimitiveT>();
    node->primitive->value.type = schema::PrimitiveType_BiasAddGrad;
    auto primitive = new schema::BiasAddGradT;
    ASSERT_NE(primitive, nullptr);
    node->primitive->value.value = primitive;
    node->name = "BiasGrad";
    meta_graph->nodes.emplace_back(std::move(node));
  }
  {
    auto node = std::make_unique<schema::CNodeT>();
    node->inputIndex = {7, 1};
    node->outputIndex = {9};
    node->primitive = std::make_unique<schema::PrimitiveT>();
    node->primitive->value.type = schema::PrimitiveType_MatMul;
    auto primitive = new schema::MatMulT;
    ASSERT_NE(primitive, nullptr);
    primitive->transpose_a = true;
    primitive->transpose_b = false;
    node->primitive->value.value = primitive;
    node->name = "MatMul2";
    meta_graph->nodes.emplace_back(std::move(node));
  }
  {
    auto node = std::make_unique<schema::CNodeT>();
    node->inputIndex = {2, 10, 11, 9, 12};
    node->outputIndex = {};
    node->primitive = std::make_unique<schema::PrimitiveT>();
    node->primitive->value.type = schema::PrimitiveType_ApplyMomentum;
    auto primitive = new schema::ApplyMomentumT;
    ASSERT_NE(primitive, nullptr);
    node->primitive->value.value = primitive;
    node->name = "Momentum";
    meta_graph->nodes.emplace_back(std::move(node));
  }
  {
    auto node = std::make_unique<schema::CNodeT>();
    node->inputIndex = {4, 13, 11, 8, 12};
    node->outputIndex = {};
    node->primitive = std::make_unique<schema::PrimitiveT>();
    node->primitive->value.type = schema::PrimitiveType_ApplyMomentum;
    auto primitive = new schema::ApplyMomentumT;
    ASSERT_NE(primitive, nullptr);
    node->primitive->value.value = primitive;
    node->name = "Momentum";
    meta_graph->nodes.emplace_back(std::move(node));
  }
  meta_graph->inputIndex = {0, 6};
  meta_graph->outputIndex = {5, 14};

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_ValueNode;
  input0->format = schema::Format_NHWC;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {BATCH_SIZE, FEATURE_SIZE};
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));
  // tensor 1 - relu
  auto relu_out = std::make_unique<schema::TensorT>();
  relu_out->nodeType = lite::NodeType_Parameter;
  relu_out->format = schema::Format_NHWC;
  relu_out->dataType = TypeId::kNumberTypeFloat32;
  relu_out->dims = {BATCH_SIZE, FEATURE_SIZE};
  relu_out->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(relu_out));
  // tensor 2 - matmul weights
  auto weight = std::make_unique<schema::TensorT>();
  weight->nodeType = lite::NodeType_ValueNode;
  weight->format = schema::Format_KHWC;
  weight->dataType = TypeId::kNumberTypeFloat32;
  weight->dims = {NUM_CLASSES, FEATURE_SIZE};
  size_t weight_size;
  char *buf;
  std::string weight_path = "./test_data/train/train_weight_10_1000.bin";
  ReadFile(weight_path.c_str(), &weight_size, &buf);
  ASSERT_NE(nullptr, buf);
  weight->data.resize(weight_size);
  std::copy(buf, buf + weight_size, weight->data.data());
  meta_graph->allTensors.emplace_back(std::move(weight));
  delete[] buf;
  // tensor 3 - matmul
  auto input3 = std::make_unique<schema::TensorT>();
  input3->nodeType = lite::NodeType_Parameter;
  input3->format = schema::Format_NHWC;
  input3->dataType = TypeId::kNumberTypeFloat32;
  input3->dims = {BATCH_SIZE, NUM_CLASSES};
  input3->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input3));
  // tensor 4 - fc bias
  auto bias = std::make_unique<schema::TensorT>();
  bias->nodeType = lite::NodeType_ValueNode;
  bias->format = schema::Format_NHWC;
  bias->dataType = TypeId::kNumberTypeFloat32;
  bias->dims = {NUM_CLASSES};
  bias->offset = -1;
  std::string bias_path = "./test_data/train/train_bias_10.bin";
  size_t bias_size;
  ReadFile(bias_path.c_str(), &bias_size, &buf);
  ASSERT_NE(nullptr, buf);
  bias->data.resize(bias_size);
  std::copy(buf, buf + bias_size, bias->data.data());
  meta_graph->allTensors.emplace_back(std::move(bias));
  delete[] buf;

  // tensor 5 - bias_add
  auto input5 = std::make_unique<schema::TensorT>();
  input5->nodeType = lite::NodeType_Parameter;
  input5->format = schema::Format_NHWC;
  input5->dataType = TypeId::kNumberTypeFloat32;
  input5->dims = {BATCH_SIZE, NUM_CLASSES};
  input5->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input5));
  // tensor 6 - Label
  {
    auto label = std::make_unique<schema::TensorT>();
    label->nodeType = lite::NodeType_ValueNode;
    label->format = schema::Format_NHWC;
    label->dataType = TypeId::kNumberTypeFloat32;
    label->dims = {BATCH_SIZE * NUM_CLASSES};
    label->offset = -1;
    meta_graph->allTensors.emplace_back(std::move(label));
  }
  // tensor 7 - Softmaxentropy
  auto input7 = std::make_unique<schema::TensorT>();
  input7->nodeType = lite::NodeType_Parameter;
  input7->format = schema::Format_NHWC;
  input7->dataType = TypeId::kNumberTypeFloat32;
  input7->dims = {BATCH_SIZE, NUM_CLASSES};
  input7->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input7));
  // tensor 8 - biasGrad
  auto input8 = std::make_unique<schema::TensorT>();
  input8->nodeType = lite::NodeType_Parameter;
  input8->format = schema::Format_NHWC;
  input8->dataType = TypeId::kNumberTypeFloat32;
  input8->dims = {NUM_CLASSES};
  input8->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input8));
  // tensor 9 - matmul2
  auto input9 = std::make_unique<schema::TensorT>();
  input9->nodeType = lite::NodeType_Parameter;
  input9->format = schema::Format_NHWC;
  input9->dataType = TypeId::kNumberTypeFloat32;
  input9->dims = {NUM_CLASSES, FEATURE_SIZE};
  input9->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input9));
  // tensor 10 weights accumulate
  auto input10 = std::make_unique<schema::TensorT>();
  input10->nodeType = lite::NodeType_ValueNode;
  input10->format = schema::Format_NHWC;
  input10->dataType = TypeId::kNumberTypeFloat32;
  input10->dims = {NUM_CLASSES, FEATURE_SIZE};
  input10->offset = -1;
  size_t input10_size = NUM_CLASSES * FEATURE_SIZE * sizeof(float);
  input10->data.resize(input10_size);
  std::fill(input10->data.data(), input10->data.data() + input10_size, 0.f);
  meta_graph->allTensors.emplace_back(std::move(input10));
  // tensor 11 - lr
  {
    auto lr = std::make_unique<schema::TensorT>();
    lr->nodeType = lite::NodeType_ValueNode;
    lr->format = schema::Format_NHWC;
    lr->dataType = TypeId::kNumberTypeFloat32;
    lr->dims = {1};
    lr->offset = -1;
    lr->data.resize(sizeof(float));
    float *data = reinterpret_cast<float *>(lr->data.data());
    *data = 0.01f;
    meta_graph->allTensors.emplace_back(std::move(lr));
  }
  // tensor 12  - momentum
  {
    auto input12 = std::make_unique<schema::TensorT>();
    input12->nodeType = lite::NodeType_ValueNode;
    input12->format = schema::Format_NHWC;
    input12->dataType = TypeId::kNumberTypeFloat32;
    input12->dims = {1};
    input12->offset = -1;
    input12->data.resize(sizeof(float));
    float *data = reinterpret_cast<float *>(input12->data.data());
    *data = 0.f;
    meta_graph->allTensors.emplace_back(std::move(input12));
  }
  // tensor 13 - bias accumulate
  auto input13 = std::make_unique<schema::TensorT>();
  input13->nodeType = lite::NodeType_ValueNode;
  input13->format = schema::Format_NHWC;
  input13->dataType = TypeId::kNumberTypeFloat32;
  input13->dims = {NUM_CLASSES};
  input13->offset = -1;
  size_t input13_size = NUM_CLASSES * sizeof(float);
  input13->data.resize(input13_size);
  std::fill(input13->data.data(), input13->data.data() + input13_size, 0.f);
  meta_graph->allTensors.emplace_back(std::move(input13));

  // tensor 14 - loss
  {
    auto loss14 = std::make_unique<schema::TensorT>();
    loss14->nodeType = lite::NodeType_ValueNode;
    loss14->format = schema::Format_NHWC;
    loss14->dataType = TypeId::kNumberTypeFloat32;
    loss14->dims = {1};
    loss14->offset = -1;
    loss14->data.resize(sizeof(float));
    float *data = reinterpret_cast<float *>(loss14->data.data());
    *data = 0.0f;
    meta_graph->allTensors.emplace_back(std::move(loss14));
  }

  //================================================================
  buf = nullptr;

  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = schema::MetaGraph::Pack(builder, meta_graph.get());
  builder.Finish(offset);
  schema::FinishMetaGraphBuffer(builder, offset);
  size_t size = builder.GetSize();
  const char *content = reinterpret_cast<char *>(builder.GetBufferPointer());
  std::cout << "build fb size= " << size << std::endl;

  meta_graph.reset();
  content = nullptr;
  lite::Context context;
  context.device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = lite::NO_BIND;
  context.thread_num_ = 1;
  auto session = session::TrainSession::CreateSession(content, size, &context);
  ASSERT_NE(nullptr, session);
  session->Train();
  session->Train();  // Just double check that calling Train twice does not cause a problem

  auto inputs = session->GetInputs();
  ASSERT_EQ(inputs.size(), 2);
  auto inTensor = inputs.at(0);
  ASSERT_NE(nullptr, inTensor);
  auto data = inTensor->MutableData();
  //===================================================
  size_t input_size;
  std::string input_path = "./test_data/train/train_input_32_1000.bin";
  ReadFile(input_path.c_str(), &input_size, &buf);
  ASSERT_NE(nullptr, buf);
  auto input_data = reinterpret_cast<float *>(buf);
  ASSERT_NE(nullptr, input_data);
  //===================================================
  ASSERT_EQ(input_size, inTensor->Size());
  memcpy(data, input_data, input_size);
  delete[] buf;
  auto labelTensor = inputs.at(1);
  ASSERT_NE(nullptr, labelTensor);
  ASSERT_EQ(BATCH_SIZE * NUM_CLASSES, labelTensor->ElementsNum());

  auto labels = reinterpret_cast<float *>(labelTensor->MutableData());
  std::fill(labels, labels + labelTensor->ElementsNum(), 0.f);
  for (int i = 0; i < BATCH_SIZE; i++) labels[i * NUM_CLASSES + (i * 97) % NUM_CLASSES] = 1.0;

  auto ret = session->RunGraph();
  ASSERT_EQ(lite::RET_OK, ret);
  auto outputs = session->GetOutputsByNodeName("SoftmaxCrossEntropyWithLogits");
  ASSERT_EQ(outputs.size(), 1);
  auto outTensor = (outputs.at(0));
  ASSERT_NE(nullptr, outTensor);
  ASSERT_EQ(TypeId::kNumberTypeFloat32, outTensor->data_type());
  auto *outData = reinterpret_cast<float *>(outTensor->MutableData());
  ASSERT_NE(nullptr, outData);
  std::cout << "==============Initial=Loss=====================" << std::endl;
  std::cout << outData[0] << ", " << std::endl;

  session->Eval();
  session->Eval();  // Just double check that calling eval twice does not cause a problem
  ret = session->RunGraph();
  outputs = session->GetOutputsByNodeName("BiasAdd");
  ASSERT_EQ(outputs.size(), 1);
  outTensor = (outputs.at(0));
  ASSERT_NE(nullptr, outTensor);
  ASSERT_EQ(TypeId::kNumberTypeFloat32, outTensor->data_type());
  outData = reinterpret_cast<float *>(outTensor->MutableData());
  ASSERT_NE(nullptr, outData);
  std::cout << "==============Scores=after-single=train========" << std::endl;
  for (int i = 0; i < 10; i++) {
    std::cout << outData[i] << ", ";
  }
  std::cout << std::endl;
  std::string output_path = "./test_data/train/train_output_32_10.bin";
  auto error = RelativeOutputError(outData, output_path);
  EXPECT_LT(error, 2e-3);

  ret = session->RunGraph();
  auto all_output_tensors = session->GetOutputs();
  outTensor = (all_output_tensors["5"]);
  ASSERT_NE(nullptr, outTensor);
  ASSERT_EQ(TypeId::kNumberTypeFloat32, outTensor->data_type());
  outData = reinterpret_cast<float *>(outTensor->MutableData());
  ASSERT_NE(nullptr, outData);
  std::cout << "==============Scores=eval-second-time==========" << std::endl;
  for (int i = 0; i < 10; i++) {
    std::cout << outData[i] << ", ";
  }
  std::cout << std::endl;
  error = RelativeOutputError(outData, output_path);
  EXPECT_LT(error, 2e-3);

  session->Train();
  session->Eval();  // do some more zig-zags
  ret = session->RunGraph();
  outTensor = session->GetOutputByTensorName("5");
  ASSERT_NE(nullptr, outTensor);
  ASSERT_EQ(TypeId::kNumberTypeFloat32, outTensor->data_type());
  outData = reinterpret_cast<float *>(outTensor->MutableData());
  ASSERT_NE(nullptr, outData);
  std::cout << "==============Scores=Just Checking 3rd time====" << std::endl;
  for (int i = 0; i < 10; i++) {
    std::cout << outData[i] << ", ";
  }
  std::cout << std::endl;
  error = RelativeOutputError(outData, output_path);
  EXPECT_LT(error, 2e-3);
}

int32_t fileIterator(mindspore::session::TrainSession *session, const std::string &path,
                     std::function<int32_t(mindspore::session::TrainSession *session, const std::string &)> cb) {
  int32_t res = 0;
  if (auto dir = opendir(path.c_str())) {
    while (auto f = readdir(dir)) {
      if (f->d_name[0] == '.') continue;
      if (f->d_type == DT_DIR) fileIterator(session, path + f->d_name + "/", cb);

      if (f->d_type == DT_REG) res |= cb(session, path + f->d_name);
    }
    closedir(dir);
  }
  return res;
}
void replaceExt(const std::string &src, std::string *dst) { *dst = src.substr(0, src.find_last_of('.')) + ".emb"; }

int32_t runNet(mindspore::session::LiteSession *session, const std::string &in, const std::string &out,
               const char *tensor_name, bool debug) {
  // setup input
  auto inputs = session->GetInputs();
  auto inTensor = inputs.at(0);
  float *data = reinterpret_cast<float *>(inTensor->MutableData());
  size_t input_size;
  float *in_buf = reinterpret_cast<float *>(lite::ReadFile(in.c_str(), &input_size));
  auto input_data = reinterpret_cast<float *>(in_buf);
  std::copy(input_data, input_data + inTensor->ElementsNum(), data);
  std::cout << "==============Input===========================" << std::endl;
  for (int i = 0; i < 10; i++) {
    std::cout << data[i] << ", ";
  }
  std::cout << std::endl;
  delete[] in_buf;

  // execute network
  session->RunGraph();
  auto output = session->GetOutputByTensorName(tensor_name);
  if (output != nullptr) {
    float *output_data = reinterpret_cast<float *>(output->MutableData());
    // compare outputs
    if (debug) {
      std::cout << "==============Output===========================" << std::endl;
      for (int i = 0; i < 10; i++) {
        std::cout << output_data[i] << ", ";
      }
      std::cout << std::endl;
    }
    return CommonTest::CompareRelativeOutput(output_data, out);
  }

  return lite::RET_ERROR;
}

TEST_F(NetworkTest, efficient_net) {
  auto context = new lite::Context;
  ASSERT_NE(context, nullptr);
  context->device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = lite::NO_BIND;
  context->thread_num_ = 1;

  std::string net = "./test_data/nets/effnetb0_fwd_nofuse.ms";
  auto session = session::TrainSession::CreateSession(net, context, false);
  ASSERT_NE(session, nullptr);

  std::string in = "./test_data/nets/effNet_input_x_1_3_224_224.bin";
  std::string out = "./test_data/nets/effNet_output_y_1_1000.bin";
  auto res = runNet(session, in, out, "650");
  delete session;
  delete context;
  ASSERT_EQ(res, 0);
}

TEST_F(NetworkTest, mobileface_net) {
  char *buf = nullptr;
  size_t net_size = 0;

  std::string net = "./test_data/nets/mobilefacenet0924.ms";
  ReadFile(net.c_str(), &net_size, &buf);
  // auto model = lite::TrainModel::Import(buf, net_size);
  auto model = lite::Model::Import(buf, net_size);
  delete[] buf;
  auto context = new lite::Context;
  ASSERT_NE(context, nullptr);
  context->device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = lite::NO_BIND;
  context->thread_num_ = 1;

  // auto session = session::TrainSession::CreateSession(context);
  auto session = session::LiteSession::CreateSession(context);
  ASSERT_NE(session, nullptr);
  auto ret = session->CompileGraph(model);
  ASSERT_EQ(lite::RET_OK, ret);
  // session->Eval();

  std::string in = "./test_data/nets/facenet_input.f32";
  std::string out = "./test_data/nets/facenet_output.f32";
  auto res = runNet(session, in, out, "354", true);

  ASSERT_EQ(res, 0);
  delete model;
  delete session;
  delete context;
}

TEST_F(NetworkTest, setname) {
  std::string net = "./test_data/nets/lenet_train.ms";
  lite::Context context;
  context.device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = lite::NO_BIND;
  context.thread_num_ = 1;
  auto session = mindspore::session::TrainSession::CreateSession(net, &context);
  ASSERT_NE(session, nullptr);

  auto tensors_map = session->GetOutputs();
  auto tensor_names = session->GetOutputTensorNames();
  EXPECT_EQ(tensors_map.size(), 1);
  EXPECT_EQ(tensors_map.begin()->first, "24");
  EXPECT_EQ(tensor_names.size(), 1);
  EXPECT_EQ(tensor_names.at(0), "Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense/BiasAdd-op107");

  auto res = session->SetLossName("nhwc");
  EXPECT_EQ(res, RET_OK);
  tensors_map = session->GetOutputs();
  tensor_names = session->GetOutputTensorNames();
  EXPECT_EQ(tensors_map.begin()->first, "8");
  EXPECT_EQ(tensor_names.at(0), "Default/network-WithLossCell/_backbone-LeNet5/max_pool2d-MaxPool2d/MaxPool-op88");

  res = session->SetLossName("loss");
  EXPECT_EQ(res, RET_OK);
  tensors_map = session->GetOutputs();
  tensor_names = session->GetOutputTensorNames();
  EXPECT_EQ(tensors_map.begin()->first, "24");
  EXPECT_EQ(tensor_names.at(0), "Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense/BiasAdd-op107");
  delete session;
}

}  // namespace mindspore
