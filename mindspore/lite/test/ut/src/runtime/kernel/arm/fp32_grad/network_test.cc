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

#include "mindspore/lite/schema/inner/model_generated.h"
#include "mindspore/lite/include/model.h"
#include "common/common_test.h"
#include "include/train_session.h"
// #include "include/lite_session.h"
#include "include/context.h"
#include "include/errorcode.h"
#include "utils/log_adapter.h"
#include "src/common/file_utils.h"
#include "src/common/file_utils_ext.h"

namespace mindspore {
class NetworkTest : public mindspore::CommonTest {
 public:
  NetworkTest() {}
};

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
#if 0
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
    primitive->type = schema::ActivationType_RELU;
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
    primitive->transposeA = false;
    primitive->transposeB = true;
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
    primitive->axis.push_back(0);
    node->primitive->value.value = primitive;
    node->name = "BiasAdd";
    meta_graph->nodes.emplace_back(std::move(node));
  }
  {
    auto node = std::make_unique<schema::CNodeT>();
    node->inputIndex = {5, 6};
    node->outputIndex = {14, 7};
    node->primitive = std::make_unique<schema::PrimitiveT>();
    node->primitive->value.type = schema::PrimitiveType_SoftmaxCrossEntropy;
    auto primitive = new schema::SoftmaxCrossEntropyT;
    primitive->axis.push_back(0);
    node->primitive->value.value = primitive;
    node->name = "SoftmaxCrossEntropy";
    meta_graph->nodes.emplace_back(std::move(node));
  }
  {
    auto node = std::make_unique<schema::CNodeT>();
    node->inputIndex = {7};
    node->outputIndex = {8};
    node->primitive = std::make_unique<schema::PrimitiveT>();
    node->primitive->value.type = schema::PrimitiveType_BiasGrad;
    auto primitive = new schema::BiasGradT;
    primitive->axis.push_back(0);
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
    primitive->transposeA = true;
    primitive->transposeB = false;
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
    node->primitive->value.value = primitive;
    node->name = "Momentum";
    meta_graph->nodes.emplace_back(std::move(node));
  }
  meta_graph->inputIndex = {0, 6};
  meta_graph->outputIndex = {5, 14};

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = schema::NodeType::NodeType_ValueNode;
  input0->format = schema::Format_NHWC;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {BATCH_SIZE, FEATURE_SIZE};
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));
  // tensor 1 - relu
  auto relu_out = std::make_unique<schema::TensorT>();
  relu_out->nodeType = schema::NodeType::NodeType_Parameter;
  relu_out->format = schema::Format_NHWC;
  relu_out->dataType = TypeId::kNumberTypeFloat32;
  relu_out->dims = {BATCH_SIZE, FEATURE_SIZE};
  relu_out->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(relu_out));
  // tensor 2 - matmul weights
  auto weight = std::make_unique<schema::TensorT>();
  weight->nodeType = schema::NodeType::NodeType_ValueNode;
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
  delete [] buf;
  // tensor 3 - matmul
  auto input3 = std::make_unique<schema::TensorT>();
  input3->nodeType = schema::NodeType::NodeType_Parameter;
  input3->format = schema::Format_NHWC;
  input3->dataType = TypeId::kNumberTypeFloat32;
  input3->dims = {BATCH_SIZE, NUM_CLASSES};
  input3->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input3));
  // tensor 4 - fc bias
  auto bias = std::make_unique<schema::TensorT>();
  bias->nodeType = schema::NodeType::NodeType_ValueNode;
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
  delete [] buf;

  // tensor 5 - bias_add
  auto input5 = std::make_unique<schema::TensorT>();
  input5->nodeType = schema::NodeType::NodeType_Parameter;
  input5->format = schema::Format_NHWC;
  input5->dataType = TypeId::kNumberTypeFloat32;
  input5->dims = {BATCH_SIZE, NUM_CLASSES};
  input5->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input5));
  // tensor 6 - Label
  {
    auto label = std::make_unique<schema::TensorT>();
    label->nodeType = schema::NodeType::NodeType_ValueNode;
    label->format = schema::Format_NHWC;
    label->dataType = TypeId::kNumberTypeInt32;
    label->dims = {BATCH_SIZE};
    label->offset = -1;
    label->data.resize(BATCH_SIZE * NUM_CLASSES * sizeof(float));
    int *data = reinterpret_cast<int *>(label->data.data());
    for (int i = 0; i < BATCH_SIZE; i++)
      for (int j = 0; j < NUM_CLASSES; j++) *(data + i * NUM_CLASSES + j) = j;
    meta_graph->allTensors.emplace_back(std::move(label));
  }
  // tensor 7 - Softmaxentropy
  auto input7 = std::make_unique<schema::TensorT>();
  input7->nodeType = schema::NodeType::NodeType_Parameter;
  input7->format = schema::Format_NHWC;
  input7->dataType = TypeId::kNumberTypeFloat32;
  input7->dims = {BATCH_SIZE, NUM_CLASSES};
  input7->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input7));
  // tensor 8 - biasGrad
  auto input8 = std::make_unique<schema::TensorT>();
  input8->nodeType = schema::NodeType::NodeType_Parameter;
  input8->format = schema::Format_NHWC;
  input8->dataType = TypeId::kNumberTypeFloat32;
  input8->dims = {NUM_CLASSES};
  input8->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input8));
  // tensor 9 - matmul2
  auto input9 = std::make_unique<schema::TensorT>();
  input9->nodeType = schema::NodeType::NodeType_Parameter;
  input9->format = schema::Format_NHWC;
  input9->dataType = TypeId::kNumberTypeFloat32;
  input9->dims = {NUM_CLASSES, FEATURE_SIZE};
  input9->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input9));
  // tensor 10 weights accumulate
  auto input10 = std::make_unique<schema::TensorT>();
  input10->nodeType = schema::NodeType::NodeType_ValueNode;
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
    lr->nodeType = schema::NodeType::NodeType_ValueNode;
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
    input12->nodeType = schema::NodeType::NodeType_ValueNode;
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
  input13->nodeType = schema::NodeType::NodeType_ValueNode;
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
    loss14->nodeType = schema::NodeType::NodeType_ValueNode;
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
  size_t size = builder.GetSize();
  const char *content = reinterpret_cast<char *>(builder.GetBufferPointer());
  std::cout << "build fb size= " << size << "\n";

#if 0  // EXPORT_FILE
  std::string path = std::string("hcdemo_train.fb");
  std::ofstream ofs(path);
  ASSERT_EQ(true, ofs.good());
  ASSERT_EQ(true, ofs.is_open());

  ofs.seekp(0, std::ios::beg);
  ofs.write(content, size);
  ofs.close();
#endif

  auto model = lite::Model::Import(content, size);
  ASSERT_NE(nullptr, model);
  meta_graph.reset();
  content = nullptr;
  lite::Context context;
  context.device_type_ = lite::DT_CPU;
  context.cpu_bind_mode_ = lite::NO_BIND;
  context.thread_num_ = 1;
  auto session = new session::TrainSession();
  ASSERT_NE(nullptr, session);
  session->Init(&context);
  auto ret = session->CompileGraph(model);
  ASSERT_EQ(lite::RET_OK, ret);
  session->train();

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
  delete [] buf;
  auto labelTensor = inputs.at(1);
  ASSERT_NE(nullptr, labelTensor);
  ASSERT_EQ(BATCH_SIZE, labelTensor->ElementsNum());
  auto labels = reinterpret_cast<int *>(labelTensor->MutableData());
  for (int i = 0; i < BATCH_SIZE; i++) labels[i] = (i * 97) % NUM_CLASSES;

  ret = session->RunGraph();
  ASSERT_EQ(lite::RET_OK, ret);
  auto outputs = session->GetOutputsByName("BiasAdd");
  ASSERT_EQ(outputs.size(), 1);
  auto outTensor = (outputs.at(0));
  ASSERT_NE(nullptr, outTensor);
  ASSERT_EQ(TypeId::kNumberTypeFloat32, outTensor->data_type());
  auto *outData = reinterpret_cast<float *>(outTensor->MutableData());
  ASSERT_NE(nullptr, outData);
  std::cout << "==============Initial=Scores===================" << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << outData[i] << ", ";
  }
  std::cout << std::endl;
  ret = session->RunGraph();
  outputs = session->GetOutputsByName("BiasAdd");
  ASSERT_EQ(outputs.size(), 1);
  outTensor = (outputs.at(0));
  ASSERT_NE(nullptr, outTensor);
  // ASSERT_EQ(28 * 28 * 32, outTensor->ElementsNum());
  ASSERT_EQ(TypeId::kNumberTypeFloat32, outTensor->data_type());
  outData = reinterpret_cast<float *>(outTensor->MutableData());
  ASSERT_NE(nullptr, outData);
  std::cout << "==============Scores=after-single=train========" << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << outData[i] << ", ";
  }
  std::string output_path = "./test_data/train/train_output_32_10.bin";
  auto error = lite::RelativeOutputError(outData, output_path);
  EXPECT_LT(error, 2e-3);
  MS_LOG(INFO) << "TuningLayer passed";

  delete model;
  delete session;
}
#endif
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

int32_t runEffNet(mindspore::lite::LiteSession *session, const std::string &in, const std::string &out) {
  // setup input
  auto inputs = session->GetInputs();
  // ASSERT_EQ(inputs.size(), 1);
  auto inTensor = inputs.at(0);
  // ASSERT_NE(nullptr, inTensor);
  float *data = reinterpret_cast<float *>(inTensor->MutableData());

  size_t input_size;
  float *in_buf = reinterpret_cast<float *>(lite::ReadFile(in.c_str(), &input_size));
  // ASSERT_NE(nullptr, data);
  auto input_data = reinterpret_cast<float *>(in_buf);
  // ASSERT_EQ(input_size, inTensor->Size());
  std::copy(input_data, input_data + inTensor->ElementsNum(), data);
  delete [] in_buf;

  // execute network
  session->RunGraph();

  // compare outputs
  auto outputs = session->GetOutputs();
  auto output = ((outputs.begin())->second);
  float *output_data = reinterpret_cast<float *>(output->MutableData());

  return mindspore::lite::CompareRelativeOutput(output_data, out.c_str());
}

TEST_F(NetworkTest, efficient_net) {
  char *buf = nullptr;
  size_t net_size = 0;
  // std::string net = "./test_data/nets/efficientnet_b0_f.ms";

  std::string net = "./test_data/nets/effnetb0_fwd_nofuse.ms";
  ReadFile(net.c_str(), &net_size, &buf);
  auto model = lite::Model::Import(buf, net_size);
  delete [] buf;
  auto context = new lite::Context;
  context->device_type_ = lite::DT_CPU;
  context->cpu_bind_mode_ = lite::NO_BIND;
  context->thread_num_ = 1;

  auto session = new mindspore::session::TrainSession();
  // auto session = new mindspore::lite::LiteSession();
  ASSERT_NE(session, nullptr);
  auto ret = session->Init(context);
  ASSERT_EQ(lite::RET_OK, ret);
  ret = session->CompileGraph(model);
  ASSERT_EQ(lite::RET_OK, ret);
  session->eval();

#if 0
  std::string path = "/opt/share/MiniBinEmbDataset/";
  auto res = fileIterator(session, path, [](mindspore::lite::LiteSession *session, const std::string &in) {
    int32_t res = 0;
    if (in.find(".bin") != std::string::npos) {
      std::string out;
      replaceExt(in, &out);
      res = runEffNet(session, in, out);
      std::cout << "input file: " << in << (res ?  " Fail" : " Pass") << std::endl;
    }
    return res;
  });
#else
  std::string in = "./test_data/nets/effNet_input_x_1_3_224_224.bin";
  std::string out = "./test_data/nets/effNet_output_y_1_1000.bin";
  auto res = runEffNet(session, in, out);
#endif
  // auto inputs = session->GetInputs();
  // ASSERT_EQ(inputs.size(), NUM_OF_INPUTS);
  // auto inTensor = inputs.at(0);
  // ASSERT_NE(nullptr, inTensor);
  // float *data = reinterpret_cast<float *>(inTensor->MutableData());

  // // fill input
  // std::string input_path = "./test_data/nets/effNet_input_x_1_3_224_224.bin";
  // // std::string input_path = "/opt/share/MiniBinEmbDataset/2_pet/n02099601_3111.bin";
  // size_t input_size;
  // char *in_buf = nullptr;
  // ReadFile(input_path.c_str(), &input_size, &in_buf);
  // ASSERT_NE(nullptr, data);
  // auto input_data = reinterpret_cast<float *>(in_buf);
  // ASSERT_EQ(input_size, inTensor->Size());
  // std::copy(input_data, input_data+inTensor->ElementsNum(), data);

  // // execute network
  // ret = session->RunGraph();

  // // compare outputs
  // std::string output_path = "./test_data/nets/effNet_output_y_1_1000.bin";
  // // std::string output_path = "/opt/share/MiniBinEmbDataset/2_pet/n02099601_3111.emb";
  // auto outputs = session->GetOutputs();
  // auto output = ((outputs.begin())->second);
  // float* output_data = reinterpret_cast<float *>(output.at(0)->MutableData());
  // int res = lite::CompareRelativeOutput(output_data, output_path);
  ASSERT_EQ(res, 0);
  delete model;
  delete session;
  delete context;
}

}  // namespace mindspore
