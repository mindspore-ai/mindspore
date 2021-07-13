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
#include "include/lite_session.h"
#include "include/context.h"
#include "include/errorcode.h"
#include "include/train/train_cfg.h"
#include "include/train/train_session.h"
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

int32_t fileIterator(mindspore::session::LiteSession *session, const std::string &path,
                     std::function<int32_t(mindspore::session::LiteSession *session, const std::string &)> cb) {
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

  std::string net = "./nets/effnetb0_fwd_nofuse.ms";
  auto session = session::TrainSession::CreateTrainSession(net, context, false);
  ASSERT_NE(session, nullptr);

  std::string in = "./nets/effNet_input_x_1_3_224_224.bin";
  std::string out = "./nets/effNet_output_y_1_1000.bin";
  auto res = runNet(session, in, out, "650");
  delete session;
  delete context;
  ASSERT_EQ(res, 0);
}

TEST_F(NetworkTest, mobileface_net) {
  char *buf = nullptr;
  size_t net_size = 0;

  std::string net = "./nets/mobilefacenet0924.ms";
  ReadFile(net.c_str(), &net_size, &buf);
  auto model = lite::Model::Import(buf, net_size);
  delete[] buf;
  auto context = new lite::Context;
  ASSERT_NE(context, nullptr);
  context->device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = lite::NO_BIND;
  context->thread_num_ = 1;

  auto session = session::LiteSession::CreateSession(context);
  ASSERT_NE(session, nullptr);
  auto ret = session->CompileGraph(model);
  ASSERT_EQ(lite::RET_OK, ret);
  // session->Eval();

  std::string in = "./nets/facenet_input.f32";
  std::string out = "./nets/facenet_output.f32";
  auto res = runNet(session, in, out, "354", true);

  ASSERT_EQ(res, 0);
  delete model;
  delete session;
  delete context;
}

TEST_F(NetworkTest, noname) {
  std::string net = "./nets/lenet_train.ms";
  lite::Context context;
  context.device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = lite::NO_BIND;
  context.thread_num_ = 1;

  lite::TrainCfg cfg;
  cfg.loss_name_ = "nhwc";
  auto session = mindspore::session::TrainSession::CreateTrainSession(net, &context, true, &cfg);
  ASSERT_NE(session, nullptr);
  auto tensors_map = session->GetOutputs();
  auto tensor_names = session->GetOutputTensorNames();
  EXPECT_EQ(tensors_map.size(), 1);
  EXPECT_EQ(tensors_map.begin()->first, "24");
  EXPECT_EQ(tensor_names.size(), 1);
  EXPECT_EQ(tensor_names.at(0), "Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense/BiasAdd-op107");
  delete session;
}

TEST_F(NetworkTest, setname) {
  std::string net = "./nets/lenet_train.ms";
  lite::Context context;
  context.device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = lite::NO_BIND;
  context.thread_num_ = 1;

  lite::TrainCfg train_cfg;
  train_cfg.loss_name_ = "nhwc";

  auto session = mindspore::session::TrainSession::CreateTrainSession(net, &context, true, &train_cfg);
  ASSERT_NE(session, nullptr);

  auto tensors_map = session->GetOutputs();
  auto tensor_names = session->GetOutputTensorNames();
  EXPECT_EQ(tensors_map.begin()->first, "8");
  EXPECT_EQ(tensor_names.at(0), "Default/network-WithLossCell/_backbone-LeNet5/max_pool2d-MaxPool2d/MaxPool-op88");
  delete session;
}

}  // namespace mindspore
