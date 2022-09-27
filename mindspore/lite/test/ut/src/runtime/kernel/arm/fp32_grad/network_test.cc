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
#include "include/errorcode.h"
#include "include/train/train_cfg.h"
#include "src/train/train_session.h"
#include "src/common/log_adapter.h"
#include "src/common/file_utils.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/cpu/fp32_grad/convolution.h"

using mindspore::lite::RET_OK;
namespace mindspore {
class NetworkTest : public mindspore::CommonTest {
 public:
  NetworkTest() {}
  int32_t runNet(mindspore::lite::LiteSession *session, const std::string &in, const std::string &out,
                 const char *tensor_name, bool debug = false);
};

int32_t fileIterator(mindspore::lite::LiteSession *session, const std::string &path,
                     std::function<int32_t(mindspore::lite::LiteSession *session, const std::string &)> cb) {
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

int32_t NetworkTest::runNet(mindspore::lite::LiteSession *session, const std::string &in, const std::string &out,
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

TEST_F(NetworkTest, mobileface_net) {
  size_t size = 0;
  char *buff = lite::ReadFile("./nets/mobilefacenet0924.ms", &size);
  auto model = lite::Model::Import(buff, size);
  delete[] buff;
  auto context = std::make_shared<lite::InnerContext>();
  ASSERT_NE(context, nullptr);
  context->device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = lite::NO_BIND;
  context->thread_num_ = 1;

  auto session = lite::LiteSession::CreateSession(context);
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
}
}  // namespace mindspore
