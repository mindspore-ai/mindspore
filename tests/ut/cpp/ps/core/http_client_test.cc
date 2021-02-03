/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <memory>

#include "common/common_test.h"
#include "ps/core/http_server.h"
#include "ps/core/http_client.h"

using namespace std;

namespace mindspore {
namespace ps {
namespace core {
class TestHttpClient : public UT::Common {
 public:
  TestHttpClient() : server_(nullptr), http_server_thread_(nullptr) {}

  virtual ~TestHttpClient() = default;

  OnRequestReceive http_get_func = std::bind(
    [](std::shared_ptr<HttpMessageHandler> resp) {
      EXPECT_STREQ(resp->GetUriPath().c_str(), "/httpget");
      const unsigned char ret[] = "get request success!\n";
      resp->QuickResponse(200, ret, 22);
    },
    std::placeholders::_1);

  OnRequestReceive http_handler_func = std::bind(
    [](std::shared_ptr<HttpMessageHandler> resp) {
      std::string host = resp->GetRequestHost();
      EXPECT_STREQ(host.c_str(), "127.0.0.1");

      std::string path_param = resp->GetPathParam("key1");
      std::string header_param = resp->GetHeadParam("headerKey");
      unsigned char *data = nullptr;
      const uint64_t len = resp->GetPostMsg(&data);
      char post_message[len + 1];
      if (memset_s(post_message, len + 1, 0, len + 1) != 0) {
        MS_LOG(EXCEPTION) << "The memset_s error";
      }
      if (memcpy_s(post_message, len, data, len) != 0) {
        MS_LOG(EXCEPTION) << "The memset_s error";
      }
      EXPECT_STREQ(path_param.c_str(), "value1");
      EXPECT_STREQ(header_param.c_str(), "headerValue");
      EXPECT_STREQ(post_message, "postKey=postValue");

      const std::string rKey("headKey");
      const std::string rVal("headValue");
      const std::string rBody("post request success!\n");
      resp->AddRespHeadParam(rKey, rVal);
      resp->AddRespString(rBody);

      resp->SetRespCode(200);
      resp->SendResponse();
    },
    std::placeholders::_1);

  void SetUp() override {
    server_ = std::make_unique<HttpServer>("0.0.0.0", 9999);

    server_->RegisterRoute("/httpget", &http_get_func);
    server_->RegisterRoute("/handler", &http_handler_func);
    http_server_thread_ = std::make_unique<std::thread>([&]() { server_->Start(); });
    http_server_thread_->detach();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }

  void TearDown() override {
    server_->Stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  }

 private:
  std::unique_ptr<HttpServer> server_;
  std::unique_ptr<std::thread> http_server_thread_;
};

TEST_F(TestHttpClient, Get) {
  HttpClient client;
  std::map<std::string, std::string> headers = {{"headerKey", "headerValue"}};
  auto output = std::make_shared<std::vector<char>>();
  auto ret = client.Get("http://127.0.0.1:9999/httpget", output, headers);
  EXPECT_STREQ("get request success!\n", output->data());
  EXPECT_EQ(Status::OK, ret);
}

TEST_F(TestHttpClient, Post) {
  HttpClient client;
  std::map<std::string, std::string> headers = {{"headerKey", "headerValue"}};
  auto output = std::make_shared<std::vector<char>>();
  std::string post_data = "postKey=postValue";
  auto ret =
    client.Post("http://127.0.0.1:9999/handler?key1=value1", post_data.c_str(), post_data.length(), output, headers);
  EXPECT_STREQ("post request success!\n", output->data());
  EXPECT_EQ(Status::OK, ret);
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
