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

#include "ps/core/http_server.h"
#include "common/common_test.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <memory>

namespace mindspore {
namespace ps {
namespace core {

class TestHttpServer : public UT::Common {
 public:
  TestHttpServer() : server_(nullptr) {}

  virtual ~TestHttpServer() = default;

  static void testGetHandler(std::shared_ptr<HttpMessageHandler> resp) {
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
  }

  void SetUp() override {
    server_ = std::make_unique<HttpServer>("0.0.0.0", 9999);
    OnRequestReceive http_get_func = std::bind(
      [](std::shared_ptr<HttpMessageHandler> resp) {
        EXPECT_STREQ(resp->GetPathParam("key1").c_str(), "value1");
        EXPECT_STREQ(resp->GetUriQuery().c_str(), "key1=value1");
        EXPECT_STREQ(resp->GetRequestUri().c_str(), "/httpget?key1=value1");
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
        std::string post_param = resp->GetPostParam("postKey");
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
        EXPECT_STREQ(post_param.c_str(), "postValue");
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
    server_->RegisterRoute("/httpget", &http_get_func);
    server_->RegisterRoute("/handler", &http_handler_func);
    std::unique_ptr<std::thread> http_server_thread_(nullptr);
    http_server_thread_ = std::make_unique<std::thread>([&]() { server_->Start(); });
    http_server_thread_->detach();
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
  }

  void TearDown() override {
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    server_->Stop();
  }

 private:
  std::unique_ptr<HttpServer> server_;
};

TEST_F(TestHttpServer, httpGetQequest) {
  char buffer[100];
  FILE *file;
  std::string cmd = "curl -X GET http://127.0.0.1:9999/httpget?key1=value1";
  std::string result;
  const char *sysCommand = cmd.data();
  if ((file = popen(sysCommand, "r")) == nullptr) {
    return;
  }
  while (fgets(buffer, sizeof(buffer) - 1, file) != nullptr) {
    result += buffer;
  }
  EXPECT_STREQ("get request success!\n", result.c_str());
  pclose(file);
}

TEST_F(TestHttpServer, messageHandler) {
  char buffer[100];
  FILE *file;
  std::string cmd =
    R"(curl -X POST -d 'postKey=postValue' -i -H "Accept: application/json" -H "headerKey: headerValue"  http://127.0.0.1:9999/handler?key1=value1)";
  std::string result;
  const char *sysCommand = cmd.data();
  if ((file = popen(sysCommand, "r")) == nullptr) {
    return;
  }
  while (fgets(buffer, sizeof(buffer) - 1, file) != nullptr) {
    result += buffer;
  }
  EXPECT_STREQ("post request success!\n", result.substr(result.find("post")).c_str());
  pclose(file);
}

TEST_F(TestHttpServer, portErrorNoException) {
  auto server_exception = std::make_unique<HttpServer>("0.0.0.0", -1);
  OnRequestReceive http_handler_func = std::bind(TestHttpServer::testGetHandler, std::placeholders::_1);
  EXPECT_NO_THROW(server_exception->RegisterRoute("/handler", &http_handler_func));
}

TEST_F(TestHttpServer, addressException) {
  auto server_exception = std::make_unique<HttpServer>("12344.0.0.0", 9998);
  OnRequestReceive http_handler_func = std::bind(TestHttpServer::testGetHandler, std::placeholders::_1);
  ASSERT_THROW(server_exception->RegisterRoute("/handler", &http_handler_func), std::exception);
}

}  // namespace core
}  // namespace ps
}  // namespace mindspore
