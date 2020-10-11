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

#include "ps/comm/http_server.h"
#include "common/common_test.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

namespace mindspore {
namespace ps {
namespace comm {

class TestHttpServer : public UT::Common {
 public:
  TestHttpServer() {}

  static void testGetHandler(HttpMessageHandler *resp) {
    std::string host = resp->GetRequestHost();
    EXPECT_STREQ(host.c_str(), "127.0.0.1");

    std::string path_param = resp->GetPathParam("key1");
    std::string header_param = resp->GetHeadParam("headerKey");
    std::string post_param = resp->GetPostParam("postKey");
    std::string post_message = resp->GetPostMsg();
    EXPECT_STREQ(path_param.c_str(), "value1");
    EXPECT_STREQ(header_param.c_str(), "headerValue");
    EXPECT_STREQ(post_param.c_str(), "postValue");
    EXPECT_STREQ(post_message.c_str(), "postKey=postValue");

    const std::string rKey("headKey");
    const std::string rVal("headValue");
    const std::string rBody("post request success!\n");
    resp->AddRespHeadParam(rKey, rVal);
    resp->AddRespString(rBody);

    resp->SetRespCode(200);
    resp->SendResponse();
  }

  void SetUp() override {
    server_ = new HttpServer("0.0.0.0", 9999);
    server_->RegisterRoute("/httpget", [](HttpMessageHandler *resp) {
      EXPECT_STREQ(resp->GetPathParam("key1").c_str(), "value1");
      EXPECT_STREQ(resp->GetUriQuery().c_str(), "key1=value1");
      EXPECT_STREQ(resp->GetRequestUri().c_str(), "/httpget?key1=value1");
      EXPECT_STREQ(resp->GetUriPath().c_str(), "/httpget");
      resp->QuickResponse(200, "get request success!\n");
    });
    server_->RegisterRoute("/handler", TestHttpServer::testGetHandler);
    std::unique_ptr<std::thread> http_server_thread_(nullptr);
    http_server_thread_.reset(new std::thread([&]() { server_->Start(); }));
    http_server_thread_->detach();
  }

  void TearDown() override { server_->Stop(); }

 private:
  HttpServer *server_;
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

TEST_F(TestHttpServer, portException) {
  HttpServer *server_exception = new HttpServer("0.0.0.0", -1);
  ASSERT_THROW(server_exception->RegisterRoute("/handler", TestHttpServer::testGetHandler), std::exception);
}

TEST_F(TestHttpServer, addressException) {
  HttpServer *server_exception = new HttpServer("12344.0.0.0", 9998);
  ASSERT_THROW(server_exception->RegisterRoute("/handler", TestHttpServer::testGetHandler), std::exception);
}

}  // namespace comm
}  // namespace ps
}  // namespace mindspore
