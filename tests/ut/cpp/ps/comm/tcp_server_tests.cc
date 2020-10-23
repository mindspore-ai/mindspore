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

#include "ps/comm/tcp_client.h"
#include "ps/comm/tcp_server.h"
#include "common/common_test.h"

#include <thread>

namespace mindspore {
namespace ps {
namespace comm {
class TestTcpServer : public UT::Common {
 public:
  TestTcpServer() = default;
  void SetUp() override {
    server_ = new TcpServer("127.0.0.1", 9000);
    std::unique_ptr<std::thread> http_server_thread_(nullptr);
    http_server_thread_ = std::make_unique<std::thread>([&]() {
      server_->ReceiveMessage([](const TcpServer &server, const TcpConnection &conn, const void *buffer, size_t num) {
        EXPECT_STREQ(std::string(reinterpret_cast<const char *>(buffer), num).c_str(), "TCP_MESSAGE");
        server.SendMessage(conn, buffer, num);
      });
      server_->InitServer();
      server_->Start();
    });
    http_server_thread_->detach();
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
  }
  void TearDown() override {
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    client_->Stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    server_->Stop();
  }

  TcpClient *client_;
  TcpServer *server_;
  const std::string test_message_ = "TCP_MESSAGE";
};

TEST_F(TestTcpServer, ServerSendMessage) {
  client_ = new TcpClient("127.0.0.1", 9000);
  std::unique_ptr<std::thread> http_client_thread(nullptr);
  http_client_thread = std::make_unique<std::thread>([&]() {
    client_->ReceiveMessage([](const TcpClient &client, const void *buffer, size_t num) {
      EXPECT_STREQ(std::string(reinterpret_cast<const char *>(buffer), num).c_str(), "TCP_MESSAGE");
    });

    client_->InitTcpClient();
    client_->SendMessage(test_message_.c_str(), test_message_.size());
    client_->Start();
  });
  http_client_thread->detach();
}
}  // namespace comm
}  // namespace ps
}  // namespace mindspore