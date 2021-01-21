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

#include "ps/core/tcp_client.h"
#include "ps/core/tcp_server.h"
#include "common/common_test.h"

#include <memory>
#include <thread>

namespace mindspore {
namespace ps {
namespace core {
class TestTcpServer : public UT::Common {
 public:
  TestTcpServer() : client_(nullptr), server_(nullptr) {}
  virtual ~TestTcpServer() = default;

  void SetUp() override {
    server_ = std::make_unique<TcpServer>("127.0.0.1", 0);
    std::unique_ptr<std::thread> http_server_thread_(nullptr);
    http_server_thread_ = std::make_unique<std::thread>([=]() {
      server_->SetMessageCallback([=](std::shared_ptr<TcpConnection> conn, std::shared_ptr<MessageMeta> meta,
                                      const Protos &protos, const void *data, size_t size) {
        KVMessage kv_message;
        kv_message.ParseFromArray(data, size);
        EXPECT_EQ(2, kv_message.keys_size());
        server_->SendMessage(conn, meta, protos, data, size);
      });
      server_->Init();
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

  std::unique_ptr<TcpClient> client_;
  std::unique_ptr<TcpServer> server_;
};

TEST_F(TestTcpServer, ServerSendMessage) {
  client_ = std::make_unique<TcpClient>("127.0.0.1", server_->BoundPort());
  std::cout << server_->BoundPort() << std::endl;
  std::unique_ptr<std::thread> http_client_thread(nullptr);
  http_client_thread = std::make_unique<std::thread>([&]() {
    client_->SetMessageCallback([&](std::shared_ptr<MessageMeta> meta, const Protos &, const void *data, size_t size) {
      KVMessage message;
      message.ParseFromArray(data, size);
      EXPECT_EQ(2, message.keys_size());
    });

    client_->Init();

    KVMessage kv_message;
    std::vector<int> keys{1, 2};
    std::vector<int> values{3, 4};
    *kv_message.mutable_keys() = {keys.begin(), keys.end()};
    *kv_message.mutable_values() = {values.begin(), values.end()};

    auto message_meta = std::make_shared<MessageMeta>();
    message_meta->set_cmd(NodeCommand::SEND_DATA);

    client_->SendMessage(message_meta, Protos::RAW, kv_message.SerializeAsString().data(), kv_message.ByteSizeLong());

    client_->Start();
  });
  http_client_thread->detach();
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore