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

#include "common/common_test.h"
#include "ps/comm/tcp_client.h"

namespace mindspore {
namespace ps {
namespace comm {
class TestTcpClient : public UT::Common {
 public:
  TestTcpClient() = default;
};

TEST_F(TestTcpClient, InitClientIPError) {
  auto client = new TcpClient("127.0.0.13543", 9000);
  client->ReceiveMessage(
    [](const TcpClient &client, const void *buffer, size_t num) { client.SendMessage(buffer, num); });

  ASSERT_THROW(client->InitTcpClient(), std::exception);
}

TEST_F(TestTcpClient, InitClientPortErrorNoException) {
  auto client = new TcpClient("127.0.0.1", -1);
  client->ReceiveMessage(
    [](const TcpClient &client, const void *buffer, size_t num) { client.SendMessage(buffer, num); });

  EXPECT_NO_THROW(client->InitTcpClient());
}

}  // namespace comm
}  // namespace ps
}  // namespace mindspore