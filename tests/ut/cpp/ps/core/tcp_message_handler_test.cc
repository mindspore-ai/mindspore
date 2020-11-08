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

#include "ps/core/tcp_message_handler.h"
#include "common/common_test.h"

#include <memory>
#include <thread>

namespace mindspore {
namespace ps {
namespace core {
class TestTcpMessageHandler : public UT::Common {
 public:
  using messageReceive = std::function<void(const CommMessage &message)>;
  TestTcpMessageHandler() = default;
  virtual ~TestTcpMessageHandler() = default;

  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(TestTcpMessageHandler, 4_Header_1003_Data) {
  TcpMessageHandler handler;
  handler.SetCallback([this](const CommMessage &message) { EXPECT_EQ(message.data().size(), 1000); });

  std::string data(1000, 'a');
  CommMessage message;
  message.set_data(data);
  uint32_t buf_size = message.ByteSizeLong();
  char result[1007];
  int ret = memcpy_s(result, 4, &buf_size, 4);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }

  std::vector<char> serialized(buf_size);
  message.SerializeToArray(serialized.data(), static_cast<int>(buf_size));
  memcpy_s(result + 4, buf_size, serialized.data(), buf_size);
  handler.ReceiveMessage(result, buf_size + 4);
}

TEST_F(TestTcpMessageHandler, 4_Header_1003_Data_4_Header_1003_Data) {
  TcpMessageHandler handler;
  handler.SetCallback([this](const CommMessage &message) { EXPECT_EQ(message.data().size(), 1000); });

  std::string data(1000, 'a');
  CommMessage message;
  message.set_data(data);
  uint32_t buf_size = message.ByteSizeLong();
  char result[2014];
  int ret = memcpy_s(result, 4, &buf_size, 4);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }
  std::vector<char> serialized(buf_size);
  message.SerializeToArray(serialized.data(), static_cast<int>(buf_size));
  ret = memcpy_s(result + 4, buf_size, serialized.data(), buf_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }
  ret = memcpy_s(result + 4 + buf_size, 4, &buf_size, 4);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }
  ret = memcpy_s(result + 4 + buf_size + 4, buf_size, serialized.data(), buf_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }

  handler.ReceiveMessage(result, 2 * buf_size + 4 * 2);
}

TEST_F(TestTcpMessageHandler, 4_Header_4090_Data_2_Header_2_header_4090_data) {
  TcpMessageHandler handler;
  handler.SetCallback([this](const CommMessage &message) { EXPECT_EQ(message.data().size(), 4087); });

  std::string data(4087, 'a');
  CommMessage message;
  message.set_data(data);
  uint32_t buf_size = message.ByteSizeLong();
  char result[4096];
  int ret = memcpy_s(result, 4, &buf_size, 4);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }
  std::vector<char> serialized(buf_size);
  message.SerializeToArray(serialized.data(), static_cast<int>(buf_size));
  ret = memcpy_s(result + 4, buf_size, serialized.data(), buf_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }

  ret = memcpy_s(result + 4 + buf_size, 2, &buf_size, 2);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }

  handler.ReceiveMessage(result, 4096);

  ret = memcpy_s(result, 2, &buf_size + 2, 2);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }
  ret = memcpy_s(result + 2, buf_size, serialized.data(), buf_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }

  handler.ReceiveMessage(result, 4092);
}

TEST_F(TestTcpMessageHandler, 4_Header_4088_Data_4_Header_4088_data) {
  TcpMessageHandler handler;
  handler.SetCallback([this](const CommMessage &message) { EXPECT_EQ(message.data().size(), 4085); });

  std::string data(4085, 'a');
  CommMessage message;
  message.set_data(data);
  uint32_t buf_size = message.ByteSizeLong();
  char result[4096];
  int ret = memcpy_s(result, 4, &buf_size, 4);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }
  std::vector<char> serialized(buf_size);
  message.SerializeToArray(serialized.data(), static_cast<int>(buf_size));
  ret = memcpy_s(result + 4, buf_size, serialized.data(), buf_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }

  ret = memcpy_s(result + 4 + buf_size, 4, &buf_size, 4);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }

  handler.ReceiveMessage(result, 4096);

  ret = memcpy_s(result, buf_size, serialized.data(), buf_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }

  handler.ReceiveMessage(result, 4088);
}

}  // namespace comm
}  // namespace ps
}  // namespace mindspore