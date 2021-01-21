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

TEST_F(TestTcpMessageHandler, 16Header_2meta_1000Data) {
  TcpMessageHandler handler;
  handler.SetCallback([this](std::shared_ptr<MessageMeta> meta, const Protos &, const void *data, size_t size) {
    EXPECT_EQ(meta->ByteSizeLong(), 2);
    EXPECT_EQ(size, 1000);
  });

  std::string data(1000, 'a');

  char result[1018];

  MessageMeta meta;
  meta.set_request_id(1);
  EXPECT_EQ(meta.ByteSizeLong(), 2);

  MessageHeader header;
  header.message_proto_ = Protos::RAW;
  header.message_meta_length_ = meta.ByteSizeLong();
  header.message_length_ = data.length() + meta.ByteSizeLong();
  int ret = memcpy_s(result, kHeaderLen, &header, kHeaderLen);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }

  memcpy_s(result + kHeaderLen, meta.ByteSizeLong(), meta.SerializeAsString().data(), meta.ByteSizeLong());
  memcpy_s(result + kHeaderLen + meta.ByteSizeLong(), data.length(), data.data(), data.length());

  handler.ReceiveMessage(result, 1018);
}

TEST_F(TestTcpMessageHandler, 16Header_2meta_1000Data_16Header_2meta_1000Data) {
  TcpMessageHandler handler;
  handler.SetCallback([this](std::shared_ptr<MessageMeta> meta, const Protos &, const void *data, size_t size) {
    EXPECT_EQ(meta->ByteSizeLong(), 2);
    EXPECT_EQ(size, 1000);
  });

  std::string data(1000, 'a');

  char result[2036];

  MessageMeta meta;
  meta.set_request_id(1);
  EXPECT_EQ(meta.ByteSizeLong(), 2);

  MessageHeader header;
  header.message_proto_ = Protos::RAW;
  header.message_meta_length_ = meta.ByteSizeLong();
  header.message_length_ = data.length() + meta.ByteSizeLong();
  int ret = memcpy_s(result, kHeaderLen, &header, kHeaderLen);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }

  memcpy_s(result + kHeaderLen, meta.ByteSizeLong(), meta.SerializeAsString().data(), meta.ByteSizeLong());
  memcpy_s(result + kHeaderLen + meta.ByteSizeLong(), data.length(), data.data(), data.length());

  memcpy_s(result + kHeaderLen + meta.ByteSizeLong() + data.length(), kHeaderLen, &header, kHeaderLen);
  memcpy_s(result + kHeaderLen * 2 + meta.ByteSizeLong() + data.length(), meta.ByteSizeLong(),
           meta.SerializeAsString().data(), meta.ByteSizeLong());
  memcpy_s(result + kHeaderLen * 2 + meta.ByteSizeLong() * 2 + data.length(), data.length(), data.data(),
           data.length());

  handler.ReceiveMessage(result, 2036);
}

TEST_F(TestTcpMessageHandler, 16header_2meta_4070data_8header_8header_2meta_4070data) {
  TcpMessageHandler handler;
  handler.SetCallback([this](std::shared_ptr<MessageMeta> meta, const Protos &, const void *data, size_t size) {
    EXPECT_EQ(meta->ByteSizeLong(), 2);
    EXPECT_EQ(size, 4070);
  });

  std::string data(4070, 'a');

  char result[4096] = {0};

  MessageMeta meta;
  meta.set_request_id(1);
  EXPECT_EQ(meta.ByteSizeLong(), 2);

  MessageHeader header;
  header.message_proto_ = Protos::RAW;
  header.message_meta_length_ = meta.ByteSizeLong();
  header.message_length_ = data.length() + meta.ByteSizeLong();
  int ret = memcpy_s(result, kHeaderLen, &header, kHeaderLen);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }

  memcpy_s(result + kHeaderLen, meta.ByteSizeLong(), meta.SerializeAsString().data(), meta.ByteSizeLong());
  memcpy_s(result + kHeaderLen + meta.ByteSizeLong(), data.length(), data.data(), data.length());

  memcpy_s(result + kHeaderLen + meta.ByteSizeLong() + data.length(), 8, &header, 8);
  handler.ReceiveMessage(result, 4096);

  auto temp = reinterpret_cast<char *>(&header);
  memcpy_s(result, 8, temp + 8, 8);
  memcpy_s(result + 8, meta.ByteSizeLong(), meta.SerializeAsString().data(), meta.ByteSizeLong());
  memcpy_s(result + 8 + 2, data.length(), data.data(), data.length());

  handler.ReceiveMessage(result, 4080);
}

TEST_F(TestTcpMessageHandler, 16Header_2meta_4062Data_16Header_2meta_4062_data) {
  TcpMessageHandler handler;
  handler.SetCallback([this](std::shared_ptr<MessageMeta> meta, const Protos &, const void *data, size_t size) {
    EXPECT_EQ(meta->ByteSizeLong(), 2);
    EXPECT_EQ(size, 4062);
  });

  std::string data(4062, 'a');

  char result[4096] = {0};

  MessageMeta meta;
  meta.set_request_id(1);
  EXPECT_EQ(meta.ByteSizeLong(), 2);

  MessageHeader header;
  header.message_proto_ = Protos::RAW;
  header.message_meta_length_ = meta.ByteSizeLong();
  header.message_length_ = data.length() + meta.ByteSizeLong();
  int ret = memcpy_s(result, kHeaderLen, &header, kHeaderLen);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }

  memcpy_s(result + kHeaderLen, meta.ByteSizeLong(), meta.SerializeAsString().data(), meta.ByteSizeLong());
  memcpy_s(result + kHeaderLen + meta.ByteSizeLong(), data.length(), data.data(), data.length());
  memcpy_s(result + kHeaderLen + meta.ByteSizeLong() + data.length(), kHeaderLen, &header, kHeaderLen);

  handler.ReceiveMessage(result, 4096);

  memcpy_s(result, meta.ByteSizeLong(), meta.SerializeAsString().data(), meta.ByteSizeLong());
  memcpy_s(result + meta.ByteSizeLong(), data.length(), data.data(), data.length());

  handler.ReceiveMessage(result, 4064);
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore