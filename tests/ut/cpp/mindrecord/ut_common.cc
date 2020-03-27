/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "ut_common.h"

using mindspore::MsLogLevel::ERROR;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

namespace mindspore {
namespace mindrecord {
namespace UT {
#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

void Common::SetUp() {}

void Common::TearDown() {}

void Common::LoadData(const std::string &directory, std::vector<json> &json_buffer, const int max_num) {
  int count = 0;
  string input_path = directory;
  ifstream infile(input_path);
  if (!infile.is_open()) {
    MS_LOG(ERROR) << "can not open the file ";
    return;
  }
  string temp;
  while (getline(infile, temp) && count != max_num) {
    count++;
    json j = json::parse(temp);
    json_buffer.push_back(j);
  }
  infile.close();
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
}  // namespace UT

const std::string FormatInfo(const std::string &message, uint32_t message_total_length) {
  // if the message is larger than message_total_length
  std::string part_message = "";
  if (message_total_length < message.length()) {
    part_message = message.substr(0, message_total_length);
  } else {
    part_message = message;
  }
  int padding_length = static_cast<int>(message_total_length - part_message.length());
  std::string left_padding(static_cast<uint64_t>(ceil(padding_length / 2.0)), '=');
  std::string right_padding(static_cast<uint64_t>(floor(padding_length / 2.0)), '=');
  return left_padding + part_message + right_padding;
}
}  // namespace mindrecord
}  // namespace mindspore
