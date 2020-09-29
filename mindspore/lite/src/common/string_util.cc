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

#include "src/common/string_util.h"

namespace mindspore {
namespace lite {

std::vector<StringPack> ParseTensorBuffer(Tensor *tensor) {
  if (tensor->MutableData() == nullptr) {
    MS_LOG(ERROR) << "Tensor data is null, cannot be parsed";
    return std::vector<StringPack>{};
  }
  return ParseStringBuffer(tensor->MutableData());
}

std::vector<StringPack> ParseStringBuffer(const void *data) {
  const int32_t *offset = reinterpret_cast<const int32_t *>(data);
  int32_t num = *offset;
  std::vector<StringPack> buffer;
  for (int i = 0; i < num; i++) {
    offset += 1;
    buffer.push_back(StringPack{(*(offset + 1)) - (*offset), reinterpret_cast<const char *>(data) + (*offset)});
  }
  return buffer;
}

int WriteStringsToTensor(Tensor *tensor, const std::vector<StringPack> &string_buffer) {
  int32_t num = string_buffer.size();
  std::vector<int32_t> offset(num + 1);
  offset[0] = 4 * (num + 2);
  for (int i = 0; i < num; i++) {
    offset[i + 1] = offset[i] + string_buffer[i].len;
  }
  std::vector<int> shape = {offset[num]};
  tensor->set_shape(shape);
  void *data = tensor->MutableData();
  if (data == nullptr) {
    return RET_ERROR;
  }

  auto *string_info = reinterpret_cast<int32_t *>(data);
  auto *string_data = reinterpret_cast<char *>(data);

  string_info[0] = num;
  for (int i = 0; i <= num; i++) {
    string_info[i + 1] = offset[i];
  }
  for (int i = 0; i < num; i++) {
    memcpy(string_data + offset[i], string_buffer[i].data, string_buffer[i].len);
  }
  return RET_OK;
}

int WriteSeperatedStringsToTensor(Tensor *tensor, const std::vector<std::vector<StringPack>> &string_buffer) {
  int32_t num = string_buffer.size();
  std::vector<int32_t> offset(num + 1);
  offset[0] = 4 * (num + 2);
  std::vector<int> len(num);
  for (int i = 0; i < num; i++) {
    len[i] = 0;
    for (int j = 0; j < static_cast<int>(string_buffer[i].size()); j++) {
      len[i] += string_buffer[i][j].len;
    }
    offset[i + 1] = offset[i] + len[i];
  }

  std::vector<int> shape = {offset[num]};
  tensor->set_shape(shape);
  void *data = tensor->MutableData();
  if (data == nullptr) {
    return RET_ERROR;
  }

  auto *string_info = reinterpret_cast<int32_t *>(data);
  auto *string_data = reinterpret_cast<char *>(data);

  string_info[0] = num;
  for (int i = 0; i <= num; i++) {
    string_info[i + 1] = offset[i];
  }
  for (int i = 0; i < num; i++) {
    auto *dst = string_data + offset[i];
    for (auto string_part : string_buffer[i]) {
      memcpy(dst, string_part.data, string_part.len);
      dst += string_part.len;
    }
  }
  return RET_OK;
}

}  // namespace lite
}  // namespace mindspore
