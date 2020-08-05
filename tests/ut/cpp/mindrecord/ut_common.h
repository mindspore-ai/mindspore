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

#ifndef TESTS_MINDRECORD_UT_UT_COMMON_H_
#define TESTS_MINDRECORD_UT_UT_COMMON_H_

#include <dirent.h>
#include <fstream>
#include <string>
#include <vector>

#include "utils/ms_utils.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "minddata/mindrecord/include/shard_index.h"
#include "minddata/mindrecord/include/shard_header.h"
#include "minddata/mindrecord/include/shard_index_generator.h"
#include "minddata/mindrecord/include/shard_writer.h"
using json = nlohmann::json;
using std::ifstream;
using std::pair;
using std::string;
using std::vector;

namespace mindspore {
namespace mindrecord {
namespace UT {
class Common : public testing::Test {
 public:
  std::string install_root;

  // every TEST_F macro will enter one
  virtual void SetUp();

  virtual void TearDown();

};
}  // namespace UT

/// \brief Format the INFO message to have the same length by inserting '=' at the start
/// and the end of the message.
/// \param[in] message the string to format
/// \param[in] message_total_length the string length of the output
///
/// return the formatted string
const std::string FormatInfo(const std::string &message, uint32_t message_total_length = 128);


void LoadData(const std::string &directory, std::vector<json> &json_buffer, const int max_num);

void LoadDataFromImageNet(const std::string &directory, std::vector<json> &json_buffer, const int max_num);

int Img2DataUint8(const std::vector<std::string> &img_absolute_path, std::vector<std::vector<uint8_t>> &bin_data);

int GetAbsoluteFiles(std::string directory, std::vector<std::string> &files_absolute_path);

void ShardWriterImageNet();

void ShardWriterImageNetOneSample();

void ShardWriterImageNetOpenForAppend(string filename);
}  // namespace mindrecord
}  // namespace mindspore
#endif  // TESTS_MINDRECORD_UT_UT_COMMON_H_
