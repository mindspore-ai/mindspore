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
#include <fcntl.h>
#include <iostream>
#include <memory>
#include "common/common_test.h"
#include "utils/system/file_system.h"
#include "utils/system/env.h"
#define private public
#include "include/utils/callback.h"
#include "nlohmann/json.hpp"
#include "utils/ms_utils.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#undef private

namespace mindspore {
class TestMemoryDumper : public UT::Common {
 public:
  TestMemoryDumper() {}
};

TEST_F(TestMemoryDumper, test_DumpToFileAbsPath) {
  int len = 1000;
  int data[len] = {0};
  for (uint32_t i = 0; i < len; i++) {
    data[i] = i % 10;
  }

  int ret;
  const std::string filename = "/tmp/dumpToFileTestFile";
  constexpr char kDumpJsonParserDumpToFile[] = "DumpJsonParserDumpToFile";
  static auto dump_json_parser_dump_to_file_callback =
    callback::CommonCallback::GetInstance()
      .GetCallback<bool, const std::string &, const void *, size_t, const ShapeVector &, TypeId>(
        kDumpJsonParserDumpToFile);
  if (dump_json_parser_dump_to_file_callback) {
    ret = dump_json_parser_dump_to_file_callback(filename, data, len * sizeof(int), ShapeVector{10, 100}, kNumberTypeInt32);
  } else {
    MS_LOG(WARNING) << "Fail to get DumpJsonParserDumpToFile, data dump function may not work.";
  }
  ASSERT_EQ(ret, true);

  int fd = open((filename + ".npy").c_str(), O_RDONLY);
  int header_size = 32;
  int npylen = len + header_size;
  int readBack[npylen] = {0};
  int readSize = read(fd, readBack, npylen * sizeof(int));
  (void)close(fd);
  ASSERT_EQ(readSize, npylen * sizeof(int));

  ret = true;
  for (uint32_t i = 0; i < len; i++) {
    // Skip the size of npy header.
    if (data[i] != readBack[i + header_size]) {
      ret = false;
      break;
    }
  }
  std::shared_ptr<system::FileSystem> fs = system::Env::GetFileSystem();
  if (fs->FileExist(filename)) {
    fs->DeleteFile(filename);
  }
  ASSERT_EQ(ret, true);
}

TEST_F(TestMemoryDumper, test_DumpToFileRelativePath) {
  int len = 1000;
  int data[len] = {0};
  for (uint32_t i = 0; i < len; i++) {
    data[i] = i % 10;
  }

  int ret;
  const std::string filename = "../../dumpToFileTestFile";
  constexpr char kDumpJsonParserDumpToFile[] = "DumpJsonParserDumpToFile";
  static auto dump_json_parser_dump_to_file_callback =
    callback::CommonCallback::GetInstance()
      .GetCallback<bool, const std::string &, const void *, size_t, const ShapeVector &, TypeId>(
        kDumpJsonParserDumpToFile);
  if (dump_json_parser_dump_to_file_callback) {
    ret = dump_json_parser_dump_to_file_callback(filename, data, len * sizeof(int), ShapeVector{10, 100}, kNumberTypeInt32);
  } else {
    MS_LOG(WARNING) << "Fail to get DumpJsonParserDumpToFile, data dump function may not work.";
  }
  ASSERT_EQ(ret, false);
}

TEST_F(TestMemoryDumper, test_DumpToFileNotExistDir) {
  int len = 1;
  int data[1] = {0};
  for (uint32_t i = 0; i < len; i++) {
    data[i] = i % 10;
  }

  const std::string filename = "./tmp/dumpToFileTestFile";
  int ret;
  constexpr char kDumpJsonParserDumpToFile[] = "DumpJsonParserDumpToFile";
  static auto dump_json_parser_dump_to_file_callback =
    callback::CommonCallback::GetInstance()
      .GetCallback<bool, const std::string &, const void *, size_t, const ShapeVector &, TypeId>(
        kDumpJsonParserDumpToFile);
  if (dump_json_parser_dump_to_file_callback) {
    ret = dump_json_parser_dump_to_file_callback(filename, data, len * sizeof(int), ShapeVector{10, 100}, kNumberTypeInt32);
  } else {
    MS_LOG(WARNING) << "Fail to get DumpJsonParserDumpToFile, data dump function may not work.";
  }
  ASSERT_EQ(ret, true);

  int fd = open((filename + ".npy").c_str(), O_RDONLY);
  int readBack[1000] = {0};
  int readSize = read(fd, readBack, len * sizeof(int));
  (void)close(fd);
  ASSERT_EQ(readSize, len * sizeof(int));

  ret = true;
  for (uint32_t i = 0; i < len; i++) {
    // Skip the size of npy header.
    if (data[i] != readBack[i + 1]) {
      ret = false;
      break;
    }
  }
  std::shared_ptr<system::FileSystem> fs = system::Env::GetFileSystem();
  if (fs->FileExist(filename)) {
    fs->DeleteFile(filename);
  }

  ASSERT_EQ(ret, true);
}
}  // namespace mindspore
