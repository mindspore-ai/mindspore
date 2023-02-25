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

#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_COMMON_SHARD_UTILS_H_
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_COMMON_SHARD_UTILS_H_

#ifndef _CRT_RAND_S
#define _CRT_RAND_S
#endif
#include <stdlib.h>
#ifndef _MSC_VER
#include <libgen.h>
#endif
#include <limits.h>
#include <sys/stat.h>
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
#include <sys/statfs.h>
#include <sys/wait.h>
#endif
#include <cassert>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "minddata/mindrecord/include/common/log_adapter.h"
#include "minddata/mindrecord/include/shard_error.h"
#include "nlohmann/json.hpp"
#include "./sqlite3.h"

/* To be used when dlog is ok #include "./slog.h" */
#ifdef DEBUG
#define MS_ASSERT(f) assert(f)
#else
#define MS_ASSERT(f) ((void)0)
#endif

namespace mindspore {
namespace mindrecord {
using json = nlohmann::json;

const int kInt0 = 0;
const int kInt1 = 1;
const int kInt2 = 2;
const int kInt3 = 3;
const int kUnsignedInt4 = 4;

enum LabelCategory { kSchemaLabel, kStatisticsLabel, kIndexLabel };

const char kVersion[] = "3.0";
const std::vector<std::string> kSupportedVersion = {"2.0", kVersion};

enum ShardType {
  kNLP = 0,
  kCV = 1,
};

enum TaskType {
  kCommonTask = 0,
  kPaddedTask = 1,
};
enum SamplerType { kCustomTopNSampler, kCustomTopPercentSampler, kSubsetRandomSampler, kPKSampler, kSubsetSampler };

enum ShuffleType { kShuffleCategory, kShuffleSample };

const double kEpsilon = 1e-7;

const int kThreadNumber = 14;

// Shard default parameters
const uint64_t kDefaultHeaderSize = 1 << 24;  // 16MB
const uint64_t kDefaultPageSize = 1 << 25;    // 32MB

// HeaderSize [16KB, 128MB]
const int kMinHeaderSize = 1 << 14;  // 16KB
const int kMaxHeaderSize = 1 << 27;  // 128MB

// PageSize [32KB, 256MB]
const int kMinPageSize = 1 << 15;  // 32KB
const int kMaxPageSize = 1 << 28;  // 256MB

// used by value length / schema id length / statistic id length ...
const uint64_t kInt64Len = 8;

// Minimum file size
const uint64_t kMinFileSize = kInt64Len;

const int kMinShardCount = 1;
const int kMaxShardCount = 1000;  // write
const int kMaxFileCount = 4096;   // read

const int kMinConsumerCount = 1;
const int kMaxConsumerCount = 128;

const int kMaxSchemaCount = 1;
const int kMaxThreadCount = 32;
const int kMaxFieldCount = 100;

// Minimum free disk size
const int kMinFreeDiskSize = 10;  // 10M

// dummy json
const json kDummyId = R"({"id": 0})"_json;

// translate type in schema to type in sqlite3(NULL, INTEGER, REAL, TEXT, BLOB)
const std::unordered_map<std::string, std::string> kDbJsonMap = {
  {"string", "TEXT"},     {"date", "DATE"},       {"date-time", "DATETIME"}, {"null", "NULL"},
  {"integer", "INTEGER"}, {"boolean", "BOOLEAN"}, {"array", "BLOB"},         {"number", "NUMERIC"},
  {"int32", "INTEGER"},   {"int64", "INTEGER"},   {"float32", "NUMERIC"},    {"float64", "NUMERIC"},
  {"bytes", "BLOB"}};

const char kPoint = '.';

const char kPathSeparator =
#if defined(_WIN32) || defined(_WIN64)
  '\\';
#else
  '/';
#endif

// field type used by check schema validation
const std::set<std::string> kFieldTypeSet = {"bytes", "string", "int32", "int64", "float32", "float64"};

// can be searched field list
const std::set<std::string> kScalarFieldTypeSet = {"string", "int32", "int64", "float32", "float64"};

// number field list
const std::set<std::string> kNumberFieldTypeSet = {"int32", "int64", "float32", "float64"};

const std::unordered_map<std::string, std::string> kTypesMap = {
  {"bool", "int32"},      {"int8", "int32"},      {"uint8", "bytes"},     {"int16", "int32"},
  {"uint16", "int32"},    {"int32", "int32"},     {"uint32", "int64"},    {"int64", "int64"},
  {"float16", "float32"}, {"float32", "float32"}, {"float64", "float64"}, {"string", "string"}};

/// \brief the max number of samples to enable lazy load
const uint32_t LAZY_LOAD_THRESHOLD = 5000000;

/// \brief parallel convert from vector<py::bytes> to vector<vector<uint8_t>>
const uint32_t kParallelConvert = 4;

/// \brief split a string using a character
/// \param[in] field target string
/// \param[in] separator a character for splitting
/// \return vector type result
std::vector<std::string> StringSplit(const std::string &field, char separator);

/// \brief validate field name is composed of '0-9' or 'a-z' or 'A-Z' or '_' or '-'
/// \param[in]  str target string
/// \return
bool ValidateFieldName(const std::string &str);

/// \brief get the filename by the path
/// \param s file path
/// \param fn_ptr shared ptr of file name
/// \return Status
Status GetFileName(const std::string &path, std::shared_ptr<std::string> *fn_ptr);

/// \brief get parent dir
/// \param path file path
/// \param pd_ptr shared ptr of parent path
/// \return Status
Status GetParentDir(const std::string &path, std::shared_ptr<std::string> *pd_ptr);

bool CheckIsValidUtf8(const std::string &str);

/// \brief judge if a path is legal file
/// \param path file path
/// \return Whether the path is legal or not
Status CheckFile(const std::string &path);

enum DiskSizeType { kTotalSize = 0, kFreeSize };

/// \brief get the free space about the disk
/// \param str_dir file path
/// \param disk_type: kTotalSize / kFreeSize
/// \param size: shared ptr of size in Megabytes
/// \return Status
Status GetDiskSize(const std::string &str_dir, const DiskSizeType &disk_type, std::shared_ptr<uint64_t> *size);

/// \brief get the max hardware concurrency
/// \return max concurrency
uint32_t GetMaxThreadNum();

/// \brief get absolute path of all mindrecord files
/// \param path path to one fo mindrecord files
/// \param addresses relative path of all mindrecord files
/// \param ds shared ptr of vector of absolute path
/// \return Status
Status GetDatasetFiles(const std::string &path, const json &addresses, std::shared_ptr<std::vector<std::string>> *ds);

/// \brief get random
/// \return std::mt19937
std::mt19937 GetRandomDevice();
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_COMMON_SHARD_UTILS_H_
