/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_ADMIN_ARG_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_ADMIN_ARG_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <sstream>
#include <thread>
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/engine/cache/cache_client.h"

namespace mindspore {
namespace dataset {

class CacheAdminArgHandler {
 public:
  static const int32_t kDefaultNumWorkers;
  static constexpr int32_t kDefaultSharedMemorySizeInGB = 4;
  static constexpr int32_t kDefaultLogLevel = 1;
  static constexpr float kMemoryCapRatio = 0.8;
  static const char kServerBinary[];

  // These are the actual command types to execute
  enum class CommandId : int16_t {
    kCmdHelp = 0,
    kCmdStart = 1,
    kCmdStop = 2,
    kCmdGenerateSession = 3,
    kCmdDestroySession = 4,
    kCmdListSessions = 5,
    kCmdServerInfo = 6,
    kCmdUnknown = 32767
  };

  CacheAdminArgHandler();

  virtual ~CacheAdminArgHandler();

  Status ParseArgStream(std::stringstream *arg_stream);

  Status RunCommand();

  void Help();

 private:
  // These are the supported argument string integer mappings
  enum class ArgValue : int16_t {
    kArgUnknown = 0,  // Must be at position 0.  invalid map lookups in arg_map_ default to value 0
    kArgStart = 1,
    kArgStop = 2,
    kArgHost = 3,
    kArgPort = 4,
    kArgHelp = 5,
    kArgGenerateSession = 6,
    kArgDestroySession = 7,
    kArgSpillDir = 8,
    kArgNumWorkers = 9,
    kArgSharedMemorySize = 10,
    kArgLogLevel = 11,
    kArgMemoryCapRatio = 12,
    kArgListSessions = 13,
    kArgServerInfo = 14,
    kArgNumArgs = 15  // Must be the last position to provide a count
  };

  Status StartServer(CommandId command_id);

  Status StopServer(CommandId command_id);

  Status ShowServerInfo();

  Status AssignArg(std::string option, int32_t *out_arg, std::stringstream *arg_stream,
                   CommandId command_id = CommandId::kCmdUnknown);

  Status AssignArg(std::string option, std::string *out_arg, std::stringstream *arg_stream,
                   CommandId command_id = CommandId::kCmdUnknown);

  Status AssignArg(std::string option, float *out_arg, std::stringstream *arg_stream,
                   CommandId command_id = CommandId::kCmdUnknown);

  Status Validate();

  CommandId command_id_;
  int32_t port_;
  int32_t num_workers_;
  int32_t shm_mem_sz_;
  int32_t log_level_;
  float memory_cap_ratio_;
  session_id_type session_id_;
  std::string hostname_;
  std::string spill_dir_;
  std::string trailing_args_;
  std::map<std::string, ArgValue> arg_map_;
  std::map<ArgValue, bool> used_args_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_ADMIN_ARG_H_
