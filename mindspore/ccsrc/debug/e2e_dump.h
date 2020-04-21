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
#ifndef MINDSPORE_E2E_DUMP_H
#define MINDSPORE_E2E_DUMP_H
#include <stdint.h>
#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>

namespace mindspore {
class Dump {
 public:
  Dump();

  ~Dump() = default;

  bool dump_enable() const { return dump_enable_; }

  bool trans_flag() const { return trans_flag_; }

  std::string dump_path() const { return dump_path_; }

  std::string dump_net_name() const { return dump_net_name_; }

  uint32_t dump_iter() const { return dump_iter_; }

  void UpdataCurIter() { cur_iter_++; }

  uint32_t cur_iter() const { return cur_iter_; }

  bool IsKernelNeedDump(const std::string &kernel_name);

  bool SetDumpConfFromJsonFile();

  static bool DumpToFile(const std::string &filename, const void *data, size_t len);

 protected:
  bool dump_enable_;
  bool trans_flag_;
  std::string dump_path_;
  std::string dump_net_name_;
  uint32_t dump_mode_;
  uint32_t dump_iter_;
  uint32_t cur_iter_;
  std::vector<std::string> dump_kernels_;

  static bool GetRealPath(const std::string &inpath, std::string *outpath);

  static bool CreateNotExistDirs(const std::string &path);

 private:
  bool ParseDumpConfig(const std::string &dump_config_file);
  bool IsConfigExist(const nlohmann::json &dumpSettings);
  bool IsConfigValid(const nlohmann::json &dumpSettings);
};

using DumpConfPtr = std::shared_ptr<Dump>;
}  // namespace mindspore
#endif  // MINDSPORE_E2E_DUMP_H
