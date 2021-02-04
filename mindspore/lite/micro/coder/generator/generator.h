/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MICRO_CODER_GENERATOR_H_
#define MINDSPORE_MICRO_CODER_GENERATOR_H_

#include <sys/stat.h>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "include/errorcode.h"
#include "src/tensor.h"
#include "coder/coder_config.h"
#include "coder/coder_context.h"
#include "coder/utils/print_utils.h"

namespace mindspore::lite::micro {
constexpr int kWarmUp = 3;

class Generator {
 public:
  explicit Generator(std::unique_ptr<CoderContext> ctx);
  virtual ~Generator();

  int GenerateCode();

 protected:
  virtual int CodeTestFile() = 0;
  virtual int CodeNetHFile() = 0;
  virtual int CodeNetCFile() = 0;
  virtual int CodeCMakeFile();
  virtual int CodeWeightFile();

  void CodeNetFileInclude(std::ofstream &ofs);
  int CodeNetFileInputOutput(std::ofstream &ofs);
  void CodeNetFileMembuffer(std::ofstream &ofs);
  void CodeNetRunFunc(std::ofstream &ofs);
  int CodeGraphInOutQuanArgs(std::ofstream &ofs);
  void CodeFreeResource(const std::map<std::string, Tensor *> &address_map, std::ofstream &ofs) const;

  Configurator *config_{nullptr};
  std::unique_ptr<CoderContext> ctx_{nullptr};

  bool is_get_quant_args_{false};
  std::string net_inc_hfile_;
  std::string net_src_cfile_;
  std::string net_main_cfile_;
  std::string net_weight_hfile_;

  std::string net_inc_file_path_;
  std::string net_src_file_path_;
  std::string net_main_file_path_;

 private:
  int CodeTestCMakeFile();
  int CodeStaticContent();
  int CodeCMakeExecutableFile(std::ofstream &ofs) const;
  void CodeWeightInitFunc(const std::map<std::string, Tensor *> &address_map, std::ofstream &ofs);

  std::string cmake_file_name_{"net.cmake"};
  // the user's generated file's permission
  mode_t user_umask_ = 0022;
  // the origin file's permission
  mode_t origin_umask_ = 0000;
};

}  // namespace mindspore::lite::micro

#endif  // MINDSPORE_MICRO_CODER_GENERATOR_H_
