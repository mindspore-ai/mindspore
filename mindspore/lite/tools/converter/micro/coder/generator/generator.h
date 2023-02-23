/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_GENERATOR_GENERATOR_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_GENERATOR_GENERATOR_H_

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
#include "coder/log.h"
#include "coder/config.h"
#include "coder/context.h"
#include "coder/utils/type_cast.h"

namespace mindspore::lite::micro {
class Generator {
 public:
  explicit Generator(std::unique_ptr<CoderContext> ctx);
  virtual ~Generator();

  int GenerateCode();

 protected:
  int CreateCommonFiles();
  int CreateModelFiles();
  virtual int CodeNetHFile() = 0;
  virtual int CodeNetCFile() = 0;
  virtual void CodeNetExecuteFunc(std::ofstream &ofs) = 0;
  int CodeWeightFile();
  virtual int CodeRegKernelHFile();

  void CodeCommonNetH(std::ofstream &ofs);
  void CodeCommonNetC(std::ofstream &ofs);

  Configurator *config_{nullptr};
  std::unique_ptr<CoderContext> ctx_{nullptr};

  bool is_get_quant_args_{false};
  std::string model_dir_;
  std::string net_inc_hfile_;
  std::string net_src_cfile_;
  std::string net_weight_hfile_;
  std::string net_weight_cfile_;
  std::string net_model_cfile_;

  std::string net_include_file_path_;
  std::string net_src_file_path_;
  std::string net_main_file_path_;

 private:
  int CodeSourceCMakeFile();
  int CodeStaticContent();
  int CodeBenchmarkHFile(const std::string &file);
  int CodeModelHandleHFile();
  int CodeCommonModelFile();
  int CodeMSModelImplement();
  int CodeDataCFile();
  int CodeAllocatorFile();

  std::string cmake_file_name_{"net.cmake"};
#ifdef _MSC_VER
  unsigned int user_umask_ = 18;
  unsigned int origin_umask_ = 0;
#else
  // the user's generated file's permission
  mode_t user_umask_ = 0022;
  // the origin file's permission
  mode_t origin_umask_ = 0000;
#endif
};
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_GENERATOR_GENERATOR_H_
