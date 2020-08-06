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
#ifndef MINNIE_TIMEPROFILE_TIMEPROFILE_H_
#define MINNIE_TIMEPROFILE_TIMEPROFILE_H_

#include <getopt.h>
#include <signal.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <utility>

#include "tools/common/flag_parser.h"
#include "src/common/file_utils.h"
#include "src/common/utils.h"
#include "schema/model_generated.h"
#include "include/model.h"
#include "include/lite_session.h"


namespace mindspore {
namespace lite {

class MS_API TimeProfileFlags : public virtual FlagParser {
 public:
  TimeProfileFlags() {
    AddFlag(&TimeProfileFlags::model_path_, "modelPath", "Input model path", "");
    AddFlag(&TimeProfileFlags::in_data_path_, "inDataPath", "Input data path, if not set, use random input", "");
    AddFlag(&TimeProfileFlags::cpu_bind_mode_, "cpuBindMode",
            "Input -1 for MID_CPU, 1 for HIGHER_CPU, 0 for NO_BIND, defalut value: 1", 1);
    AddFlag(&TimeProfileFlags::loop_count_, "loopCount", "Run loop count", 10);
    AddFlag(&TimeProfileFlags::num_threads_, "numThreads", "Run threads number", 2);
  }

  ~TimeProfileFlags() override = default;

 public:
  std::string model_path_;
  std::string in_data_path_;
  int cpu_bind_mode_ = 1;
  int loop_count_;
  int num_threads_;
};

class MS_API TimeProfile {
 public:
  explicit TimeProfile(TimeProfileFlags *flags) : _flags(flags) {}
  ~TimeProfile() = default;

  int Init();
  int RunTimeProfile();

 private:
  int GenerateRandomData(size_t size, void *data);
  int GenerateInputData();
  int LoadInput();
  int ReadInputFile();
  int InitCallbackParameter();
  int InitSession();
  int PrintResult(const std::vector<std::string>& title, const std::map<std::string, std::pair<int, float>>& result);

 private:
  TimeProfileFlags *_flags;
  std::vector<mindspore::tensor::MSTensor *> ms_inputs_;
  session::LiteSession *session_;

  // callback parameters
  uint64_t op_begin_ = 0;
  int op_call_times_total_ = 0;
  float op_cost_total_ = 0.0f;
  std::map<std::string, std::pair<int, float>> op_times_by_type_;
  std::map<std::string, std::pair<int, float>> op_times_by_name_;

  session::KernelCallBack before_call_back_;
  session::KernelCallBack after_call_back_;
};

int MS_API RunTimeProfile(int argc, const char **argv);
}  // namespace lite
}  // namespace mindspore
#endif  // MINNIE_TIMEPROFILE_TIMEPROFILE_H_
