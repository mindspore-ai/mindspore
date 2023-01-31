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
#ifndef MINDSPORE_LITE_PROVIDERS_NNIE_SRC_NNIE_CFG_PARSER_H_
#define MINDSPORE_LITE_PROVIDERS_NNIE_SRC_NNIE_CFG_PARSER_H_
#include <vector>
#include <map>
#include <string>
#include "include/api/kernel.h"
#include "include/hi_type.h"

namespace mindspore {
namespace nnie {
typedef struct {
  HI_U64 phy_;
  HI_U32 size_;
} MEM_ITEM;
/**
 * Flags is a config container.
 * Member objects:
 *  1.time_step_: step num only for rnn or lstm model. Default is 1.
 *  2.max_roi_num_: maximum number of ROI area, which is single picture supports, must be greater than 0.Default is 300.
 *  3.core_ids_: running kernels' id, support multi-core, separated by commas when setting, such as {0, 1, 2}.
 *               each element must be an integer, wch meet such inequality 0 <= val < 8.
 *               Default is {0}.
 */
class Flags {
 public:
  Flags() = default;
  ~Flags() = default;
  int Init(const kernel::Kernel &kernel);

 public:
  int time_step_{1};
  int max_roi_num_{300};
  std::vector<int> core_ids_{0};
  bool keep_origin_output_{false};

 private:
  int ParserInt(const std::map<std::string, std::string> &nnie_arg, const std::string key, int *val);
  int ParserBool(const std::map<std::string, std::string> &nnie_arg, const std::string key, bool *val);
};
}  // namespace nnie
}  // namespace mindspore
#endif
