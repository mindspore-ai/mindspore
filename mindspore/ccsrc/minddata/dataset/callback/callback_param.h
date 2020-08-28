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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_CALLBACK_PARAM_H
#define MINDSPORE_CCSRC_MINDDATA_DATASET_CALLBACK_PARAM_H

#include <nlohmann/json.hpp>

namespace mindspore {
namespace dataset {

/// Callback Param is the object a DatasetOp uses to pass run-time information to user defined function.
/// This is a prototype for now, more fields will be added
class CallbackParam {
 public:
  CallbackParam(int64_t epoch_num, int64_t cur_epoch_step, int64_t total_step_num)
      : cur_epoch_num_(epoch_num), cur_epoch_step_num_(cur_epoch_step), cur_step_num_(total_step_num) {}

  ~CallbackParam() = default;

  // these are constant public fields for easy access and consistency with python cb_param
  // the names and orders are consistent with batchInfo
  const int64_t cur_epoch_num_;       // current epoch
  const int64_t cur_epoch_step_num_;  // step number of the current epoch
  const int64_t cur_step_num_;        // step number since the first row
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CALLBACK_PARAM_H
