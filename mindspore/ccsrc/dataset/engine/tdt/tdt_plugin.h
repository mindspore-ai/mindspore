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
#ifndef DATASET_ENGINE_TDT_TDT_PLUGIN_H_
#define DATASET_ENGINE_TDT_TDT_PLUGIN_H_

#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "tdt/tdt_host_interface.h"

#include "dataset/core/data_type.h"
#include "dataset/core/tensor.h"
#include "dataset/core/tensor_row.h"

namespace mindspore {
namespace dataset {
enum TdtStatus { SUCCESS, FAILED };

using tdt::DataItem;

class TdtPlugin {
 public:
  static std::shared_ptr<TdtPlugin> GetInstance();

  TdtStatus hostPush(TensorRow ts_row, bool is_wait, std::string channel_name, bool profilig, int32_t &time);

 private:
  TdtPlugin() {}

  TdtStatus getTdtType(DataType d_type, std::string &datatype);

  TdtStatus translate(const TensorRow &ts_row, std::vector<DataItem> &items);

  void *tdt_handle_ = nullptr;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_ENGINE_TDT_TDT_PLUGIN_H_
