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

#include "predict/generator/utils/ir_model_util.h"
namespace mindspore {
namespace generator {
IRModelUtil &IRModelUtil::GetInstance() {
  static IRModelUtil instance;
  return instance;
}

void IRModelUtil::Init() {
  MS_LOG(INFO) << "IRModel init success";
  version_ = "defaultVersion";
  stream_num_ = 0;
  event_num_ = 0;
  batch_num_ = 0;
  memory_size_ = 0;
  weight_size_ = 0;
  var_size_ = 0;
  logic_mem_base_ = 0;
  logic_var_base_ = 0;
  logic_var_base_ = 0;
  priority_ = 0;
  is_enable_save_model_ = false;
  min_static_offset_ = 0;
  max_dynamic_offset_ = 0;
}
}  // namespace generator
}  // namespace mindspore
