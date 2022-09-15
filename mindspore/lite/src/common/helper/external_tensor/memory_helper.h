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

#ifndef MINDSPORE_LITE_SRC_COMMON_HELPER_EXTERNAL_TENSOR_MEMORY_HELPER_H_
#define MINDSPORE_LITE_SRC_COMMON_HELPER_EXTERNAL_TENSOR_MEMORY_HELPER_H_

#include <map>
#include <string>

#include "src/common/helper/external_tensor/helper.h"

namespace mindspore::infer::helper {
class MemoryExternalTensorHelper : public ExternalTensorHelper {
 public:
  MemoryExternalTensorHelper() {}
  explicit MemoryExternalTensorHelper(bool is_copy_data) : is_copy_data_(is_copy_data) {}
  virtual ~MemoryExternalTensorHelper() { free_data(); }
  void *GetExternalTensorData(const mindspore::schema::ExternalData *external_info) override;
  void SetExternalTensorData(const mindspore::schema::ExternalData *external_info, void *data) override;

 private:
  void free_data();

 private:
  std::map<std::string, void *> data_map_;
  bool is_copy_data_ = true;
};
}  // namespace mindspore::infer::helper

#endif  // MINDSPORE_LITE_SRC_COMMON_HELPER_EXTERNAL_TENSOR_MEMORY_HELPER_H_
