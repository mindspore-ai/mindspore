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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_PLUGIN_SHARED_LIB_UTIL_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_PLUGIN_SHARED_LIB_UTIL_H_

#include <string>

namespace mindspore {
namespace dataset {

// This class is a collection of util functions which aims at abstracting the dependency on OS
class SharedLibUtil {
 public:
  static void *Load(const std::string &name);

  static void *FindSym(void *handle, const std::string &name);

  static int32_t Close(void *handle);

  static std::string ErrMsg();
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_PLUGIN_SHARED_LIB_UTIL_H_
