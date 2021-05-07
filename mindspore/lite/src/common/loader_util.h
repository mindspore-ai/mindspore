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

#ifndef MINDSPORE_LITE_SRC_COMMON_LOADER_UTIL_H_
#define MINDSPORE_LITE_SRC_COMMON_LOADER_UTIL_H_

namespace mindspore {
namespace lite {

class DynamicLibraryLoader {
 public:
  int Open(const char *lib_path);
  void *GetFunc(const char *func_name);
  int Close();

 private:
  void *handler_;
};

}  // namespace lite
}  // namespace mindspore

#endif
