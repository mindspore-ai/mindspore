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

#ifndef MINDSPORE_CCSRC_UTILS_MISC_H_
#define MINDSPORE_CCSRC_UTILS_MISC_H_

#include <cxxabi.h>
#include <list>
#include <memory>
#include <string>
#include <sstream>

#include "utils/log_adapter.h"

namespace mindspore {

extern const int RET_SUCCESS;
extern const int RET_FAILED;
extern const int RET_CONTINUE;
extern const int RET_BREAK;

// demangle the name to make it human reablable.
extern std::string demangle(const char* name);

}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_UTILS_MISC_H_
