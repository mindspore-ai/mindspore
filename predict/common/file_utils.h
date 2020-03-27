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

#ifndef PREDICT_COMMON_FILE_UTILS_H_
#define PREDICT_COMMON_FILE_UTILS_H_

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <iostream>
#include <memory>
#include <fstream>
#include "common/utils.h"
#include "common/mslog.h"
#include "include/tensor.h"

namespace mindspore {
namespace predict {
char *ReadFile(const char *file, size_t *size);

std::string RealPath(const char *path);
}  // namespace predict
}  // namespace mindspore

#endif  // PREDICT_COMMON_FILE_UTILS_H_
