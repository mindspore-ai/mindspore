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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_UTILS_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_UTILS_H
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include "securec/include/securec.h"
#include "mindapi/base/base.h"
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::STATUS;

namespace mindspore {
namespace lite {
namespace converter {
bool RemoveDir(const std::string &path);
int InnerPredict(const std::string &model_name, const std::string &in_data_file,
                 const std::vector<std::string> &output_names, const std::string &dump_directory,
                 const std::vector<std::vector<int64_t>> &input_shapes = std::vector<std::vector<int64_t>>());
}  // namespace converter
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_NNIE_NNIE_SEGDATA_GENERATE_H
