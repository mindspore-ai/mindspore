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
#ifndef MINDSPORE_SERVING_FILE_SYSTEM_OPERATION_H_
#define MINDSPORE_SERVING_FILE_SYSTEM_OPERATION_H_

#include <string>
#include <vector>
#include <ctime>

namespace mindspore {
namespace serving {
char *ReadFile(const char *file, size_t *size);
bool DirOrFileExist(const std::string &file_path);
std::vector<std::string> GetAllSubDirs(const std::string &dir_path);
time_t GetModifyTime(const std::string &file_path);
}  // namespace serving
}  // namespace mindspore

#endif  // !MINDSPORE_SERVING_FILE_SYSTEM_OPERATION_H_
