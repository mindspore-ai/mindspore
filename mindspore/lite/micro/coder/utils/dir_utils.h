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

#ifndef MINDSPORE_LITE_MICRO_CODER_UTILS_DIRS_H_
#define MINDSPORE_LITE_MICRO_CODER_UTILS_DIRS_H_
#include <string>
namespace mindspore::lite::micro {
#if defined(_WIN32) || defined(_WIN64)
static const char kSlash[] = "\\";
#else
static const char kSlash[] = "/";
#endif

int InitProjDirs(const std::string &project_root_dir, const std::string &proj_name);

bool DirExists(const std::string &dir_path);

bool FileExists(const std::string &dir_path);

}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_MICRO_CODER_UTILS_DIRS_H_
