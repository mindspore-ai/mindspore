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

#ifndef MINDSPORE_MICRO_CODER_GENERATOR_GENERATOR_UTILS_H_
#define MINDSPORE_MICRO_CODER_GENERATOR_GENERATOR_UTILS_H_

#include <map>
#include <string>
#include <vector>
#include "src/tensor.h"

namespace mindspore::lite::micro {

int WriteContentToFile(const std::string &file, const std::string &content);

void CodeReadModelParams(const std::map<std::string, Tensor *> &saved_weights,
                         const std::map<Tensor *, std::string> &tensors_map, std::ofstream &ofs);

int SaveDataToNet(const std::map<std::string, Tensor *> &tensors_map, const std::string &net_file);

void CodeModelParamsDefine(const std::map<std::string, Tensor *> &address_map, std::ofstream &hfile,
                           std::ofstream &cfile);

void CodeModelParamsDefineAndData(const std::map<std::string, Tensor *> &address_map, std::ofstream &hfile,
                                  std::ofstream &cfile);

int PrintMicroTensors(std::ofstream &ofs, std::vector<Tensor *> tensors, const std::string &name,
                      const std::map<Tensor *, std::string> &tensors_map);

void IncludeCmsisDirectories(std::ofstream &ofs);

}  // namespace mindspore::lite::micro

#endif  // MINDSPORE_MICRO_CODER_GENERATOR_GENERATOR_UTILS_H_
