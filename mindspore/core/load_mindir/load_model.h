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
#ifndef MINDSPORE_CORE_LOAD_MODEL_H
#define MINDSPORE_CORE_LOAD_MODEL_H

#include <vector>
#include <string>
#include <memory>

#include "proto/mind_ir.pb.h"
#include "ir/func_graph.h"

namespace mindspore {
std::shared_ptr<FuncGraph> LoadMindIR(const std::string &file_name, bool is_lite = false);
std::shared_ptr<std::vector<char>> ReadProtoFile(const std::string &file);
std::shared_ptr<FuncGraph> ConvertStreamToFuncGraph(const char *buf, const size_t buf_size, bool is_lite = false);
}  // namespace mindspore
#endif  // MINDSPORE_CORE_LOAD_MODEL_H
