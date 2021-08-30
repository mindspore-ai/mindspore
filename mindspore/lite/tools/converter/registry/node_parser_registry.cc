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

#include "include/registry/node_parser_registry.h"
#include <map>
#include <mutex>
#include <string>

namespace mindspore {
namespace registry {
namespace {
std::map<converter::FmkType, std::map<std::string, converter::NodeParserPtr>> node_parser_room;
std::mutex node_mutex;
}  // namespace
NodeParserRegistry::NodeParserRegistry(converter::FmkType fmk_type, const std::string &node_type,
                                       const converter::NodeParserPtr &node_parser) {
  std::unique_lock<std::mutex> lock(node_mutex);
  node_parser_room[fmk_type][node_type] = node_parser;
}

converter::NodeParserPtr NodeParserRegistry::GetNodeParser(converter::FmkType fmk_type, const std::string &node_type) {
  auto iter_level1 = node_parser_room.find(fmk_type);
  if (iter_level1 == node_parser_room.end()) {
    return nullptr;
  }
  if (node_type.empty()) {
    return nullptr;
  }
  auto iter_level2 = iter_level1->second.find(node_type);
  if (iter_level2 == iter_level1->second.end()) {
    return nullptr;
  }
  return iter_level2->second;
}
}  // namespace registry
}  // namespace mindspore
