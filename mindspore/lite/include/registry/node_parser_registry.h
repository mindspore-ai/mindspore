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

#ifndef MINDSPORE_LITE_INCLUDE_REGISTRY_NODE_PARSER_REGISTRY_H_
#define MINDSPORE_LITE_INCLUDE_REGISTRY_NODE_PARSER_REGISTRY_H_

#include <string>
#include <vector>
#include "include/registry/node_parser.h"
#include "include/api/dual_abi_helper.h"

namespace mindspore {
namespace registry {
/// \brief NodeParserRegistry defined registration of NodeParser.
class MS_API NodeParserRegistry {
 public:
  /// \brief Constructor of NodeParserRegistry to register NodeParser.
  ///
  /// \param[in] fmk_type Define the framework.
  /// \param[in] node_type Define the type of the node to be resolved.
  /// \param[in] node_parser Define the NodeParser instance to parse the node.
  inline NodeParserRegistry(converter::FmkType fmk_type, const std::string &node_type,
                            const converter::NodeParserPtr &node_parser);

  /// \brief Destructor
  ~NodeParserRegistry() = default;

  /// \brief Static method to obtain NodeParser instance of a certain node.
  ///
  /// \param[in] fmk_type Define the framework.
  /// \param[in] node_type Define the type of the node to be resolved.
  ///
  /// \return NodeParser instance.
  inline static converter::NodeParserPtr GetNodeParser(converter::FmkType fmk_type, const std::string &node_type);

 private:
  NodeParserRegistry(converter::FmkType fmk_type, const std::vector<char> &node_type,
                     const converter::NodeParserPtr &node_parser);
  static converter::NodeParserPtr GetNodeParser(converter::FmkType fmk_type, const std::vector<char> &node_type);
};

NodeParserRegistry::NodeParserRegistry(converter::FmkType fmk_type, const std::string &node_type,
                                       const converter::NodeParserPtr &node_parser)
    : NodeParserRegistry(fmk_type, StringToChar(node_type), node_parser) {}

converter::NodeParserPtr NodeParserRegistry::GetNodeParser(converter::FmkType fmk_type, const std::string &node_type) {
  return GetNodeParser(fmk_type, StringToChar(node_type));
}

/// \brief Defined registering macro to register NodeParser instance.
///
/// \param[in] fmk_type Define the framework.
/// \param[in] node_type Define the type of the node to be resolved.
/// \param[in] node_parser instance corresponding with its framework and node type.
#define REG_NODE_PARSER(fmk_type, node_type, node_parser) \
  static mindspore::registry::NodeParserRegistry g_##fmk_type##node_type##ParserReg(fmk_type, #node_type, node_parser);
}  // namespace registry
}  // namespace mindspore

#endif  // MINDSPORE_LITE_INCLUDE_REGISTRY_NODE_PARSER_REGISTRY_H_
