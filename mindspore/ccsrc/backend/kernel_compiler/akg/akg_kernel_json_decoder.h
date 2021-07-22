/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_AKG_KERNEL_JSON_DECODER_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_AKG_KERNEL_JSON_DECODER_H_
#include <string>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>
#include "ir/scalar.h"
#include "ir/anf.h"
#include "ir/func_graph.h"

namespace mindspore {
namespace kernel {
class AkgKernelJsonDecoder {
 public:
  AkgKernelJsonDecoder() { nodes_map_.clear(); }
  ~AkgKernelJsonDecoder() = default;

  FuncGraphPtr DecodeFusedNodes(const nlohmann::json &kernel_json);
  FuncGraphPtr DecodeFusedNodes(const std::string &kernel_json_str);

 private:
  ParameterPtr DecodeParameter(const nlohmann::json &parameter_json, const FuncGraphPtr &func_graph);
  CNodePtr DecodeCNode(const nlohmann::json &cnode_json, const FuncGraphPtr &func_graph, const std::string &processor);
  AnfNodePtr DecodeOutput(const std::vector<nlohmann::json> &output_descs, const FuncGraphPtr &func_graph);
  std::map<std::string, AnfNodePtr> nodes_map_;
};

// infer abstract shape by device_shape and data_format
ShapeVector GetFakeAbstractShape(const ShapeVector &device_shape, const std::string &format);
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_AKG_KERNEL_JSON_DECODER_H_
