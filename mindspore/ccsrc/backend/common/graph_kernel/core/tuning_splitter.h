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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_TUNING_SPLITTER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_TUNING_SPLITTER_H_

#include <string>
#include <nlohmann/json.hpp>
#include "ir/func_graph.h"
#include "backend/common/graph_kernel/core/split_schemer.h"

namespace mindspore::graphkernel {
class TuningSplitSchemer : public CommonSplitSchemer {
 public:
  explicit TuningSplitSchemer(const std::string &path) : tuning_path_(path) {}
  ~TuningSplitSchemer() = default;
  bool Split(const FuncGraphPtr &func_graph) override;

 protected:
  bool ReadCache(const std::string &filename, nlohmann::json *result) const;
  bool ParseResult(const AnfNodePtrList &nodes, const nlohmann::json &result);

  std::string tuning_path_;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_TUNING_SPLITTER_H_
