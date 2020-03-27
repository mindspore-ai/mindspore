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

#ifndef TRANSFORM_GRAPH_RUNNER_H_
#define TRANSFORM_GRAPH_RUNNER_H_

#include <set>
#include <string>
#include <vector>
#include <map>
#include <memory>

#include "transform/types.h"
#include "transform/util.h"
#include "ir/meta_tensor.h"
#include "transform/df_graph_manager.h"

namespace mindspore {
namespace transform {
using SessionOptions = std::map<std::string, std::string>;

struct GraphRunnerOptions {
  std::string target{"default_graph_runner"};
  SessionOptions options;
  // if sess_ptr is nullptr, GraphRunner will create a new ge session
  std::shared_ptr<ge::Session> sess_ptr{nullptr};
};

struct RunOptions {
  // graph's name
  std::string name;
};

class GraphRunner {
 public:
  explicit GraphRunner(const GraphRunnerOptions& options);
  ~GraphRunner() { sess_ = nullptr; }
  Status RunGraph(const RunOptions& options, const std::vector<MeTensorPtr>& inputs, std::vector<MeTensorPtr>* outputs);
  Status RunGraph(const RunOptions& options, const std::vector<GeTensorPtr>& inputs, std::vector<GeTensorPtr>* outputs);
  static std::shared_ptr<ge::Session> NewSession(const SessionOptions& sess_options);

 private:
  std::shared_ptr<ge::Session> sess_;
  transform::GraphRunnerOptions options_;
  DfGraphManager& graph_manager_;
};
}  // namespace transform
}  // namespace mindspore

#endif  // TRANSFORM_GRAPH_RUNNER_H_
