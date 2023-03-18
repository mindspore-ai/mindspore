/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef DATASET_ENGINE_OPT_POST_AUTO_WORKER_PASS_H_
#define DATASET_ENGINE_OPT_POST_AUTO_WORKER_PASS_H_

#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"
#include "minddata/dataset/engine/opt/pass.h"

namespace mindspore {
namespace dataset {

class AutoWorkerPass : public IRTreePass {
 public:
  // this map will contain weight for the basic pipeline ops. Pipeline op takes up 1 thread but doesn't have workers
  const std::vector<std::map<std::string, float>> kOpWeightConfigs = {
    {{"MappableSource", 8}, {"NonMappableSource", 8}, {kBatchNode, 8}, {kMapNode, 8}},  // config1 leaf:batch:map=1:1:1
    {{"MappableSource", 8}, {"NonMappableSource", 8}, {kBatchNode, 4}, {kMapNode, 4}},  // config2 leaf:batch:map=2:1:1
    {{"MappableSource", 4}, {"NonMappableSource", 4}, {kBatchNode, 8}, {kMapNode, 4}},  // config3 leaf:batch:map=1:2:1
    {{"MappableSource", 4}, {"NonMappableSource", 4}, {kBatchNode, 4}, {kMapNode, 8}},  // config4 leaf:batch:map=1:1:2
    {{"MappableSource", 8}, {"NonMappableSource", 8}, {kBatchNode, 8}, {kMapNode, 4}},  // config5 leaf:batch:map=2:2:1
    {{"MappableSource", 8}, {"NonMappableSource", 8}, {kBatchNode, 4}, {kMapNode, 8}},  // config6 leaf:batch:map=2:1:2
    {{"MappableSource", 4}, {"NonMappableSource", 4}, {kBatchNode, 8}, {kMapNode, 8}},  // config7 leaf:batch:map=1:2:2
  };
  AutoWorkerPass() : min_num_workers_(1), thread_cnt_(GlobalContext::config_manager()->num_cpu_threads()) {}

  /// \brief destructor, by doing "= default", compiler will automatically generate the correct destructor
  ~AutoWorkerPass() override = default;

  Status RunOnTree(std::shared_ptr<DatasetNode> root_ir, bool *const modified) override;

 private:
  class OpWeightPass : public IRNodePass {
   public:
    explicit OpWeightPass(const std::map<std::string, float> &weight_profile)
        : IRNodePass(), weight_sum_(0), weight_profile_(weight_profile) {}

    /// \brief destructor, by doing "= default", compiler will automatically generate the correct destructor
    ~OpWeightPass() override = default;

    // this is the base class function which contains the logic to handle most of the pipeline ops
    // pipeline ops although can't config num_workers it still runs 1 thread they need to be factored into weight
    Status Visit(std::shared_ptr<DatasetNode> node, bool *const modified) override;
    // these functions calculate the weights of more complex Nodes which may depend on its input arg. these functions
    // will also push these nodes to a vector whose num_workers will be set int the Tree Pass
    Status Visit(std::shared_ptr<BatchNode> node, bool *const modified) override;
    Status Visit(std::shared_ptr<MapNode> node, bool *const modified) override;
    Status Visit(std::shared_ptr<MappableSourceNode> node, bool *const modified) override;
    Status Visit(std::shared_ptr<NonMappableSourceNode> node, bool *const modified) override;

    // helper function to look up weight according to the name of this Op.
    float GetNodeWeightFromProfile(std::shared_ptr<DatasetNode> node);

    int32_t weight_sum_;                                 // sum of all weights in the pipeline
    const std::map<std::string, float> weight_profile_;  // key: name of ir node, val: weight of this node
    std::vector<std::pair<std::shared_ptr<DatasetNode>, float>> parallel_ops_;  // first: node second: weight
  };

  const int32_t min_num_workers_;      // minimum number of threads allowed for each op
  const int32_t max_num_workers_ = 8;  // maximum number of threads allowed for each op
  const int32_t thread_cnt_;           // thread cnt of current CPU, obtained through config manager
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_OPT_POST_AUTO_WORKER_PASS_H_
