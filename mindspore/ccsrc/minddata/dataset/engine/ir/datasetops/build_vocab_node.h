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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_BUILD_VOCAB_NODE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_BUILD_VOCAB_NODE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

namespace mindspore {
namespace dataset {

class BuildVocabNode : public DatasetNode {
 public:
  /// \brief Constructor
  BuildVocabNode(std::shared_ptr<DatasetNode> child, std::shared_ptr<Vocab> vocab,
                 const std::vector<std::string> &columns, const std::pair<int64_t, int64_t> &freq_range, int64_t top_k,
                 const std::vector<std::string> &special_tokens, bool special_first);

  /// \brief Destructor
  ~BuildVocabNode() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return The list of shared pointers to the newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return Status Status::OK() if all the parameters are valid
  Status ValidateParams() override;

 private:
  std::shared_ptr<Vocab> vocab_;
  std::vector<std::string> columns_;
  std::pair<int64_t, int64_t> freq_range_;
  int64_t top_k_;
  std::vector<std::string> special_tokens_;
  bool special_first_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_BUILD_VOCAB_NODE_H_
