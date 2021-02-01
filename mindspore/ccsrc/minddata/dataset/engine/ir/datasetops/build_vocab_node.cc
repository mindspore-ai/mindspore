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

#include "minddata/dataset/engine/ir/datasetops/build_vocab_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/build_vocab_op.h"
#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

BuildVocabNode::BuildVocabNode(std::shared_ptr<DatasetNode> child, std::shared_ptr<Vocab> vocab,
                               const std::vector<std::string> &columns, const std::pair<int64_t, int64_t> &freq_range,
                               int64_t top_k, const std::vector<std::string> &special_tokens, bool special_first)
    : vocab_(vocab),
      columns_(columns),
      freq_range_(freq_range),
      top_k_(top_k),
      special_tokens_(special_tokens),
      special_first_(special_first) {
  this->AddChild(child);
}

std::shared_ptr<DatasetNode> BuildVocabNode::Copy() {
  auto node =
    std::make_shared<BuildVocabNode>(nullptr, vocab_, columns_, freq_range_, top_k_, special_tokens_, special_first_);
  return node;
}

void BuildVocabNode::Print(std::ostream &out) const {
  out << Name() + "(<vocab>," + "columns:" + PrintColumns(columns_) + ",...)";
}

// Function to build BuildVocabNode
Status BuildVocabNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  std::shared_ptr<BuildVocabOp> build_vocab_op;
  build_vocab_op = std::make_shared<BuildVocabOp>(vocab_, columns_, freq_range_, top_k_, special_tokens_,
                                                  special_first_, num_workers_, connector_que_size_);
  build_vocab_op->set_total_repeats(GetTotalRepeats());
  build_vocab_op->set_num_repeats_per_epoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(build_vocab_op);
  return Status::OK();
}

Status BuildVocabNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  if (vocab_ == nullptr) {
    std::string err_msg = "BuildVocabNode: vocab is null.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (top_k_ <= 0) {
    std::string err_msg = "BuildVocabNode: top_k should be positive, but got: " + std::to_string(top_k_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (freq_range_.first < 0 || freq_range_.second > kDeMaxFreq || freq_range_.first > freq_range_.second) {
    std::string err_msg = "BuildVocabNode: frequency_range [a,b] violates 0 <= a <= b (a,b are inclusive)";
    MS_LOG(ERROR) << "BuildVocabNode: frequency_range [a,b] should be 0 <= a <= b (a,b are inclusive), "
                  << "but got [" << freq_range_.first << ", " << freq_range_.second << "]";
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (!columns_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("BuildVocabNode", "columns", columns_));
  }

  return Status::OK();
}

// Visitor accepting method for IRNodePass
Status BuildVocabNode::Accept(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->Visit(shared_from_base<BuildVocabNode>(), modified);
}

// Visitor accepting method for IRNodePass
Status BuildVocabNode::AcceptAfter(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->VisitAfter(shared_from_base<BuildVocabNode>(), modified);
}
}  // namespace dataset
}  // namespace mindspore
