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

#include "minddata/dataset/engine/ir/datasetops/build_sentence_piece_vocab_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/build_sentence_piece_vocab_op.h"
#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

BuildSentenceVocabNode::BuildSentenceVocabNode(std::shared_ptr<DatasetNode> child,
                                               std::shared_ptr<SentencePieceVocab> vocab,
                                               const std::vector<std::string> &col_names, int32_t vocab_size,
                                               float character_coverage, SentencePieceModel model_type,
                                               const std::unordered_map<std::string, std::string> &params)
    : vocab_(vocab),
      col_names_(col_names),
      vocab_size_(vocab_size),
      character_coverage_(character_coverage),
      model_type_(model_type),
      params_(params) {
  this->AddChild(child);
}

std::shared_ptr<DatasetNode> BuildSentenceVocabNode::Copy() {
  auto node = std::make_shared<BuildSentenceVocabNode>(nullptr, vocab_, col_names_, vocab_size_, character_coverage_,
                                                       model_type_, params_);
  return node;
}

void BuildSentenceVocabNode::Print(std::ostream &out) const {
  out << Name() + "<vocab>," + "columns:" + PrintColumns(col_names_) + ",vocab_size:" + std::to_string(vocab_size_) +
           ",...)";
}

// Function to build BuildSentenceVocabNode
Status BuildSentenceVocabNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  auto op = std::make_shared<BuildSentencePieceVocabOp>(vocab_, col_names_, vocab_size_, character_coverage_,
                                                        model_type_, params_, connector_que_size_);
  op->set_total_repeats(GetTotalRepeats());
  op->set_num_repeats_per_epoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(op);
  return Status::OK();
}

Status BuildSentenceVocabNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  if (vocab_ == nullptr) {
    std::string err_msg = "BuildSentenceVocabNode: vocab is null.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (vocab_size_ <= 0) {
    std::string err_msg =
      "BuildSentenceVocabNode: vocab_size should be positive, but got: " + std::to_string(vocab_size_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (character_coverage_ < 0.98f || character_coverage_ > 1.0f) {
    std::string err_msg = "BuildSentenceVocabNode: character_coverage should to be between 0.98 and 1.0, but got " +
                          std::to_string(character_coverage_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (!col_names_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("BuildVocabNode", "columns", col_names_));
  }

  return Status::OK();
}

// Visitor accepting method for IRNodePass
Status BuildSentenceVocabNode::Accept(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->Visit(shared_from_base<BuildSentenceVocabNode>(), modified);
}

// Visitor accepting method for IRNodePass
Status BuildSentenceVocabNode::AcceptAfter(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->VisitAfter(shared_from_base<BuildSentenceVocabNode>(), modified);
}
}  // namespace dataset
}  // namespace mindspore
