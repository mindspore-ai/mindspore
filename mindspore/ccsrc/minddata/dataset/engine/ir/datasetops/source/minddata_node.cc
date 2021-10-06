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

#include "minddata/dataset/engine/ir/datasetops/source/minddata_node.h"

#include "minddata/dataset/engine/datasetops/source/mindrecord_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/mind_record_sampler.h"
#include "minddata/dataset/engine/ir/datasetops/cache_lookup_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/mindrecord_sampler_ir.h"
#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

MindDataNode::MindDataNode(const std::vector<std::string> &dataset_files, const std::vector<std::string> &columns_list,
                           const std::shared_ptr<SamplerObj> &sampler, nlohmann::json padded_sample, int64_t num_padded,
                           ShuffleMode shuffle_mode, std::shared_ptr<DatasetCache> cache)
    : MappableSourceNode(std::move(cache)),
      dataset_file_(std::string()),
      dataset_files_(dataset_files),
      search_for_pattern_(false),
      columns_list_(columns_list),
      input_sampler_(sampler),
      sampler_(std::make_shared<MindRecordSamplerObj>()),
      padded_sample_(padded_sample),
      sample_bytes_({}),
      num_padded_(num_padded),
      shuffle_mode_(shuffle_mode) {}

MindDataNode::MindDataNode(const std::string &dataset_file, const std::vector<std::string> &columns_list,
                           const std::shared_ptr<SamplerObj> &sampler, nlohmann::json padded_sample, int64_t num_padded,
                           ShuffleMode shuffle_mode, std::shared_ptr<DatasetCache> cache)
    : MappableSourceNode(std::move(cache)),
      dataset_file_(dataset_file),
      dataset_files_({}),
      search_for_pattern_(true),
      columns_list_(columns_list),
      input_sampler_(sampler),
      sampler_(std::make_shared<MindRecordSamplerObj>()),
      padded_sample_(padded_sample),
      sample_bytes_({}),
      num_padded_(num_padded),
      shuffle_mode_(shuffle_mode) {}

std::shared_ptr<DatasetNode> MindDataNode::Copy() {
  std::shared_ptr<MindDataNode> node;
  std::shared_ptr<SamplerObj> sampler = (input_sampler_ == nullptr) ? nullptr : input_sampler_->SamplerCopy();
  if (dataset_files_.empty()) {
    node = std::make_shared<MindDataNode>(dataset_file_, columns_list_, sampler, padded_sample_, num_padded_,
                                          shuffle_mode_, cache_);
  } else {
    node = std::make_shared<MindDataNode>(dataset_files_, columns_list_, sampler, padded_sample_, num_padded_,
                                          shuffle_mode_, cache_);
  }
  node->SetSampleBytes(&sample_bytes_);
  return node;
}

void MindDataNode::Print(std::ostream &out) const { out << (Name() + "(file:" + dataset_file_ + ",...)"); }

Status MindDataNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  constexpr size_t max_len = 4096;
  if (!search_for_pattern_ && dataset_files_.size() > max_len) {
    std::string err_msg =
      "MindDataNode: length of dataset_file must be less than or equal to 4096, dataset_file length: " +
      std::to_string(dataset_file_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (shuffle_mode_ != ShuffleMode::kFalse && shuffle_mode_ != ShuffleMode::kFiles &&
      shuffle_mode_ != ShuffleMode::kGlobal && shuffle_mode_ != ShuffleMode::kInfile) {
    std::string err_msg = "TFRecordNode: Invalid ShuffleMode, check input value of enum.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  std::vector<std::string> dataset_file_vec =
    search_for_pattern_ ? std::vector<std::string>{dataset_file_} : dataset_files_;
  RETURN_IF_NOT_OK(ValidateDatasetFilesParam("MindDataNode", dataset_file_vec));

  RETURN_IF_NOT_OK(ValidateDatasetSampler("MindDataNode", input_sampler_));

  if (!columns_list_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("MindDataNode", "columns_list", columns_list_));
  }

  if (padded_sample_ != nullptr) {
    if (num_padded_ < 0 || num_padded_ > INT_MAX) {
      std::string err_msg =
        "MindDataNode: num_padded must to be between 0 and INT32_MAX, but got: " + std::to_string(num_padded_);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
    if (columns_list_.empty()) {
      std::string err_msg = "MindDataNode: padded_sample is specified and requires columns_list as well";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
    for (std::string &column : columns_list_) {
      if (padded_sample_.find(column) == padded_sample_.end()) {
        std::string err_msg = "MindDataNode: " + column + " in columns_list does not match any column in padded_sample";
        MS_LOG(ERROR) << err_msg << ", padded_sample: " << padded_sample_;
        RETURN_STATUS_SYNTAX_ERROR(err_msg);
      }
    }
  }
  if (num_padded_ > 0) {
    if (padded_sample_ == nullptr) {
      std::string err_msg = "MindDataNode: num_padded is specified but padded_sample is not";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }

  return Status::OK();
}

// Helper function to create runtime sampler for minddata dataset
Status MindDataNode::BuildMindDatasetSamplerChain(const std::shared_ptr<SamplerObj> &sampler,
                                                  std::vector<std::shared_ptr<mindrecord::ShardOperator>> *operators_,
                                                  int64_t num_padded, ShuffleMode shuffle_mode) {
  std::shared_ptr<mindrecord::ShardOperator> op = sampler->BuildForMindDataset();
  if (op == nullptr) {
    std::string err_msg =
      "MindDataNode: Unsupported sampler is supplied for MindDataset. Supported sampler list: "
      "SubsetRandomSampler, PkSampler, RandomSampler, SequentialSampler and DistributedSampler";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  std::stack<std::shared_ptr<mindrecord::ShardOperator>> stack_ops;
  while (op != nullptr) {
    // update the shuffle mode for sampler op or shuffle op
    if (shuffle_mode != ShuffleMode::kFalse) {
      op->UpdateShuffleMode(shuffle_mode);
    }

    auto distributed_sampler_op = std::dynamic_pointer_cast<mindrecord::ShardDistributedSample>(op);
    if (distributed_sampler_op && num_padded > 0) {
      distributed_sampler_op->SetNumPaddedSamples(num_padded);
      stack_ops.push(distributed_sampler_op);
    } else {
      stack_ops.push(op);
    }
    op = op->GetChildOp();
  }
  while (!stack_ops.empty()) {
    operators_->push_back(stack_ops.top());
    stack_ops.pop();
  }
  return Status::OK();
}

// Helper function to set sample_bytes from py::byte type
void MindDataNode::SetSampleBytes(std::map<std::string, std::string> *sample_bytes) { sample_bytes_ = *sample_bytes; }

Status MindDataNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  RETURN_IF_NOT_OK(BuildMindDatasetSamplerChain(input_sampler_, &operators_, num_padded_, shuffle_mode_));

  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  // Build the sampler IR into a runtime sampler.
  // This will also create a shard reader object, saved in this node's sampler_.
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));

  // Now we need to acquire the newly created shard reader from this node's sampler_.
  // There are two cases:
  // 1. If this node is cached, now after cache transform pass, its sampler_ has already been replaced by cache lookup
  // node, and we should find the shard reader from cache lookup node's sampler_.
  // 2. If this node is not cached, just acquire the shard reader from this node's sampler_.
  std::unique_ptr<ShardReader> shard_reader;
  if (IsDescendantOfCache()) {
    auto cache_lookup_sampler = std::dynamic_pointer_cast<CacheLookupNode>(sampler_);
    CHECK_FAIL_RETURN_UNEXPECTED(cache_lookup_sampler != nullptr,
                                 "Internal error. MindDataNode is cached, its sampler should be cache lookup node");
    auto mr_sampler = std::dynamic_pointer_cast<MindRecordSamplerObj>(cache_lookup_sampler->Sampler());
    CHECK_FAIL_RETURN_UNEXPECTED(mr_sampler != nullptr,
                                 "Internal error. CacheLookupNode's sampler should be a MindRecordSamplerObj object");
    RETURN_IF_NOT_OK(mr_sampler->GetShardReader(&shard_reader));
  } else {
    auto mr_sampler = std::dynamic_pointer_cast<MindRecordSamplerObj>(sampler_);
    CHECK_FAIL_RETURN_UNEXPECTED(mr_sampler != nullptr,
                                 "Internal error. MindDataNode's sampler should be a MindRecordSamplerObj object");
    RETURN_IF_NOT_OK(mr_sampler->GetShardReader(&shard_reader));
  }

  std::shared_ptr<MindRecordOp> mindrecord_op;
  // If pass a string to MindData(), it will be treated as a pattern to search for matched files,
  // else if pass a vector to MindData(), it will be treated as specified files to be read
  if (search_for_pattern_) {
    std::vector<std::string> dataset_file_vec_ = {dataset_file_};
    mindrecord_op = std::make_shared<MindRecordOp>(
      num_workers_, dataset_file_vec_, search_for_pattern_, connector_que_size_, columns_list_, operators_, num_padded_,
      padded_sample_, sample_bytes_, shuffle_mode_, std::move(shard_reader), std::move(sampler_rt));
  } else {
    mindrecord_op = std::make_shared<MindRecordOp>(
      num_workers_, dataset_files_, search_for_pattern_, connector_que_size_, columns_list_, operators_, num_padded_,
      padded_sample_, sample_bytes_, shuffle_mode_, std::move(shard_reader), std::move(sampler_rt));
  }

  RETURN_IF_NOT_OK(mindrecord_op->Init());
  mindrecord_op->SetTotalRepeats(GetTotalRepeats());
  mindrecord_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(mindrecord_op);

  return Status::OK();
}

// Get the shard id of node
Status MindDataNode::GetShardId(int32_t *shard_id) {
  *shard_id = input_sampler_->ShardId();

  return Status::OK();
}

// Get Dataset size
Status MindDataNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                    int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t num_rows = -1;
  std::vector<std::shared_ptr<ShardOperator>> operators;
  RETURN_IF_NOT_OK(BuildMindDatasetSamplerChain(input_sampler_, &operators, num_padded_, shuffle_mode_));

  if (search_for_pattern_) {
    dataset_files_ = {dataset_file_};
  }

  // The last operator is parent sampler
  std::shared_ptr<ShardOperator> op = operators.back();
  RETURN_IF_NOT_OK(MindRecordOp::CountTotalRows(dataset_files_, search_for_pattern_, op, &num_rows, num_padded_));
  *dataset_size = num_rows;
  dataset_size_ = *dataset_size;
  return Status::OK();
}

// Visitor accepting method for IRNodePass
Status MindDataNode::Accept(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->Visit(shared_from_base<MindDataNode>(), modified);
}

// Visitor accepting method for IRNodePass
Status MindDataNode::AcceptAfter(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->VisitAfter(shared_from_base<MindDataNode>(), modified);
}
}  // namespace dataset
}  // namespace mindspore
