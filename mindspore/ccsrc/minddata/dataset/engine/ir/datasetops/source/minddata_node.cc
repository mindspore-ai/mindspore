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

#include "minddata/dataset/engine/ir/datasetops/source/minddata_node.h"

#include <map>
#include <memory>
#include <stack>
#include <string>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/mindrecord_op.h"

#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

MindDataNode::MindDataNode(const std::vector<std::string> &dataset_files, const std::vector<std::string> &columns_list,
                           const std::shared_ptr<SamplerObj> &sampler, nlohmann::json padded_sample, int64_t num_padded)
    : MappableSourceNode(),
      dataset_file_(std::string()),
      dataset_files_(dataset_files),
      search_for_pattern_(false),
      columns_list_(columns_list),
      sampler_(sampler),
      padded_sample_(padded_sample),
      sample_bytes_({}),
      num_padded_(num_padded) {}

MindDataNode::MindDataNode(const std::string &dataset_file, const std::vector<std::string> &columns_list,
                           const std::shared_ptr<SamplerObj> &sampler, nlohmann::json padded_sample, int64_t num_padded)
    : MappableSourceNode(),
      dataset_file_(dataset_file),
      dataset_files_({}),
      search_for_pattern_(true),
      columns_list_(columns_list),
      sampler_(sampler),
      padded_sample_(padded_sample),
      sample_bytes_({}),
      num_padded_(num_padded) {}

std::shared_ptr<DatasetNode> MindDataNode::Copy() {
  std::shared_ptr<MindDataNode> node;
  std::shared_ptr<SamplerObj> sampler = sampler_ == nullptr ? nullptr : sampler_->Copy();
  if (dataset_files_.empty()) {
    node = std::make_shared<MindDataNode>(dataset_file_, columns_list_, sampler, padded_sample_, num_padded_);
  } else {
    node = std::make_shared<MindDataNode>(dataset_files_, columns_list_, sampler, padded_sample_, num_padded_);
  }
  return node;
}

void MindDataNode::Print(std::ostream &out) const { out << Name() + "(file:" + dataset_file_ + ",...)"; }

Status MindDataNode::ValidateParams() {
  if (!search_for_pattern_ && dataset_files_.size() > 4096) {
    std::string err_msg =
      "MindDataNode: length of dataset_file must be less than or equal to 4096, dataset_file length: " +
      std::to_string(dataset_file_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  std::vector<std::string> dataset_file_vec =
    search_for_pattern_ ? std::vector<std::string>{dataset_file_} : dataset_files_;
  RETURN_IF_NOT_OK(ValidateDatasetFilesParam("MindDataNode", dataset_file_vec));

  RETURN_IF_NOT_OK(ValidateDatasetSampler("MindDataNode", sampler_));

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
                                                  int64_t num_padded) {
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
    auto sampler_op = std::dynamic_pointer_cast<mindrecord::ShardDistributedSample>(op);
    if (sampler_op && num_padded > 0) {
      sampler_op->SetNumPaddedSamples(num_padded);
      stack_ops.push(sampler_op);
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

std::vector<std::shared_ptr<DatasetOp>> MindDataNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  std::vector<std::shared_ptr<ShardOperator>> operators_;
  build_status = BuildMindDatasetSamplerChain(sampler_, &operators_, num_padded_);
  RETURN_EMPTY_IF_ERROR(build_status);  // remove me after changing return val of Build()

  std::shared_ptr<MindRecordOp> mindrecord_op;
  // If pass a string to MindData(), it will be treated as a pattern to search for matched files,
  // else if pass a vector to MindData(), it will be treated as specified files to be read
  if (search_for_pattern_) {
    std::vector<std::string> dataset_file_vec_ = {dataset_file_};
    mindrecord_op = std::make_shared<MindRecordOp>(num_workers_, rows_per_buffer_, dataset_file_vec_,
                                                   search_for_pattern_, connector_que_size_, columns_list_, operators_,
                                                   num_padded_, padded_sample_, sample_bytes_);
  } else {
    mindrecord_op = std::make_shared<MindRecordOp>(num_workers_, rows_per_buffer_, dataset_files_, search_for_pattern_,
                                                   connector_que_size_, columns_list_, operators_, num_padded_,
                                                   padded_sample_, sample_bytes_);
  }

  build_status = mindrecord_op->Init();  // remove me after changing return val of Build()
  RETURN_EMPTY_IF_ERROR(build_status);
  node_ops.push_back(mindrecord_op);

  return node_ops;
}

// Get the shard id of node
Status MindDataNode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();

  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
