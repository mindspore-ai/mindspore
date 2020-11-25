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

#include "minddata/dataset/engine/ir/datasetops/source/tf_record_node.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/tf_reader_op.h"
#include "minddata/dataset/engine/jagged_connector.h"
#include "minddata/dataset/util/status.h"
#include "utils/system/crc32c.h"

namespace mindspore {
namespace dataset {

// Validator for TFRecordNode
Status TFRecordNode::ValidateParams() {
  if (dataset_files_.empty()) {
    std::string err_msg = "TFRecordNode: dataset_files is not specified.";
    MS_LOG(ERROR) << err_msg;
    return Status(StatusCode::kSyntaxError, __LINE__, __FILE__, err_msg);
  }

  for (const auto &f : dataset_files_) {
    Path dataset_file(f);
    if (!dataset_file.Exists()) {
      std::string err_msg = "TFRecordNode: dataset file: [" + f + "] is invalid or does not exist.";
      MS_LOG(ERROR) << err_msg;

      return Status(StatusCode::kSyntaxError, __LINE__, __FILE__, err_msg);
    }
  }

  if (num_samples_ < 0) {
    std::string err_msg = "TFRecordNode: Invalid number of samples: " + std::to_string(num_samples_);
    MS_LOG(ERROR) << err_msg;

    return Status(StatusCode::kSyntaxError, __LINE__, __FILE__, err_msg);
  }

  if (num_shards_ <= 0) {
    std::string err_msg = "TFRecordNode: Invalid num_shards: " + std::to_string(num_shards_);
    MS_LOG(ERROR) << err_msg;

    return Status(StatusCode::kSyntaxError, __LINE__, __FILE__, err_msg);
  }

  if (shard_id_ < 0 || shard_id_ >= num_shards_) {
    std::string err_msg = "TFRecordNode: Invalid input, shard_id: " + std::to_string(shard_id_) +
                          ", num_shards: " + std::to_string(num_shards_);
    MS_LOG(ERROR) << err_msg;

    return Status(StatusCode::kSyntaxError, __LINE__, __FILE__, err_msg);
  }

  std::vector<std::string> invalid_files(dataset_files_.size());
  auto it = std::copy_if(dataset_files_.begin(), dataset_files_.end(), invalid_files.begin(),
                         [](const std::string &filename) { return !TFReaderOp::ValidateFirstRowCrc(filename); });
  invalid_files.resize(std::distance(invalid_files.begin(), it));
  std::string err_msg;
  if (!invalid_files.empty()) {
    err_msg += "Invalid file, the following files either cannot be opened, or are not valid tfrecord files:\n";

    std::string accumulated_filenames = std::accumulate(
      invalid_files.begin(), invalid_files.end(), std::string(""),
      [](const std::string &accumulated, const std::string &next) { return accumulated + "    " + next + "\n"; });
    err_msg += accumulated_filenames;
  }
  return err_msg.empty() ? Status::OK() : Status(StatusCode::kSyntaxError, __LINE__, __FILE__, err_msg);
}

// Function to build TFRecordNode
std::vector<std::shared_ptr<DatasetOp>> TFRecordNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  // Sort the datasets file in a lexicographical order
  std::vector<std::string> sorted_dir_files = dataset_files_;
  std::sort(sorted_dir_files.begin(), sorted_dir_files.end());

  // Create Schema Object
  std::unique_ptr<DataSchema> data_schema = std::make_unique<DataSchema>();
  if (!schema_path_.empty()) {
    RETURN_EMPTY_IF_ERROR(data_schema->LoadSchemaFile(schema_path_, columns_list_));
  } else if (schema_obj_ != nullptr) {
    std::string schema_json_string = schema_obj_->to_json();
    RETURN_EMPTY_IF_ERROR(data_schema->LoadSchemaString(schema_json_string, columns_list_));
  }

  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);

  // TFReaderOp by itself is a non-mappable dataset that does not support sampling.
  // However, if a cache operator is injected at some other place higher in the tree, that cache can
  // inherit this sampler from the leaf, providing sampling support from the caching layer.
  // That is why we save the sampler here in a leaf node that does not use sampling.
  std::shared_ptr<SamplerObj> sampler_ = SelectSampler(num_samples_, shuffle_files, num_shards_, shard_id_);

  // Create and initialize TFReaderOp
  std::shared_ptr<TFReaderOp> tf_reader_op =
    std::make_shared<TFReaderOp>(num_workers_, worker_connector_size_, rows_per_buffer_, num_samples_, sorted_dir_files,
                                 std::move(data_schema), connector_que_size_, columns_list_, shuffle_files, num_shards_,
                                 shard_id_, shard_equal_rows_, std::move(sampler_->Build()));

  build_status = tf_reader_op->Init();  // remove me after changing return val of Build()
  RETURN_EMPTY_IF_ERROR(build_status);

  if (cache_ == nullptr && shuffle_ == ShuffleMode::kGlobal) {
    // Inject ShuffleOp

    std::shared_ptr<DatasetOp> shuffle_op = nullptr;
    int64_t num_rows = 0;

    // First, get the number of rows in the dataset
    build_status = TFReaderOp::CountTotalRows(&num_rows, sorted_dir_files);
    RETURN_EMPTY_IF_ERROR(build_status);  // remove me after changing return val of Build()

    // Add the shuffle op after this op
    build_status = AddShuffleOp(sorted_dir_files.size(), num_shards_, num_rows, 0, connector_que_size_,
                                rows_per_buffer_, &shuffle_op);
    RETURN_EMPTY_IF_ERROR(build_status);  // remove me after changing return val of Build()
    node_ops.push_back(shuffle_op);
  }
  build_status = AddCacheOp(&node_ops);  // remove me after changing return val of Build()
  RETURN_EMPTY_IF_ERROR(build_status);

  // Add TFReaderOp
  node_ops.push_back(tf_reader_op);
  return node_ops;
}

// Get the shard id of node
Status TFRecordNode::GetShardId(int32_t *shard_id) {
  *shard_id = shard_id_;

  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
