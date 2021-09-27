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

#include "minddata/dataset/engine/ir/datasetops/source/tf_record_node.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "utils/file_utils.h"
#include "minddata/dataset/engine/datasetops/source/tf_reader_op.h"
#include "minddata/dataset/engine/jagged_connector.h"
#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/util/status.h"
#include "utils/system/crc32c.h"

namespace mindspore {
namespace dataset {

std::shared_ptr<DatasetNode> TFRecordNode::Copy() {
  std::shared_ptr<TFRecordNode> node;
  if (schema_obj_ != nullptr) {
    node = std::make_shared<TFRecordNode>(dataset_files_, schema_obj_, columns_list_, num_samples_, shuffle_,
                                          num_shards_, shard_id_, shard_equal_rows_, cache_);
  } else {
    node = std::make_shared<TFRecordNode>(dataset_files_, schema_path_, columns_list_, num_samples_, shuffle_,
                                          num_shards_, shard_id_, shard_equal_rows_, cache_);
  }
  return node;
}

void TFRecordNode::Print(std::ostream &out) const {
  out << (Name() + "(num_samples:" + std::to_string(num_samples_) + ",num_shards:" + std::to_string(num_shards_) +
          ",shard_id:" + std::to_string(shard_id_) + ",...)");
}

// Validator for TFRecordNode
Status TFRecordNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());

  if (shuffle_ != ShuffleMode::kFalse && shuffle_ != ShuffleMode::kFiles && shuffle_ != ShuffleMode::kGlobal) {
    std::string err_msg = "TFRecordNode: Invalid ShuffleMode, check input value of enum.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (dataset_files_.empty()) {
    std::string err_msg = "TFRecordNode: dataset_files is not specified.";
    MS_LOG(ERROR) << err_msg;
    return Status(StatusCode::kMDSyntaxError, __LINE__, __FILE__, err_msg);
  }

  for (const auto &f : dataset_files_) {
    auto realpath = FileUtils::GetRealPath(f.data());
    CHECK_FAIL_RETURN_UNEXPECTED(realpath.has_value(),
                                 "TFRecordNode: dataset file: [" + f + "] is invalid or does not exist.");
  }

  if (num_samples_ < 0) {
    std::string err_msg = "TFRecordNode: Invalid number of samples: " + std::to_string(num_samples_);
    MS_LOG(ERROR) << err_msg;

    return Status(StatusCode::kMDSyntaxError, __LINE__, __FILE__, err_msg);
  }

  if (num_shards_ <= 0) {
    std::string err_msg = "TFRecordNode: Invalid num_shards: " + std::to_string(num_shards_);
    MS_LOG(ERROR) << err_msg;

    return Status(StatusCode::kMDSyntaxError, __LINE__, __FILE__, err_msg);
  }

  if (shard_id_ < 0 || shard_id_ >= num_shards_) {
    std::string err_msg = "TFRecordNode: Invalid input, shard_id: " + std::to_string(shard_id_) +
                          ", num_shards: " + std::to_string(num_shards_);
    MS_LOG(ERROR) << err_msg;

    return Status(StatusCode::kMDSyntaxError, __LINE__, __FILE__, err_msg);
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
  return err_msg.empty() ? Status::OK() : Status(StatusCode::kMDSyntaxError, __LINE__, __FILE__, err_msg);
}

// Function to build TFRecordNode
Status TFRecordNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  RETURN_UNEXPECTED_IF_NULL(node_ops);
  // Sort the datasets file in a lexicographical order
  std::vector<std::string> sorted_dir_files = dataset_files_;
  std::sort(sorted_dir_files.begin(), sorted_dir_files.end());

  // Create Schema Object
  std::unique_ptr<DataSchema> data_schema = std::make_unique<DataSchema>();
  if (!schema_path_.empty()) {
    RETURN_IF_NOT_OK(data_schema->LoadSchemaFile(schema_path_, columns_list_));
  } else if (schema_obj_ != nullptr) {
    std::string schema_json_string = schema_obj_->to_json();
    RETURN_IF_NOT_OK(data_schema->LoadSchemaString(schema_json_string, columns_list_));
  }

  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);

  // Create and initialize TFReaderOp
  std::shared_ptr<TFReaderOp> tf_reader_op = std::make_shared<TFReaderOp>(
    num_workers_, worker_connector_size_, num_samples_, sorted_dir_files, std::move(data_schema), connector_que_size_,
    columns_list_, shuffle_files, num_shards_, shard_id_, shard_equal_rows_);

  RETURN_IF_NOT_OK(tf_reader_op->Init());

  // If a global shuffle is used for TFRecord, it will inject a shuffle op over the TFRecord.
  // But, if there is a cache in the tree, we do not need the global shuffle and the shuffle op should not be built.
  // This is achieved in the cache transform pass where we call MakeSimpleProducer to reset TFRecord's shuffle
  // option to false.
  if (shuffle_ == ShuffleMode::kGlobal) {
    // Inject ShuffleOp

    std::shared_ptr<DatasetOp> shuffle_op = nullptr;
    int64_t num_rows = 0;

    // First, get the number of rows in the dataset
    RETURN_IF_NOT_OK(TFReaderOp::CountTotalRows(&num_rows, sorted_dir_files));

    // Add the shuffle op after this op
    RETURN_IF_NOT_OK(AddShuffleOp(sorted_dir_files.size(), num_shards_, num_rows, 0, connector_que_size_, &shuffle_op));
    shuffle_op->SetTotalRepeats(GetTotalRepeats());
    shuffle_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
    node_ops->push_back(shuffle_op);
  }
  tf_reader_op->SetTotalRepeats(GetTotalRepeats());
  tf_reader_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  // Add TFReaderOp
  node_ops->push_back(tf_reader_op);
  return Status::OK();
}

// Get the shard id of node
Status TFRecordNode::GetShardId(int32_t *const shard_id) {
  *shard_id = shard_id_;

  return Status::OK();
}

// Get Dataset size
Status TFRecordNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                    int64_t *dataset_size) {
  RETURN_UNEXPECTED_IF_NULL(size_getter);
  RETURN_UNEXPECTED_IF_NULL(dataset_size);
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t num_rows;
  constexpr int64_t kThreadCount = 8;
  // By default, TFRecord will do file-based sharding. But when cache is injected, it will be row-based sharding.
  if (!shard_equal_rows_ && !IsCached()) {
    // Data will be sharded by file
    std::vector<std::string> shard_file_list;
    RETURN_IF_NOT_OK(GetShardFileList(&shard_file_list));
    RETURN_IF_NOT_OK(TFReaderOp::CountTotalRows(&num_rows, shard_file_list, kThreadCount, estimate));
  } else {
    // Data will be sharded by row
    RETURN_IF_NOT_OK(TFReaderOp::CountTotalRows(&num_rows, dataset_files_, kThreadCount, estimate));
    num_rows = static_cast<int64_t>(ceil(num_rows / (num_shards_ * 1.0)));
  }
  *dataset_size = num_samples_ > 0 ? std::min(num_rows, num_samples_) : num_rows;
  dataset_size_ = *dataset_size;
  return Status::OK();
}

// Get the file list of the specific shard ID
Status TFRecordNode::GetShardFileList(std::vector<std::string> *shard_filenames) {
  RETURN_UNEXPECTED_IF_NULL(shard_filenames);
  if (!shard_filenames->empty()) {
    RETURN_STATUS_UNEXPECTED("The initial file list must be empty.");
  }
  for (int index = 0; index < dataset_files_.size(); index++) {
    if (index % num_shards_ == shard_id_) {
      shard_filenames->push_back(dataset_files_.at(index));
    }
  }
  return Status::OK();
}

Status TFRecordNode::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["num_parallel_workers"] = num_workers_;
  args["dataset_files"] = dataset_files_;
  args["columns_list"] = columns_list_;
  args["num_samples"] = num_samples_;
  args["shuffle_global"] = (shuffle_ == ShuffleMode::kGlobal);
  args["shuffle_files"] = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);
  args["shuffle"] = shuffle_;
  args["num_shards"] = num_shards_;
  args["shard_id"] = shard_id_;
  args["shard_equal_rows"] = shard_equal_rows_;
  if (cache_ != nullptr) {
    nlohmann::json cache_args;
    RETURN_IF_NOT_OK(cache_->to_json(&cache_args));
    args["cache"] = cache_args;
  }
  if (schema_obj_ != nullptr) {
    schema_obj_->set_dataset_type("TF");
    schema_obj_->set_num_rows(num_samples_);
    nlohmann::json schema_json_string;
    schema_obj_->schema_to_json(&schema_json_string);
    args["schema_json_string"] = schema_json_string;
  } else {
    args["schema_file_path"] = schema_path_;
  }
  *out_json = args;
  return Status::OK();
}

Status TFRecordNode::from_json(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("num_parallel_workers") != json_obj.end(),
                               "Failed to find num_parallel_workers");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("dataset_files") != json_obj.end(), "Failed to find dataset_files");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("columns_list") != json_obj.end(), "Failed to find columns_list");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("num_samples") != json_obj.end(), "Failed to find num_samples");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("shuffle") != json_obj.end(), "Failed to find shuffle");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("num_shards") != json_obj.end(), "Failed to find num_shards");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("shard_id") != json_obj.end(), "Failed to find shard_id");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("shard_equal_rows") != json_obj.end(), "Failed to find shard_equal_rows");
  std::vector<std::string> dataset_files = json_obj["dataset_files"];
  std::vector<std::string> columns_list = json_obj["columns_list"];
  int64_t num_samples = json_obj["num_samples"];
  ShuffleMode shuffle = static_cast<ShuffleMode>(json_obj["shuffle"]);
  int32_t num_shards = json_obj["num_shards"];
  int32_t shard_id = json_obj["shard_id"];
  bool shard_equal_rows = json_obj["shard_equal_rows"];
  std::shared_ptr<DatasetCache> cache = nullptr;
  RETURN_IF_NOT_OK(DatasetCache::from_json(json_obj, &cache));
  if (json_obj.find("schema_file_path") != json_obj.end()) {
    std::string schema_file_path = json_obj["schema_file_path"];
    *ds = std::make_shared<TFRecordNode>(dataset_files, schema_file_path, columns_list, num_samples, shuffle,
                                         num_shards, shard_id, shard_equal_rows, cache);
  } else {
    CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("schema_json_string") != json_obj.end(),
                                 "Failed to find either schema_file_path or schema_json_string");
    std::shared_ptr<SchemaObj> schema_obj = Schema();
    RETURN_IF_NOT_OK(schema_obj->from_json(json_obj["schema_json_string"]));
    *ds = std::make_shared<TFRecordNode>(dataset_files, schema_obj, columns_list, num_samples, shuffle, num_shards,
                                         shard_id, shard_equal_rows, cache);
  }
  (*ds)->SetNumWorkers(json_obj["num_parallel_workers"]);
  return Status::OK();
}

// Note: The following two functions are common among NonMappableSourceNode and should be promoted to its parent class.
// TFRecord by itself is a non-mappable dataset that does not support sampling.
// However, if a cache operator is injected at some other place higher in the tree, that cache can
// inherit this sampler from the leaf, providing sampling support from the caching layer.
// That is why we setup the sampler for a leaf node that does not use sampling.
Status TFRecordNode::SetupSamplerForCache(std::shared_ptr<SamplerObj> *sampler) {
  RETURN_UNEXPECTED_IF_NULL(sampler);
  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);
  *sampler = SelectSampler(num_samples_, shuffle_files, num_shards_, shard_id_);
  return Status::OK();
}

// If a cache has been added into the ascendant tree over this TFRecord node, then the cache will be executing
// a sampler for fetching the data.  As such, any options in the TFRecord node need to be reset to its defaults so
// that this TFRecord node will produce the full set of data into the cache.
Status TFRecordNode::MakeSimpleProducer() {
  shard_id_ = 0;
  num_shards_ = 1;
  shuffle_ = ShuffleMode::kFalse;
  num_samples_ = 0;
  shard_equal_rows_ = false;
  return Status::OK();
}

// Visitor accepting method for IRNodePass
Status TFRecordNode::Accept(IRNodePass *p, bool *const modified) {
  RETURN_UNEXPECTED_IF_NULL(p);
  RETURN_UNEXPECTED_IF_NULL(modified);
  // Downcast shared pointer then call visitor
  return p->Visit(shared_from_base<TFRecordNode>(), modified);
}

// Visitor accepting method for IRNodePass
Status TFRecordNode::AcceptAfter(IRNodePass *const p, bool *const modified) {
  RETURN_UNEXPECTED_IF_NULL(p);
  RETURN_UNEXPECTED_IF_NULL(modified);
  // Downcast shared pointer then call visitor
  return p->VisitAfter(shared_from_base<TFRecordNode>(), modified);
}
}  // namespace dataset
}  // namespace mindspore
