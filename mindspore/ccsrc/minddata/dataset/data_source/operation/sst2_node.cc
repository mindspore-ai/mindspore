/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/data_source/operation/sst2_node.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
SST2Node::SST2Node(const std::string &dataset_dir, const std::string &usage, int64_t num_samples, ShuffleMode shuffle,
                   int32_t num_shards, int32_t shard_id, std::shared_ptr<DatasetCache> cache)
    : NonMappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      usage_(usage),
      num_samples_(num_samples),
      shuffle_(shuffle),
      num_shards_(num_shards),
      shard_id_(shard_id) {
  // Update the num_shards_ in global context. this number is only used for now by auto_num_worker_pass. User discretion
  // is advised. Auto_num_worker_pass is currently an experimental feature which can still work if the num_shards_ isn't
  // 100% correct. The reason behind is for now, PreBuildSampler doesn't offer a way to return num_shards. Once
  // PreBuildSampler is phased out, this can be cleaned up.
  GlobalContext::config_manager()->set_num_shards_for_auto_num_workers(num_shards_);
}

std::shared_ptr<DatasetNode> SST2Node::Copy() {
  auto node = std::make_shared<SST2Node>(dataset_dir_, usage_, num_samples_, shuffle_, num_shards_, shard_id_, cache_);
  (void)node->SetNumWorkers(num_workers_);
  (void)node->SetConnectorQueueSize(connector_que_size_);
  return node;
}

void SST2Node::Print(std::ostream &out) const {
  out << (Name() + "(cache: " + ((cache_ != nullptr) ? "true" : "false") +
          ", num_shards: " + std::to_string(num_shards_) + ", shard_id: " + std::to_string(shard_id_) + ")");
}

Status SST2Node::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("SST2Node", dataset_dir_));
  RETURN_IF_NOT_OK(ValidateStringValue("SST2Node", usage_, {"dev", "train", "test"}));
  RETURN_IF_NOT_OK(ValidateScalar("SST2Node", "num_samples", num_samples_, {0}, false));
  RETURN_IF_NOT_OK(ValidateDatasetShardParams("SST2Node", num_shards_, shard_id_));

  return Status::OK();
}

// Function to build SST2Node
Status SST2Node::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  RETURN_UNEXPECTED_IF_NULL(node_ops);
  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);

  // Sort the dataset files in a lexicographical order
  std::vector<std::string> sorted_dataset_files;
  RETURN_IF_NOT_OK(WalkAllFiles(dataset_dir_, usage_, &sorted_dataset_files));
  std::sort(sorted_dataset_files.begin(), sorted_dataset_files.end());

  char field_delim = '\t';

  std::vector<std::string> column_names;

  std::vector<std::shared_ptr<CsvOp::BaseRecord>> column_default_list;

  std::shared_ptr<SST2Op> sst2_op = std::make_shared<SST2Op>(
    sorted_dataset_files, usage_, field_delim, column_default_list, column_names, num_workers_, num_samples_,
    worker_connector_size_, connector_que_size_, shuffle_files, num_shards_, shard_id_);

  RETURN_IF_NOT_OK(sst2_op->Init());

  // If a global shuffle is used for SST2, it will inject a shuffle op over the SST2.
  // But, if there is a cache in the tree, we do not need the global shuffle and the shuffle op should not be built.
  // This is achieved in the cache transform pass where we call MakeSimpleProducer to reset SST2's shuffle
  // option to false.
  if (shuffle_ == ShuffleMode::kGlobal) {
    // Inject ShuffleOp
    std::shared_ptr<ShuffleOp> shuffle_op = nullptr;
    int64_t num_rows = 0;

    // First, get the number of rows in the dataset
    RETURN_IF_NOT_OK(SST2Op::CountAllFileRows(sorted_dataset_files, column_names.empty(), &num_rows));

    // Add the shuffle op after this op
    RETURN_IF_NOT_OK(
      AddShuffleOp(sorted_dataset_files.size(), num_shards_, num_rows, 0, connector_que_size_, &shuffle_op));
    shuffle_op->SetTotalRepeats(GetTotalRepeats());
    shuffle_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
    shuffle_op->Skip(skip_steps_);
    node_ops->push_back(shuffle_op);
  }
  sst2_op->SetTotalRepeats(GetTotalRepeats());
  sst2_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(sst2_op);

  return Status::OK();
}

Status SST2Node::WalkAllFiles(const std::string &dataset_dir, const std::string &usage,
                              std::vector<std::string> *dataset_files) {
  RETURN_UNEXPECTED_IF_NULL(dataset_files);
  Path train_file_name("train.tsv");
  Path test_file_name("test.tsv");
  Path dev_file_name("dev.tsv");
  Path dir(dataset_dir);
  if (usage == "train") {
    Path file_path = dir / train_file_name;
    dataset_files->push_back(file_path.ToString());
  } else if (usage == "test") {
    Path file_path = dir / test_file_name;
    dataset_files->push_back(file_path.ToString());
  } else if (usage == "dev") {
    Path file_path = dir / dev_file_name;
    dataset_files->push_back(file_path.ToString());
  }
  return Status::OK();
}

// Get the shard id of node
Status SST2Node::GetShardId(int32_t *shard_id) {
  RETURN_UNEXPECTED_IF_NULL(shard_id);
  *shard_id = shard_id_;
  return Status::OK();
}

// Get Dataset size
Status SST2Node::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                int64_t *dataset_size) {
  RETURN_UNEXPECTED_IF_NULL(dataset_size);
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t num_rows;
  int64_t sample_size;
  std::vector<std::string> dataset_files;

  RETURN_IF_NOT_OK(WalkAllFiles(dataset_dir_, usage_, &dataset_files));
  RETURN_IF_NOT_OK(SST2Op::CountAllFileRows(dataset_files, true, &num_rows));
  sample_size = num_samples_;
  num_rows = static_cast<int64_t>(ceil(num_rows / (1.0 * num_shards_)));
  *dataset_size = sample_size > 0 ? std::min(num_rows, sample_size) : num_rows;
  dataset_size_ = *dataset_size;
  return Status::OK();
}

Status SST2Node::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["num_parallel_workers"] = num_workers_;
  args["connector_queue_size"] = connector_que_size_;
  args["dataset_dir"] = dataset_dir_;
  args["usage"] = usage_;
  args["num_samples"] = num_samples_;
  args["shuffle"] = shuffle_;
  args["num_shards"] = num_shards_;
  args["shard_id"] = shard_id_;
  if (cache_ != nullptr) {
    nlohmann::json cache_args;
    RETURN_IF_NOT_OK(cache_->to_json(&cache_args));
    args["cache"] = cache_args;
  }
  *out_json = args;
  return Status::OK();
}

Status SST2Node::from_json(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  RETURN_UNEXPECTED_IF_NULL(ds);
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "num_parallel_workers", kSST2Node));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "connector_queue_size", kSST2Node));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "dataset_dir", kSST2Node));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "usage", kSST2Node));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "num_samples", kSST2Node));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "shuffle", kSST2Node));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "num_shards", kSST2Node));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "shard_id", kSST2Node));
  std::string dataset_dir = json_obj["dataset_dir"];
  std::string usage = json_obj["usage"];
  int64_t num_samples = json_obj["num_samples"];
  ShuffleMode shuffle = static_cast<ShuffleMode>(json_obj["shuffle"]);
  int32_t num_shards = json_obj["num_shards"];
  int32_t shard_id = json_obj["shard_id"];
  std::shared_ptr<DatasetCache> cache;
  RETURN_IF_NOT_OK(DatasetCache::from_json(json_obj, &cache));
  *ds = std::make_shared<SST2Node>(dataset_dir, usage, num_samples, shuffle, num_shards, shard_id, cache);
  (void)((*ds)->SetNumWorkers(json_obj["num_parallel_workers"]));
  (void)((*ds)->SetConnectorQueueSize(json_obj["connector_queue_size"]));
  return Status::OK();
}

// Note: The following two functions are common among NonMappableSourceNode and should be promoted to its parent class.
// SST2 by itself is a non-mappable dataset that does not support sampling.
// However, if a cache operator is injected at some other place higher in the tree, that cache can
// inherit this sampler from the leaf, providing sampling support from the caching layer.
// That is why we setup the sampler for a leaf node that does not use sampling.
Status SST2Node::SetupSamplerForCache(std::shared_ptr<SamplerObj> *sampler) {
  *sampler = SelectSampler(num_samples_, shuffle_, num_shards_, shard_id_);
  return Status::OK();
}

// If a cache has been added into the ascendant tree over this SST2 node, then the cache will be executing
// a sampler for fetching the data.  As such, any options in the SST2 node need to be reset to its defaults so
// that this SST2 node will produce the full set of data into the cache.
Status SST2Node::MakeSimpleProducer() {
  shard_id_ = 0;
  num_shards_ = 1;
  shuffle_ = ShuffleMode::kFalse;
  num_samples_ = 0;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
