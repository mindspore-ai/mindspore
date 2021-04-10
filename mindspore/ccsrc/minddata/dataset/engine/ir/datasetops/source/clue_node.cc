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

#include "minddata/dataset/engine/ir/datasetops/source/clue_node.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/clue_op.h"

#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

// Constructor for CLUENode
CLUENode::CLUENode(const std::vector<std::string> clue_files, std::string task, std::string usage, int64_t num_samples,
                   ShuffleMode shuffle, int32_t num_shards, int32_t shard_id, std::shared_ptr<DatasetCache> cache)
    : NonMappableSourceNode(std::move(cache)),
      dataset_files_(clue_files),
      task_(task),
      usage_(usage),
      num_samples_(num_samples),
      shuffle_(shuffle),
      num_shards_(num_shards),
      shard_id_(shard_id) {}

std::shared_ptr<DatasetNode> CLUENode::Copy() {
  auto node =
    std::make_shared<CLUENode>(dataset_files_, task_, usage_, num_samples_, shuffle_, num_shards_, shard_id_, cache_);
  return node;
}

void CLUENode::Print(std::ostream &out) const {
  out << Name() + "(cache:" + ((cache_ != nullptr) ? "true" : "false") + ",..." +
           ",num_shards:" + std::to_string(num_shards_) + ",shard_id:" + std::to_string(shard_id_) + ")";
}

Status CLUENode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetFilesParam("CLUENode", dataset_files_));

  RETURN_IF_NOT_OK(ValidateStringValue("CLUENode", task_, {"AFQMC", "TNEWS", "IFLYTEK", "CMNLI", "WSC", "CSL"}));

  RETURN_IF_NOT_OK(ValidateStringValue("CLUENode", usage_, {"train", "test", "eval"}));

  if (num_samples_ < 0) {
    std::string err_msg = "CLUENode: Invalid number of samples: " + std::to_string(num_samples_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  RETURN_IF_NOT_OK(ValidateDatasetShardParams("CLUENode", num_shards_, shard_id_));

  return Status::OK();
}

// Function to split string based on a character delimiter
std::vector<std::string> CLUENode::split(const std::string &s, char delim) {
  std::vector<std::string> res;
  std::stringstream ss(s);
  std::string item;

  while (getline(ss, item, delim)) {
    res.push_back(item);
  }
  return res;
}

std::map<std::string, std::string> CLUENode::CreateKeyMapForAFQMCOrCMNLITask() {
  std::map<std::string, std::string> key_map;
  if (usage_ == "train" || usage_ == "eval") {
    key_map["label"] = "label";
  } else {  // usage_ == "test"
    key_map["id"] = "id";
  }
  key_map["sentence1"] = "sentence1";
  key_map["sentence2"] = "sentence2";
  return key_map;
}

std::map<std::string, std::string> CLUENode::CreateKeyMapForCSLTask() {
  std::map<std::string, std::string> key_map;
  if (usage_ == "train" || usage_ == "eval") {
    key_map["label"] = "label";
  }
  key_map["id"] = "id";
  key_map["abst"] = "abst";
  key_map["keyword"] = "keyword";
  return key_map;
}

std::map<std::string, std::string> CLUENode::CreateKeyMapForIFLYTEKTask() {
  std::map<std::string, std::string> key_map;
  if (usage_ == "train" || usage_ == "eval") {
    key_map["label"] = "label";
    key_map["label_des"] = "label_des";
  } else {  // usage_ == "test"
    key_map["id"] = "id";
  }
  key_map["sentence"] = "sentence";
  return key_map;
}

std::map<std::string, std::string> CLUENode::CreateKeyMapForTNEWSTask() {
  std::map<std::string, std::string> key_map;
  if (usage_ == "train" || usage_ == "eval") {
    key_map["label"] = "label";
    key_map["label_desc"] = "label_desc";
  } else {  // usage_ == "test"
    key_map["id"] = "id";
  }
  key_map["sentence"] = "sentence";
  key_map["keywords"] = "keywords";
  return key_map;
}

std::map<std::string, std::string> CLUENode::CreateKeyMapForWSCTask() {
  std::map<std::string, std::string> key_map;
  if (usage_ == "train" || usage_ == "eval") {
    key_map["label"] = "label";
  }
  key_map["span1_index"] = "target/span1_index";
  key_map["span2_index"] = "target/span2_index";
  key_map["span1_text"] = "target/span1_text";
  key_map["span2_text"] = "target/span2_text";
  key_map["idx"] = "idx";
  key_map["text"] = "text";
  return key_map;
}

std::map<std::string, std::string> CLUENode::CreateKeyMap() {
  std::map<std::string, std::string> key_map;
  if (task_ == "AFQMC" || task_ == "CMNLI") {
    key_map = CreateKeyMapForAFQMCOrCMNLITask();
  } else if (task_ == "CSL") {
    key_map = CreateKeyMapForCSLTask();
  } else if (task_ == "IFLYTEK") {
    key_map = CreateKeyMapForIFLYTEKTask();
  } else if (task_ == "TNEWS") {
    key_map = CreateKeyMapForTNEWSTask();
  } else if (task_ == "WSC") {
    key_map = CreateKeyMapForWSCTask();
  }
  return key_map;
}

// Function to build CLUENode
Status CLUENode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  auto key_map = CreateKeyMap();
  ColKeyMap ck_map;
  for (auto &p : key_map) {
    ck_map.insert({p.first, split(p.second, '/')});
  }

  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);

  // Sort the dataset files in a lexicographical order
  std::vector<std::string> sorted_dataset_files = dataset_files_;
  std::sort(sorted_dataset_files.begin(), sorted_dataset_files.end());

  std::shared_ptr<ClueOp> clue_op =
    std::make_shared<ClueOp>(num_workers_, rows_per_buffer_, num_samples_, worker_connector_size_, ck_map,
                             sorted_dataset_files, connector_que_size_, shuffle_files, num_shards_, shard_id_);

  RETURN_IF_NOT_OK(clue_op->Init());

  if (cache_ == nullptr && shuffle_ == ShuffleMode::kGlobal && !IsDescendantOfCache()) {
    // Inject ShuffleOp
    std::shared_ptr<DatasetOp> shuffle_op = nullptr;
    int64_t num_rows = 0;

    // First, get the number of rows in the dataset
    RETURN_IF_NOT_OK(ClueOp::CountAllFileRows(sorted_dataset_files, &num_rows));

    // Add the shuffle op after this op
    RETURN_IF_NOT_OK(AddShuffleOp(sorted_dataset_files.size(), num_shards_, num_rows, 0, connector_que_size_,
                                  rows_per_buffer_, &shuffle_op));
    shuffle_op->set_total_repeats(GetTotalRepeats());
    shuffle_op->set_num_repeats_per_epoch(GetNumRepeatsPerEpoch());
    node_ops->push_back(shuffle_op);
  }
  clue_op->set_total_repeats(GetTotalRepeats());
  clue_op->set_num_repeats_per_epoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(clue_op);

  return Status::OK();
}

// Get the shard id of node
Status CLUENode::GetShardId(int32_t *shard_id) {
  *shard_id = shard_id_;

  return Status::OK();
}

// Get Dataset size
Status CLUENode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t num_rows, sample_size;
  RETURN_IF_NOT_OK(ClueOp::CountAllFileRows(dataset_files_, &num_rows));
  sample_size = num_samples_;
  num_rows = static_cast<int64_t>(ceil(num_rows / (1.0 * num_shards_)));
  *dataset_size = sample_size > 0 ? std::min(num_rows, sample_size) : num_rows;
  dataset_size_ = *dataset_size;
  return Status::OK();
}

Status CLUENode::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["num_parallel_workers"] = num_workers_;
  args["dataset_dir"] = dataset_files_;
  args["task"] = task_;
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

// Note: The following two functions are common among NonMappableSourceNode and should be promoted to its parent
// class. CLUE by itself is a non-mappable dataset that does not support sampling. However, if a cache operator is
// injected at some other place higher in the tree, that cache can inherit this sampler from the leaf, providing
// sampling support from the caching layer. That is why we setup the sampler for a leaf node that does not use
// sampling.
Status CLUENode::SetupSamplerForCache(std::shared_ptr<SamplerObj> *sampler) {
  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);
  *sampler = SelectSampler(num_samples_, shuffle_files, num_shards_, shard_id_);
  return Status::OK();
}

// If a cache has been added into the ascendant tree over this clue node, then the cache will be executing
// a sampler for fetching the data.  As such, any options in the clue node need to be reset to its defaults so
// that this clue node will produce the full set of data into the cache.
Status CLUENode::MakeSimpleProducer() {
  shard_id_ = 0;
  num_shards_ = 1;
  shuffle_ = ShuffleMode::kFalse;
  num_samples_ = 0;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
