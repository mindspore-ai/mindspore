/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/data_source/operation/wiki_text_node.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/data_source/wiki_text_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// Constructor for WikiTextNode.
WikiTextNode::WikiTextNode(const std::string &dataset_dir, const std::string &usage, int64_t num_samples,
                           ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                           const std::shared_ptr<DatasetCache> &cache)
    : NonMappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      usage_(usage),
      num_samples_(num_samples),
      shuffle_(shuffle),
      num_shards_(num_shards),
      shard_id_(shard_id),
      wikitext_files_list_(WalkAllFiles(usage, dataset_dir)) {
  // Update the num_shards_ in global context. this number is only used for now by auto_num_worker_pass. User discretion
  // is advised. Auto_num_worker_pass is currently an experimental feature which can still work if the num_shards_ isn't
  // 100% correct. The reason behind is for now, PreBuildSampler doesn't offer a way to return num_shards. Once
  // PreBuildSampler is phased out, this can be cleaned up.
  GlobalContext::config_manager()->set_num_shards_for_auto_num_workers(num_shards_);
}

std::shared_ptr<DatasetNode> WikiTextNode::Copy() {
  auto node =
    std::make_shared<WikiTextNode>(dataset_dir_, usage_, num_samples_, shuffle_, num_shards_, shard_id_, cache_);
  (void)node->SetNumWorkers(num_workers_);
  (void)node->SetConnectorQueueSize(connector_que_size_);
  return node;
}

void WikiTextNode::Print(std::ostream &out) const {
  out << (Name() + "(cache: " + ((cache_ != nullptr) ? "true" : "false") +
          ", num_shards: " + std::to_string(num_shards_) + ", shard_id: " + std::to_string(shard_id_) + ")");
}

Status WikiTextNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("WikiTextDataset", dataset_dir_));
  RETURN_IF_NOT_OK(ValidateStringValue("WikiTextDataset", usage_, {"train", "test", "valid", "all"}));
  RETURN_IF_NOT_OK(ValidateScalar("WikiTextDataset", "num_samples", num_samples_, {0}, false));
  RETURN_IF_NOT_OK(ValidateEnum("WikiTextDataset", "ShuffleMode", shuffle_,
                                {ShuffleMode::kFalse, ShuffleMode::kFiles, ShuffleMode::kGlobal}));
  RETURN_IF_NOT_OK(ValidateDatasetShardParams("WikiTextDataset", num_shards_, shard_id_));
  return Status::OK();
}

// Function to build WikiTextNode.
Status WikiTextNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);
  // Sort the dataset files in a lexicographical order.
  std::vector<std::string> sorted_dataset_files = wikitext_files_list_;
  std::sort(sorted_dataset_files.begin(), sorted_dataset_files.end());
  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("text", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  // Create and initialize WikiTextNode.
  std::shared_ptr<WikiTextOp> wikitext_op =
    std::make_shared<WikiTextOp>(num_workers_, num_samples_, worker_connector_size_, std::move(schema),
                                 sorted_dataset_files, connector_que_size_, shuffle_files, num_shards_, shard_id_);
  RETURN_IF_NOT_OK(wikitext_op->Init());
  // If a global shuffle is used for WikiText, it will inject a shuffle op over the WikiText.
  // But, if there is a cache in the tree, we do not need the global shuffle and the shuffle op should not be built.
  // This is achieved in the cache transform pass where we call MakeSimpleProducer to reset WikiText's shuffle
  // option to false.
  if (shuffle_ == ShuffleMode::kGlobal) {
    // Inject ShuffleOp.
    std::shared_ptr<ShuffleOp> shuffle_op = nullptr;
    int64_t num_rows = 0;
    // First, get the number of rows in the dataset.
    RETURN_IF_NOT_OK(WikiTextOp::CountAllFileRows(wikitext_files_list_, &num_rows));
    // Add the shuffle op after this op.
    RETURN_IF_NOT_OK(
      AddShuffleOp(sorted_dataset_files.size(), num_shards_, num_rows, 0, connector_que_size_, &shuffle_op));
    shuffle_op->SetTotalRepeats(GetTotalRepeats());
    shuffle_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
    shuffle_op->Skip(skip_steps_);
    node_ops->push_back(shuffle_op);
  }
  wikitext_op->SetTotalRepeats(GetTotalRepeats());
  wikitext_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  // Add WikiTextNode.
  node_ops->push_back(wikitext_op);
  return Status::OK();
}

// Get the shard id of node.
Status WikiTextNode::GetShardId(int32_t *shard_id) {
  *shard_id = shard_id_;
  return Status::OK();
}

// Get Dataset size.
Status WikiTextNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                    int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t num_rows, sample_size = num_samples_;
  RETURN_IF_NOT_OK(WikiTextOp::CountAllFileRows(wikitext_files_list_, &num_rows));
  num_rows = static_cast<int64_t>(ceil(num_rows / (1.0 * num_shards_)));
  *dataset_size = sample_size > 0 ? std::min(num_rows, sample_size) : num_rows;
  dataset_size_ = *dataset_size;
  return Status::OK();
}

Status WikiTextNode::to_json(nlohmann::json *out_json) {
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

Status WikiTextNode::from_json(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  RETURN_UNEXPECTED_IF_NULL(ds);
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "num_parallel_workers", kWikiTextNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "connector_queue_size", kWikiTextNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "dataset_dir", kWikiTextNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "usage", kWikiTextNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "num_samples", kWikiTextNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "shuffle", kWikiTextNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "num_shards", kWikiTextNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "shard_id", kWikiTextNode));
  std::string dataset_dir = json_obj["dataset_dir"];
  std::string usage = json_obj["usage"];
  int64_t num_samples = json_obj["num_samples"];
  ShuffleMode shuffle = static_cast<ShuffleMode>(json_obj["shuffle"]);
  int32_t num_shards = json_obj["num_shards"];
  int32_t shard_id = json_obj["shard_id"];
  std::shared_ptr<DatasetCache> cache;
  RETURN_IF_NOT_OK(DatasetCache::from_json(json_obj, &cache));
  *ds = std::make_shared<WikiTextNode>(dataset_dir, usage, num_samples, shuffle, num_shards, shard_id, cache);
  (void)((*ds)->SetNumWorkers(json_obj["num_parallel_workers"]));
  (void)((*ds)->SetConnectorQueueSize(json_obj["connector_queue_size"]));
  return Status::OK();
}

// Note: The following two functions are common among NonMappableSourceNode and should be promoted to its parent class.
// WikiText by itself is a non-mappable dataset that does not support sampling.
// However, if a cache operator is injected at some other place higher in the tree, that cache can
// inherit this sampler from the leaf, providing sampling support from the caching layer.
// That is why we setup the sampler for a leaf node that does not use sampling.
Status WikiTextNode::SetupSamplerForCache(std::shared_ptr<SamplerObj> *sampler) {
  *sampler = SelectSampler(num_samples_, shuffle_, num_shards_, shard_id_);
  return Status::OK();
}

// If a cache has been added into the ascendant tree over this WikiText node, then the cache will be executing
// a sampler for fetching the data. As such, any options in the WikiText node need to be reset to its defaults so
// that this WikiText node will produce the full set of data into the cache.
Status WikiTextNode::MakeSimpleProducer() {
  shard_id_ = 0;
  num_shards_ = 1;
  shuffle_ = ShuffleMode::kFalse;
  num_samples_ = 0;
  return Status::OK();
}

std::vector<std::string> WikiTextNode::WalkAllFiles(const std::string &usage, const std::string &dataset_dir) {
  std::vector<std::string> wikitext_files_list;
  Path train_prefix("wiki.train.tokens");
  Path test_prefix("wiki.test.tokens");
  Path valid_prefix("wiki.valid.tokens");
  Path dir(dataset_dir);

  if (usage == "train") {
    Path temp_path = dir / train_prefix;
    wikitext_files_list.push_back(temp_path.ToString());
  } else if (usage == "test") {
    Path temp_path = dir / test_prefix;
    wikitext_files_list.push_back(temp_path.ToString());
  } else if (usage == "valid") {
    Path temp_path = dir / valid_prefix;
    wikitext_files_list.push_back(temp_path.ToString());
  } else {
    Path temp_path = dir / train_prefix;
    wikitext_files_list.push_back(temp_path.ToString());
    Path temp_path1 = dir / test_prefix;
    wikitext_files_list.push_back(temp_path1.ToString());
    Path temp_path2 = dir / valid_prefix;
    wikitext_files_list.push_back(temp_path2.ToString());
  }
  return wikitext_files_list;
}
}  // namespace dataset
}  // namespace mindspore
