/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/ir/datasetops/source/squad_node.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/squad_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// Constructor for SQuADNode.
SQuADNode::SQuADNode(const std::string &dataset_dir, const std::string &usage, int64_t num_samples, ShuffleMode shuffle,
                     int32_t num_shards, int32_t shard_id, std::shared_ptr<DatasetCache> cache)
    : NonMappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      usage_(usage),
      num_samples_(num_samples),
      shuffle_(shuffle),
      num_shards_(num_shards),
      shard_id_(shard_id) {}

void SQuADNode::Print(std::ostream &out) const {
  out << (Name() + "(cache: " + ((cache_ != nullptr) ? "true" : "false") +
          ", num_shards: " + std::to_string(num_shards_) + ", shard_id: " + std::to_string(shard_id_) + ")");
}

std::shared_ptr<DatasetNode> SQuADNode::Copy() {
  auto node = std::make_shared<SQuADNode>(dataset_dir_, usage_, num_samples_, shuffle_, num_shards_, shard_id_, cache_);
  (void)node->SetNumWorkers(num_workers_);
  (void)node->SetConnectorQueueSize(connector_que_size_);
  return node;
}

Status SQuADNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("SQuADDataset", dataset_dir_));
  RETURN_IF_NOT_OK(ValidateStringValue("SQuADDataset", usage_, {"train", "dev", "all"}));
  RETURN_IF_NOT_OK(ValidateEnum("SQuADDataset", "ShuffleMode", shuffle_,
                                {ShuffleMode::kFalse, ShuffleMode::kFiles, ShuffleMode::kGlobal}));
  RETURN_IF_NOT_OK(ValidateScalar("SQuADDataset", "num_samples", num_samples_, {0}, false));
  RETURN_IF_NOT_OK(ValidateDatasetShardParams("SQuADDataset", num_shards_, shard_id_));

  return Status::OK();
}

// Function to build SQuADNode.
Status SQuADNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);

  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor(std::string("context"), DataType(DataType::DE_STRING), TensorImpl::kFlexible, 1)));
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor(std::string("question"), DataType(DataType::DE_STRING), TensorImpl::kFlexible, 1)));
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor(std::string("text"), DataType(DataType::DE_STRING), TensorImpl::kFlexible, 1)));
  RETURN_IF_NOT_OK(schema->AddColumn(
    ColDescriptor(std::string("answer_start"), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));

  // Create and initialize SQuADOp.
  std::shared_ptr<SQuADOp> squad_op =
    std::make_shared<SQuADOp>(dataset_dir_, usage_, num_workers_, num_samples_, worker_connector_size_,
                              std::move(schema), connector_que_size_, shuffle_files, num_shards_, shard_id_);

  RETURN_IF_NOT_OK(squad_op->Init());

  // If a global shuffle is used for SQuAD, it will inject a shuffle op over the SQuAD.
  // But, if there is a cache in the tree, we do not need the global shuffle and the shuffle op should not be built.
  // This is achieved in the cache transform pass where we call MakeSimpleProducer to reset SQuAD's shuffle
  // option to false.
  if (shuffle_ == ShuffleMode::kGlobal) {
    // Inject ShuffleOp.
    std::shared_ptr<DatasetOp> shuffle_op = nullptr;
    int64_t num_rows = 0;
    // Get the number of rows in the dataset.
    RETURN_IF_NOT_OK(SQuADOp::CountAllFileRows(dataset_dir_, usage_, &num_rows));
    // Add the shuffle op after this op.
    if (usage_ == "train" || usage_ == "dev") {
      int64_t num_files = 1;
      RETURN_IF_NOT_OK(AddShuffleOp(num_files, num_shards_, num_rows, 0, connector_que_size_, &shuffle_op));
    } else {
      int64_t num_files = 2;
      RETURN_IF_NOT_OK(AddShuffleOp(num_files, num_shards_, num_rows, 0, connector_que_size_, &shuffle_op));
    }
    shuffle_op->SetTotalRepeats(GetTotalRepeats());
    shuffle_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
    node_ops->push_back(shuffle_op);
  }
  squad_op->SetTotalRepeats(GetTotalRepeats());
  squad_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(squad_op);

  return Status::OK();
}

// Get the shard id of node.
Status SQuADNode::GetShardId(int32_t *shard_id) {
  *shard_id = shard_id_;
  return Status::OK();
}

// Get Dataset size.
Status SQuADNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                 int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t num_rows, sample_size;
  RETURN_IF_NOT_OK(SQuADOp::CountAllFileRows(dataset_dir_, usage_, &num_rows));
  sample_size = num_samples_;
  num_rows = static_cast<int64_t>(ceil(num_rows / (1.0 * num_shards_)));
  *dataset_size = sample_size > 0 ? std::min(num_rows, sample_size) : num_rows;
  dataset_size_ = *dataset_size;
  return Status::OK();
}

Status SQuADNode::to_json(nlohmann::json *out_json) {
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

// Note: The following two functions are common among NonMappableSourceNode and should be promoted to its parent
// class. SQuAD by itself is a non-mappable dataset that does not support sampling. However, if a cache operator is
// injected at some other place higher in the tree, that cache can inherit this sampler from the leaf, providing
// sampling support from the caching layer. That is why we setup the sampler for a leaf node that does not use
// sampling.
Status SQuADNode::SetupSamplerForCache(std::shared_ptr<SamplerObj> *sampler) {
  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);
  *sampler = SelectSampler(num_samples_, shuffle_files, num_shards_, shard_id_);
  return Status::OK();
}

// If a cache has been added into the ascendant tree over this SQuAD node, then the cache will be executing
// a sampler for fetching the data. As such, any options in the SQuAD node need to be reset to its defaults so
// that this SQuAD node will produce the full set of data into the cache.
Status SQuADNode::MakeSimpleProducer() {
  shard_id_ = 0;
  num_shards_ = 1;
  shuffle_ = ShuffleMode::kFalse;
  num_samples_ = 0;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
