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

#include "minddata/dataset/engine/ir/datasetops/source/multi30k_node.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/multi30k_op.h"

namespace mindspore {
namespace dataset {
Multi30kNode::Multi30kNode(const std::string &dataset_dir, const std::string &usage,
                           const std::vector<std::string> &language_pair, int32_t num_samples, ShuffleMode shuffle,
                           int32_t num_shards, int32_t shard_id, std::shared_ptr<DatasetCache> cache)
    : NonMappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      usage_(usage),
      language_pair_(language_pair),
      num_samples_(num_samples),
      shuffle_(shuffle),
      num_shards_(num_shards),
      shard_id_(shard_id),
      multi30k_files_list_(WalkAllFiles(usage, dataset_dir)) {
  GlobalContext::config_manager()->set_num_shards_for_auto_num_workers(num_shards_);
}

void Multi30kNode::Print(std::ostream &out) const {
  out << (Name() + "(cache: " + ((cache_ != nullptr) ? "true" : "false") +
          ", num_shards: " + std::to_string(num_shards_) + ", shard_id: " + std::to_string(shard_id_) + ")");
}

std::shared_ptr<DatasetNode> Multi30kNode::Copy() {
  auto node = std::make_shared<Multi30kNode>(dataset_dir_, usage_, language_pair_, num_samples_, shuffle_, num_shards_,
                                             shard_id_, cache_);
  (void)node->SetNumWorkers(num_workers_);
  (void)node->SetConnectorQueueSize(connector_que_size_);
  return node;
}

// Function to build Multi30kNode
Status Multi30kNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);

  std::vector<std::string> sorted_dataset_files = multi30k_files_list_;
  std::sort(sorted_dataset_files.begin(), sorted_dataset_files.end());

  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("text", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 1)));
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("translation", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 1)));

  std::shared_ptr<Multi30kOp> multi30k_op =
    std::make_shared<Multi30kOp>(num_workers_, num_samples_, language_pair_, worker_connector_size_, std::move(schema),
                                 sorted_dataset_files, connector_que_size_, shuffle_files, num_shards_, shard_id_);
  RETURN_IF_NOT_OK(multi30k_op->Init());

  if (shuffle_ == ShuffleMode::kGlobal) {
    // Inject ShuffleOp
    std::shared_ptr<DatasetOp> shuffle_op = nullptr;
    int64_t num_rows = 0;

    // First, get the number of rows in the dataset
    RETURN_IF_NOT_OK(Multi30kOp::CountAllFileRows(sorted_dataset_files, &num_rows));

    // Add the shuffle op after this op
    RETURN_IF_NOT_OK(
      AddShuffleOp(sorted_dataset_files.size(), num_shards_, num_rows, 0, connector_que_size_, &shuffle_op));
    shuffle_op->SetTotalRepeats(GetTotalRepeats());
    shuffle_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
    node_ops->push_back(shuffle_op);
  }
  multi30k_op->SetTotalRepeats(GetTotalRepeats());
  multi30k_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  // Add Multi30kOp
  node_ops->push_back(multi30k_op);

  return Status::OK();
}

Status Multi30kNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("Multi30kDataset", dataset_dir_));
  RETURN_IF_NOT_OK(ValidateDatasetFilesParam("Multi30kDataset", multi30k_files_list_));
  RETURN_IF_NOT_OK(ValidateStringValue("Multi30kDataset", usage_, {"train", "valid", "test", "all"}));
  RETURN_IF_NOT_OK(ValidateEnum("Multi30kDataset", "ShuffleMode", shuffle_,
                                {ShuffleMode::kFalse, ShuffleMode::kFiles, ShuffleMode::kGlobal}));

  const int kLanguagePairSize = 2;
  if (language_pair_.size() != kLanguagePairSize) {
    std::string err_msg =
      "Multi30kDataset: language_pair expecting size 2, but got: " + std::to_string(language_pair_.size());
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  const std::vector<std::vector<std::string>> support_language_pair = {{"en", "de"}, {"de", "en"}};
  if (language_pair_ != support_language_pair[0] && language_pair_ != support_language_pair[1]) {
    std::string err_msg = R"(Multi30kDataset: language_pair must be {"en", "de"} or {"de", "en"}.)";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  RETURN_IF_NOT_OK(ValidateScalar("Multi30kDataset", "num_samples", num_samples_, {0}, false));
  RETURN_IF_NOT_OK(ValidateDatasetShardParams("Multi30kDataset", num_shards_, shard_id_));
  return Status::OK();
}

Status Multi30kNode::GetShardId(int32_t *shard_id) {
  *shard_id = shard_id_;
  return Status::OK();
}

Status Multi30kNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                    int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t num_rows, sample_size = num_samples_;
  RETURN_IF_NOT_OK(Multi30kOp::CountAllFileRows(multi30k_files_list_, &num_rows));
  num_rows = static_cast<int64_t>(ceil(num_rows / (1.0 * num_shards_)));
  *dataset_size = sample_size > 0 ? std::min(num_rows, sample_size) : num_rows;
  dataset_size_ = *dataset_size;
  return Status::OK();
}

Status Multi30kNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["num_parallel_workers"] = num_workers_;
  args["connector_queue_size"] = connector_que_size_;
  args["dataset_dir"] = dataset_dir_;
  args["num_samples"] = num_samples_;
  args["shuffle"] = shuffle_;
  args["num_shards"] = num_shards_;
  args["shard_id"] = shard_id_;
  args["language_pair"] = language_pair_;
  if (cache_ != nullptr) {
    nlohmann::json cache_args;
    RETURN_IF_NOT_OK(cache_->to_json(&cache_args));
    args["cache"] = cache_args;
  }
  *out_json = args;
  return Status::OK();
}

Status Multi30kNode::SetupSamplerForCache(std::shared_ptr<SamplerObj> *sampler) {
  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);
  *sampler = SelectSampler(num_samples_, shuffle_files, num_shards_, shard_id_);
  return Status::OK();
}

Status Multi30kNode::MakeSimpleProducer() {
  shard_id_ = 0;
  num_shards_ = 1;
  shuffle_ = ShuffleMode::kFalse;
  num_samples_ = 0;
  return Status::OK();
}

std::vector<std::string> Multi30kNode::WalkAllFiles(const std::string &usage, const std::string &dataset_dir) {
  std::vector<std::string> multi30k_files_list;
  Path train_en("training/train.en");
  Path test_en("mmt16_task1_test/test.en");
  Path valid_en("validation/val.en");
  Path dir(dataset_dir);

  if (usage == "train") {
    Path temp_path = dir / train_en;
    multi30k_files_list.push_back(temp_path.ToString());
  } else if (usage == "test") {
    Path temp_path = dir / test_en;
    multi30k_files_list.push_back(temp_path.ToString());
  } else if (usage == "valid") {
    Path temp_path = dir / valid_en;
    multi30k_files_list.push_back(temp_path.ToString());
  } else {
    Path temp_path = dir / train_en;
    multi30k_files_list.push_back(temp_path.ToString());
    Path temp_path1 = dir / test_en;
    multi30k_files_list.push_back(temp_path1.ToString());
    Path temp_path2 = dir / valid_en;
    multi30k_files_list.push_back(temp_path2.ToString());
  }
  return multi30k_files_list;
}
}  // namespace dataset
}  // namespace mindspore
