/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/ir/datasetops/source/sogou_news_node.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
SogouNewsNode::SogouNewsNode(const std::string &dataset_dir, const std::string &usage, int64_t num_samples,
                             ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                             const std::shared_ptr<DatasetCache> &cache)
    : NonMappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      num_samples_(num_samples),
      shuffle_(shuffle),
      num_shards_(num_shards),
      shard_id_(shard_id),
      usage_(usage),
      sogou_news_files_list_(WalkAllFiles(usage, dataset_dir)) {
  // Update the num_shards_ in global context. this number is only used for now by auto_num_worker_pass.
  // User discretion is advised. Auto_num_worker_pass is currently an experimental feature which can still work
  // if the num_shards_ isn't 100% correct. The reason behind is for now, PreBuildSampler doesn't offer a way to
  // return num_shards. Once PreBuildSampler is phased out, this can be cleaned up.
  GlobalContext::config_manager()->set_num_shards_for_auto_num_workers(num_shards_);
}

std::shared_ptr<DatasetNode> SogouNewsNode::Copy() {
  auto node =
    std::make_shared<SogouNewsNode>(dataset_dir_, usage_, num_samples_, shuffle_, num_shards_, shard_id_, cache_);
  (void)node->SetNumWorkers(num_workers_);
  (void)node->SetConnectorQueueSize(connector_que_size_);
  return node;
}

void SogouNewsNode::Print(std::ostream &out) const {
  out << (Name() + "(cache: " + ((cache_ != nullptr) ? "true" : "false") +
          ", num_shards: " + std::to_string(num_shards_) + ", shard_id: " + std::to_string(shard_id_) + ")");
}

Status SogouNewsNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("SogouNewsNode", dataset_dir_));
  RETURN_IF_NOT_OK(ValidateStringValue("SogouNewsNode", usage_, {"train", "test", "all"}));
  RETURN_IF_NOT_OK(ValidateEnum("SogouNewsNode", "ShuffleMode", shuffle_,
                                {ShuffleMode::kFalse, ShuffleMode::kFiles, ShuffleMode::kGlobal}));
  if (num_samples_ < 0) {
    std::string err_msg = "SogouNewsNode: Invalid number of samples: " + std::to_string(num_samples_);
    MS_LOG(ERROR) << err_msg;
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  RETURN_IF_NOT_OK(ValidateDatasetShardParams("SogouNewsNode", num_shards_, shard_id_));
  return Status::OK();
}

Status SogouNewsNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);

  // Sort the dataset files in a lexicographical order.
  std::vector<std::string> sorted_dataset_files = sogou_news_files_list_;
  std::sort(sorted_dataset_files.begin(), sorted_dataset_files.end());

  std::vector<std::shared_ptr<SogouNewsOp::BaseRecord>> column_default;
  column_default.push_back(std::make_shared<SogouNewsOp::Record<std::string>>(SogouNewsOp::STRING, ""));
  column_default.push_back(std::make_shared<SogouNewsOp::Record<std::string>>(SogouNewsOp::STRING, ""));
  column_default.push_back(std::make_shared<SogouNewsOp::Record<std::string>>(SogouNewsOp::STRING, ""));

  std::vector<std::string> column_name = {"index", "title", "content"};
  char field_delim = ',';
  auto sogou_news_op = std::make_shared<SogouNewsOp>(num_workers_, num_samples_, worker_connector_size_,
                                                     connector_que_size_, shuffle_files, num_shards_, shard_id_,
                                                     field_delim, column_default, column_name, sogou_news_files_list_);

  RETURN_IF_NOT_OK(sogou_news_op->Init());

  // If a global shuffle is used for SogouNews, it will inject a shuffle op over the SogouNews.
  // But, if there is a cache in the tree, we do not need the global shuffle and the shuffle op should not be
  // built.This is achieved in the cache transform pass where we call MakeSimpleProducer to reset SogouNews
  // shuffle option to false.
  if (shuffle_ == ShuffleMode::kGlobal) {
    // Inject ShuffleOp.
    std::shared_ptr<DatasetOp> shuffle_op = nullptr;
    int64_t num_rows = 0;

    // First, get the number of rows in the dataset.
    RETURN_IF_NOT_OK(SogouNewsOp::CountAllFileRows(sogou_news_files_list_, false, &num_rows));
    // Add the shuffle op after this op.
    RETURN_IF_NOT_OK(
      AddShuffleOp(sorted_dataset_files.size(), num_shards_, num_rows, 0, connector_que_size_, &shuffle_op));
    shuffle_op->SetTotalRepeats(GetTotalRepeats());
    shuffle_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
    node_ops->push_back(shuffle_op);
  }
  sogou_news_op->SetTotalRepeats(GetTotalRepeats());
  sogou_news_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(sogou_news_op);
  return Status::OK();
}

Status SogouNewsNode::GetShardId(int32_t *shard_id) {
  *shard_id = shard_id_;
  return Status::OK();
}

Status SogouNewsNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                     int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }

  int64_t num_rows, sample_size;
  RETURN_IF_NOT_OK(SogouNewsOp::CountAllFileRows(sogou_news_files_list_, false, &num_rows));
  sample_size = num_samples_;
  num_rows = static_cast<int64_t>(ceil(num_rows / (1.0 * num_shards_)));
  *dataset_size = sample_size > 0 ? std::min(num_rows, sample_size) : num_rows;
  dataset_size_ = *dataset_size;
  return Status::OK();
}

Status SogouNewsNode::to_json(nlohmann::json *out_json) {
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

Status SogouNewsNode::SetupSamplerForCache(std::shared_ptr<SamplerObj> *sampler) {
  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);
  *sampler = SelectSampler(num_samples_, shuffle_files, num_shards_, shard_id_);
  return Status::OK();
}

Status SogouNewsNode::MakeSimpleProducer() {
  shard_id_ = 0;
  num_shards_ = 1;
  shuffle_ = ShuffleMode::kFalse;
  num_samples_ = 0;
  return Status::OK();
}

std::vector<std::string> SogouNewsNode::WalkAllFiles(const std::string &usage, const std::string &dataset_dir) {
  std::vector<std::string> sogou_news_files_list;
  Path train_prefix("train.csv");
  Path test_prefix("test.csv");
  Path dir(dataset_dir);

  if (usage == "train") {
    Path temp_path = dir / train_prefix;
    sogou_news_files_list.push_back(temp_path.ToString());
  } else if (usage == "test") {
    Path temp_path = dir / test_prefix;
    sogou_news_files_list.push_back(temp_path.ToString());
  } else {
    Path temp_path = dir / train_prefix;
    if (temp_path.Exists()) {
      sogou_news_files_list.push_back(temp_path.ToString());
    }
    Path temp_path1 = dir / test_prefix;
    if (temp_path1.Exists()) {
      sogou_news_files_list.push_back(temp_path1.ToString());
    }
  }
  return sogou_news_files_list;
}
}  // namespace dataset
}  // namespace mindspore
