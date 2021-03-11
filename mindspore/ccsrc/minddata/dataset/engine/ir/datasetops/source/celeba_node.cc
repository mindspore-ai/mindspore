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

#include "minddata/dataset/engine/ir/datasetops/source/celeba_node.h"

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include "minddata/dataset/engine/datasetops/source/celeba_op.h"
#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

// Constructor for CelebANode
CelebANode::CelebANode(const std::string &dataset_dir, const std::string &usage,
                       const std::shared_ptr<SamplerObj> &sampler, const bool &decode,
                       const std::set<std::string> &extensions, const std::shared_ptr<DatasetCache> &cache)
    : MappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      usage_(usage),
      sampler_(sampler),
      decode_(decode),
      extensions_(extensions) {}

std::shared_ptr<DatasetNode> CelebANode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  auto node = std::make_shared<CelebANode>(dataset_dir_, usage_, sampler, decode_, extensions_, cache_);
  return node;
}

void CelebANode::Print(std::ostream &out) const {
  out << Name() + "(cache:" + ((cache_ != nullptr) ? "true" : "false") + ")";
}

Status CelebANode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("CelebANode", dataset_dir_));

  RETURN_IF_NOT_OK(ValidateDatasetSampler("CelebANode", sampler_));

  RETURN_IF_NOT_OK(ValidateStringValue("CelebANode", usage_, {"all", "train", "valid", "test"}));

  return Status::OK();
}

// Function to build CelebANode
Status CelebANode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  // label is like this:0 1 0 0 1......
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("attr", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));

  auto celeba_op = std::make_shared<CelebAOp>(num_workers_, rows_per_buffer_, dataset_dir_, connector_que_size_,
                                              decode_, usage_, extensions_, std::move(schema), std::move(sampler_rt));
  celeba_op->set_total_repeats(GetTotalRepeats());
  celeba_op->set_num_repeats_per_epoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(celeba_op);

  return Status::OK();
}

// Get the shard id of node
Status CelebANode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();

  return Status::OK();
}

// Get Dataset size
Status CelebANode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                  int64_t *dataset_size) {
  int64_t num_rows, sample_size;
  std::ifstream partition_file;
  std::string line;
  Path folder_path(dataset_dir_);
  std::ifstream attr_file((folder_path / "list_attr_celeba.txt").toString());
  if (!attr_file.is_open()) {
    std::string attr_file_name = (folder_path / "list_attr_celeba.txt").toString();
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open Celeba attr file: " + attr_file_name);
  }

  std::string rows_num;
  (void)getline(attr_file, rows_num);
  try {
    num_rows = static_cast<int64_t>(std::stoul(rows_num));  // First line is rows number in attr file
  } catch (std::invalid_argument &e) {
    RETURN_STATUS_UNEXPECTED(
      "Invalid data, failed to convert rows_num from attr_file to unsigned long, invalid argument: " + rows_num);
  } catch (std::out_of_range &e) {
    RETURN_STATUS_UNEXPECTED(
      "Invalid data, failed to convert rows_num from attr_file to unsigned long, out of range: " + rows_num);
  }
  if (usage_ != "all") {
    int64_t partition_num = 0;
    char usage_type;
    if (usage_ == "train") {
      usage_type = '0';
    } else {
      if (usage_ == "valid") {
        usage_type = '1';
      } else {
        if (usage_ == "test")
          usage_type = '2';
        else
          RETURN_STATUS_UNEXPECTED("Invalid usage.");
      }
    }
    if (!partition_file.is_open()) {
      partition_file.open((folder_path / "list_eval_partition.txt").toString());
    }
    if (partition_file.is_open()) {
      while (getline(partition_file, line)) {
        int start = line.find(' ');
        if (line.at(start + 1) == usage_type) {
          partition_num++;
        }
      }
    } else {
      std::string partition_file_name = "list_eval_partition.txt";
      RETURN_STATUS_UNEXPECTED("Invalid file, failed to open CelebA partition file: " + partition_file_name);
    }
    num_rows = std::min(num_rows, partition_num);
  }

  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));
  sample_size = sampler_rt->CalculateNumSamples(num_rows);
  if (sample_size == -1) {
    RETURN_IF_NOT_OK(size_getter->DryRun(shared_from_this(), &sample_size));
  }
  *dataset_size = sample_size;
  return Status::OK();
}

Status CelebANode::to_json(nlohmann::json *out_json) {
  nlohmann::json args, sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["num_parallel_workers"] = num_workers_;
  args["dataset_dir"] = dataset_dir_;
  args["decode"] = decode_;
  args["extensions"] = extensions_;
  args["usage"] = usage_;
  if (cache_ != nullptr) {
    nlohmann::json cache_args;
    RETURN_IF_NOT_OK(cache_->to_json(&cache_args));
    args["cache"] = cache_args;
  }
  *out_json = args;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
