/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/ir/datasetops/source/cityscapes_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/cityscapes_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// Constructor for CityscapesNode
CityscapesNode::CityscapesNode(const std::string &dataset_dir, const std::string &usage,
                               const std::string &quality_mode, const std::string &task, bool decode,
                               std::shared_ptr<SamplerObj> sampler, std::shared_ptr<DatasetCache> cache)
    : MappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      usage_(usage),
      quality_mode_(quality_mode),
      task_(task),
      sampler_(sampler),
      decode_(decode) {}

std::shared_ptr<DatasetNode> CityscapesNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  auto node = std::make_shared<CityscapesNode>(dataset_dir_, usage_, quality_mode_, task_, decode_, sampler, cache_);
  return node;
}

void CityscapesNode::Print(std::ostream &out) const {
  out << Name() + "(dataset dir:" + dataset_dir_;
  out << ", task:" + task_ << ", quality mode:" + quality_mode_ << ", usage:" + usage_;
  if (sampler_ != nullptr) {
    out << ", sampler";
  }
  if (cache_ != nullptr) {
    out << ", cache";
  }
  out << ")";
}

Status CityscapesNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("CityscapesNode", dataset_dir_));

  RETURN_IF_NOT_OK(ValidateStringValue("CityscapesNode", task_, {"instance", "semantic", "polygon", "color"}));
  RETURN_IF_NOT_OK(ValidateStringValue("CityscapesNode", quality_mode_, {"fine", "coarse"}));
  if (quality_mode_ == "fine") {
    RETURN_IF_NOT_OK(ValidateStringValue("CityscapesNode", usage_, {"train", "test", "val", "all"}));
  } else {
    RETURN_IF_NOT_OK(ValidateStringValue("CityscapesNode", usage_, {"train", "train_extra", "val", "all"}));
  }
  RETURN_IF_NOT_OK(ValidateDatasetSampler("CityscapesNode", sampler_));
  return Status::OK();
}

// Function to build CityscapesOp for Cityscapes
Status CityscapesNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  TensorShape scalar = TensorShape::CreateScalar();
  if (task_ == "polygon") {
    RETURN_IF_NOT_OK(
      schema->AddColumn(ColDescriptor("task", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 0, &scalar)));
  } else {
    RETURN_IF_NOT_OK(
      schema->AddColumn(ColDescriptor("task", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 0, &scalar)));
  }
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));

  auto cityscapes_op = std::make_shared<CityscapesOp>(num_workers_, dataset_dir_, usage_, quality_mode_, task_, decode_,
                                                      connector_que_size_, std::move(schema), std::move(sampler_rt));
  cityscapes_op->SetTotalRepeats(GetTotalRepeats());
  cityscapes_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(cityscapes_op);
  return Status::OK();
}

// Get the shard id of node
Status CityscapesNode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();
  return Status::OK();
}

// Get Dataset size
Status CityscapesNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                      int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }

  int64_t num_rows, sample_size;
  RETURN_IF_NOT_OK(CityscapesOp::CountTotalRows(dataset_dir_, usage_, quality_mode_, task_, &num_rows));
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));
  sample_size = sampler_rt->CalculateNumSamples(num_rows);
  if (sample_size == -1) {
    RETURN_IF_NOT_OK(size_getter->DryRun(shared_from_this(), &sample_size));
  }

  *dataset_size = sample_size;
  dataset_size_ = *dataset_size;
  return Status::OK();
}

Status CityscapesNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args, sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["num_parallel_workers"] = num_workers_;
  args["dataset_dir"] = dataset_dir_;
  args["usage"] = usage_;
  args["quality_mode"] = quality_mode_;
  args["task"] = task_;
  args["decode"] = decode_;
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
