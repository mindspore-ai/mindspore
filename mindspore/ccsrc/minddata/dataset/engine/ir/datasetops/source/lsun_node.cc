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

#include "minddata/dataset/engine/ir/datasetops/source/lsun_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/lsun_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
LSUNNode::LSUNNode(const std::string &dataset_dir, const std::string &usage, const std::vector<std::string> &classes,
                   bool decode, std::shared_ptr<SamplerObj> sampler, std::shared_ptr<DatasetCache> cache = nullptr)
    : MappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      usage_(usage),
      classes_(classes),
      decode_(decode),
      sampler_(sampler) {}

std::shared_ptr<DatasetNode> LSUNNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  auto node = std::make_shared<LSUNNode>(dataset_dir_, usage_, classes_, decode_, sampler, cache_);
  return node;
}

void LSUNNode::Print(std::ostream &out) const {
  out << (Name() + "(path: " + dataset_dir_ + ", decode: " + (decode_ ? "true" : "false") + ")");
}

Status LSUNNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("LSUNDataset", dataset_dir_));
  RETURN_IF_NOT_OK(ValidateDatasetSampler("LSUNDataset", sampler_));
  RETURN_IF_NOT_OK(ValidateStringValue("LSUNDataset", usage_, {"train", "test", "valid", "all"}));
  for (auto class_name : classes_) {
    RETURN_IF_NOT_OK(ValidateStringValue("LSUNDataset", class_name,
                                         {"bedroom", "bridge", "church_outdoor", "classroom", "conference_room",
                                          "dining_room", "kitchen", "living_room", "restaurant", "tower"}));
  }

  return Status::OK();
}

Status LSUNNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  // Do internal Schema generation.
  // This arg is exist in LSUNOp, but not externalized (in Python API).
  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));

  auto op = std::make_shared<LSUNOp>(num_workers_, dataset_dir_, connector_que_size_, usage_, classes_, decode_,
                                     std::move(schema), std::move(sampler_rt));
  op->SetTotalRepeats(GetTotalRepeats());
  op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  RETURN_UNEXPECTED_IF_NULL(node_ops);
  node_ops->push_back(op);
  return Status::OK();
}

// Get the shard id of node
Status LSUNNode::GetShardId(int32_t *shard_id) {
  RETURN_UNEXPECTED_IF_NULL(shard_id);
  *shard_id = sampler_->ShardId();

  return Status::OK();
}

// Get Dataset size
Status LSUNNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t sample_size, num_rows;
  RETURN_IF_NOT_OK(LSUNOp::CountRowsAndClasses(dataset_dir_, usage_, classes_, &num_rows, nullptr));
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

Status LSUNNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args, sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["num_parallel_workers"] = num_workers_;
  args["dataset_dir"] = dataset_dir_;
  args["usage"] = usage_;
  args["classes"] = classes_;
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
