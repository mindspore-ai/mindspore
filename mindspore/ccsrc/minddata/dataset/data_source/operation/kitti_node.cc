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

#include "minddata/dataset/data_source/operation/kitti_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/data_source/kitti_op.h"
#include "minddata/dataset/engine/serdes.h"
#include "minddata/dataset/util/status.h"

namespace mindspore::dataset {
// Constructor for KITTINode
KITTINode::KITTINode(const std::string &dataset_dir, const std::string &usage, bool decode,
                     const std::shared_ptr<SamplerObj> &sampler, const std::shared_ptr<DatasetCache> &cache = nullptr)
    : MappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      usage_(usage),
      decode_(decode),
      sampler_(sampler) {}

std::shared_ptr<DatasetNode> KITTINode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  auto node = std::make_shared<KITTINode>(dataset_dir_, usage_, decode_, sampler, cache_);
  return node;
}

void KITTINode::Print(std::ostream &out) const {
  out << (Name() + "(cache: " + ((cache_ != nullptr) ? "true" : "false") + ")");
}

Status KITTINode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  Path dir(dataset_dir_);
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("KITTIDataset", dataset_dir_));
  RETURN_IF_NOT_OK(ValidateDatasetSampler("KITTIDataset", sampler_));
  RETURN_IF_NOT_OK(ValidateStringValue("KITTIDataset", usage_, {"train", "test"}));
  return Status::OK();
}

// Function to build KITTINode
Status KITTINode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  RETURN_UNEXPECTED_IF_NULL(node_ops);
  auto schema = std::make_unique<DataSchema>();

  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor(std::string("image"), DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  if (usage_ == "train") {
    RETURN_IF_NOT_OK(
      schema->AddColumn(ColDescriptor(std::string("label"), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
    RETURN_IF_NOT_OK(schema->AddColumn(
      ColDescriptor(std::string("truncated"), DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
    RETURN_IF_NOT_OK(schema->AddColumn(
      ColDescriptor(std::string("occluded"), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
    RETURN_IF_NOT_OK(
      schema->AddColumn(ColDescriptor(std::string("alpha"), DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
    RETURN_IF_NOT_OK(
      schema->AddColumn(ColDescriptor(std::string("bbox"), DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
    RETURN_IF_NOT_OK(schema->AddColumn(
      ColDescriptor(std::string("dimensions"), DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
    RETURN_IF_NOT_OK(schema->AddColumn(
      ColDescriptor(std::string("location"), DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
    RETURN_IF_NOT_OK(schema->AddColumn(
      ColDescriptor(std::string("rotation_y"), DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
  }
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));

  std::shared_ptr<KITTIOp> kitti_op;
  kitti_op = std::make_shared<KITTIOp>(dataset_dir_, usage_, num_workers_, connector_que_size_, decode_,
                                       std::move(schema), std::move(sampler_rt));
  kitti_op->SetTotalRepeats(GetTotalRepeats());
  kitti_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(kitti_op);
  return Status::OK();
}

// Get the shard id of node
Status KITTINode::GetShardId(int32_t *shard_id) {
  RETURN_UNEXPECTED_IF_NULL(shard_id);
  *shard_id = sampler_->ShardId();

  return Status::OK();
}

// Get Dataset size
Status KITTINode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                 int64_t *dataset_size) {
  RETURN_UNEXPECTED_IF_NULL(dataset_size);
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t num_rows = 0, sample_size;
  std::vector<std::shared_ptr<DatasetOp>> ops;
  RETURN_IF_NOT_OK(Build(&ops));
  CHECK_FAIL_RETURN_UNEXPECTED(!ops.empty(), "Unable to build KITTIOp.");
  auto op = std::dynamic_pointer_cast<KITTIOp>(ops.front());
  RETURN_IF_NOT_OK(op->CountTotalRows(&num_rows));
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

Status KITTINode::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args, sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["num_parallel_workers"] = num_workers_;
  args["connector_queue_size"] = connector_que_size_;
  args["dataset_dir"] = dataset_dir_;
  args["usage"] = usage_;
  args["decode"] = decode_;
  if (cache_ != nullptr) {
    nlohmann::json cache_args;
    RETURN_IF_NOT_OK(cache_->to_json(&cache_args));
    args["cache"] = cache_args;
  }
  *out_json = args;
  return Status::OK();
}

Status KITTINode::from_json(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  RETURN_UNEXPECTED_IF_NULL(ds);
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "num_parallel_workers", kKITTINode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "connector_queue_size", kKITTINode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "dataset_dir", kKITTINode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "usage", kKITTINode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "decode", kKITTINode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "sampler", kKITTINode));
  std::string dataset_dir = json_obj["dataset_dir"];
  std::string usage = json_obj["usage"];
  bool decode = json_obj["decode"];
  std::shared_ptr<SamplerObj> sampler;
  RETURN_IF_NOT_OK(Serdes::ConstructSampler(json_obj["sampler"], &sampler));
  std::shared_ptr<DatasetCache> cache;
  RETURN_IF_NOT_OK(DatasetCache::from_json(json_obj, &cache));
  *ds = std::make_shared<KITTINode>(dataset_dir, usage, decode, sampler, cache);
  (void)((*ds)->SetNumWorkers(json_obj["num_parallel_workers"]));
  (void)((*ds)->SetConnectorQueueSize(json_obj["connector_queue_size"]));
  return Status::OK();
}
}  // namespace mindspore::dataset
