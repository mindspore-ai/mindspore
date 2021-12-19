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

#include "minddata/dataset/engine/ir/datasetops/source/lfw_node.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/lfw_op.h"
#include "minddata/dataset/engine/serdes.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
LFWNode::LFWNode(const std::string &dataset_dir, const std::string &task, const std::string &usage,
                 const std::string &image_set, bool decode, const std::shared_ptr<SamplerObj> &sampler,
                 const std::shared_ptr<DatasetCache> &cache = nullptr)
    : MappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      task_(task),
      usage_(usage),
      image_set_(image_set),
      decode_(decode),
      sampler_(sampler) {}

std::shared_ptr<DatasetNode> LFWNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  auto node = std::make_shared<LFWNode>(dataset_dir_, task_, usage_, image_set_, decode_, sampler, cache_);
  return node;
}

void LFWNode::Print(std::ostream &out) const { out << Name(); }

Status LFWNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("LFWDataset", dataset_dir_));
  RETURN_IF_NOT_OK(ValidateStringValue("LFWDataset", task_, {"people", "pairs"}));
  RETURN_IF_NOT_OK(ValidateStringValue("LFWDataset", usage_, {"10fold", "train", "test", "all"}));
  RETURN_IF_NOT_OK(ValidateStringValue("LFWDataset", image_set_, {"original", "funneled", "deepfunneled"}));
  RETURN_IF_NOT_OK(ValidateDatasetSampler("LFWDataset", sampler_));
  return Status::OK();
}

Status LFWNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  // Do internal Schema generation.
  // This arg is exist in LFWOp, but not externalized (in Python API).
  RETURN_UNEXPECTED_IF_NULL(node_ops);
  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  if (task_ == "people") {
    RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  } else if (task_ == "pairs") {
    RETURN_IF_NOT_OK(
      schema->AddColumn(ColDescriptor("image1", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
    RETURN_IF_NOT_OK(
      schema->AddColumn(ColDescriptor("image2", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  }
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor(std::string("label"), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));

  auto op = std::make_shared<LFWOp>(num_workers_, dataset_dir_, task_, usage_, image_set_, connector_que_size_, decode_,
                                    std::move(schema), std::move(sampler_rt));
  op->SetTotalRepeats(GetTotalRepeats());
  op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(op);
  return Status::OK();
}

// Get the shard id of node
Status LFWNode::GetShardId(int32_t *shard_id) {
  RETURN_UNEXPECTED_IF_NULL(shard_id);
  *shard_id = sampler_->ShardId();

  return Status::OK();
}

// Get Dataset size.
Status LFWNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                               int64_t *dataset_size) {
  RETURN_UNEXPECTED_IF_NULL(dataset_size);
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t sample_size, num_rows;
  std::vector<std::shared_ptr<DatasetOp>> ops;
  RETURN_IF_NOT_OK(Build(&ops));
  CHECK_FAIL_RETURN_UNEXPECTED(!ops.empty(), "[Internal ERROR] Unable to build LFWOp.");
  auto op = std::dynamic_pointer_cast<LFWOp>(ops.front());
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

Status LFWNode::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args, sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["num_parallel_workers"] = num_workers_;
  args["dataset_dir"] = dataset_dir_;
  args["task"] = task_;
  args["usage"] = usage_;
  args["image_set"] = image_set_;
  args["decode"] = decode_;
  if (cache_ != nullptr) {
    nlohmann::json cache_args;
    RETURN_IF_NOT_OK(cache_->to_json(&cache_args));
    args["cache"] = cache_args;
  }
  *out_json = args;
  return Status::OK();
}

#ifndef ENABLE_ANDROID
Status LFWNode::from_json(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  RETURN_UNEXPECTED_IF_NULL(ds);
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "num_parallel_workers", kTFRecordNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "dataset_dir", kTFRecordNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "task", kTFRecordNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "usage", kTFRecordNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "image_set", kTFRecordNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "decode", kTFRecordNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "sampler", kTFRecordNode));
  std::string dataset_dir = json_obj["dataset_dir"];
  std::string task = json_obj["task"];
  std::string usage = json_obj["usage"];
  std::string image_set = json_obj["image_set"];
  bool decode = json_obj["decode"];
  std::shared_ptr<SamplerObj> sampler;
  RETURN_IF_NOT_OK(Serdes::ConstructSampler(json_obj["sampler"], &sampler));
  std::shared_ptr<DatasetCache> cache = nullptr;
  RETURN_IF_NOT_OK(DatasetCache::from_json(json_obj, &cache));
  *ds = std::make_shared<LFWNode>(dataset_dir, task, usage, image_set, decode, sampler, cache);
  (*ds)->SetNumWorkers(json_obj["num_parallel_workers"]);
  return Status::OK();
}
#endif
}  // namespace dataset
}  // namespace mindspore
