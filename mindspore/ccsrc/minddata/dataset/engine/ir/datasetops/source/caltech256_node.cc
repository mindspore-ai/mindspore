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
#include "minddata/dataset/engine/ir/datasetops/source/caltech256_node.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/caltech_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/serdes.h"
#endif
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
const std::set<std::string> kExts = {".jpg", ".JPEG"};

Caltech256Node::Caltech256Node(const std::string &dataset_dir, bool decode, std::shared_ptr<SamplerObj> sampler,
                               std::shared_ptr<DatasetCache> cache = nullptr)
    : MappableSourceNode(std::move(cache)), dataset_dir_(dataset_dir), decode_(decode), sampler_(sampler) {}

std::shared_ptr<DatasetNode> Caltech256Node::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  auto node = std::make_shared<Caltech256Node>(dataset_dir_, decode_, sampler, cache_);
  return node;
}

void Caltech256Node::Print(std::ostream &out) const {
  out << (Name() + "(path: " + dataset_dir_ + ", decode: " + (decode_ ? "true" : "false") + ")");
}

Status Caltech256Node::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("Caltech256Node", dataset_dir_));
  RETURN_IF_NOT_OK(ValidateDatasetSampler("Caltech256Node", sampler_));
  return Status::OK();
}

Status Caltech256Node::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  // Do internal Schema generation.
  // This arg exists in CaltechOp, but is not externalized (in Python API).
  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));

  auto op = std::make_shared<CaltechOp>(num_workers_, dataset_dir_, connector_que_size_, decode_, std::move(schema),
                                        std::move(sampler_rt));
  op->SetTotalRepeats(GetTotalRepeats());
  op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(op);
  return Status::OK();
}

// Get the shard id of node.
Status Caltech256Node::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();
  return Status::OK();
}

// Get Dataset size.
Status Caltech256Node::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                      int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t sample_size, num_rows;
  RETURN_IF_NOT_OK(CaltechOp::CountRowsAndClasses(dataset_dir_, kExts, &num_rows, nullptr, {}));
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

Status Caltech256Node::to_json(nlohmann::json *out_json) {
  nlohmann::json args, sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["num_parallel_workers"] = num_workers_;
  args["connector_queue_size"] = connector_que_size_;
  args["dataset_dir"] = dataset_dir_;
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
Status Caltech256Node::from_json(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "num_parallel_workers", kCaltech256Node));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "connector_queue_size", kCaltech256Node));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "dataset_dir", kCaltech256Node));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "decode", kCaltech256Node));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "sampler", kCaltech256Node));
  std::string dataset_dir = json_obj["dataset_dir"];
  bool decode = json_obj["decode"];
  std::shared_ptr<SamplerObj> sampler;
  RETURN_IF_NOT_OK(Serdes::ConstructSampler(json_obj["sampler"], &sampler));
  std::shared_ptr<DatasetCache> cache = nullptr;
  RETURN_IF_NOT_OK(DatasetCache::from_json(json_obj, &cache));
  *ds = std::make_shared<Caltech256Node>(dataset_dir, decode, sampler, cache);
  (void)(*ds)->SetNumWorkers(json_obj["num_parallel_workers"]);
  (void)(*ds)->SetConnectorQueueSize(json_obj["connector_queue_size"]);
  return Status::OK();
}
#endif
}  // namespace dataset
}  // namespace mindspore
