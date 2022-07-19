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

#include "minddata/dataset/engine/ir/datasetops/source/qmnist_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/qmnist_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/serdes.h"
#endif
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
QMnistNode::QMnistNode(const std::string &dataset_dir, const std::string &usage, bool compat,
                       std::shared_ptr<SamplerObj> sampler, std::shared_ptr<DatasetCache> cache)
    : MappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      usage_(usage),
      compat_(compat),
      sampler_(sampler) {}

std::shared_ptr<DatasetNode> QMnistNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  auto node = std::make_shared<QMnistNode>(dataset_dir_, usage_, compat_, sampler, cache_);
  (void)node->SetNumWorkers(num_workers_);
  (void)node->SetConnectorQueueSize(connector_que_size_);
  return node;
}

void QMnistNode::Print(std::ostream &out) const {
  out << (Name() + "(dataset dir: " + dataset_dir_ + ", usage: " + usage_ +
          ", compat: " + (compat_ ? "true" : "false") + ", cache: " + ((cache_ != nullptr) ? "true" : "false") + ")");
}

Status QMnistNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("QMnistDataset", dataset_dir_));
  RETURN_IF_NOT_OK(ValidateDatasetSampler("QMnistDataset", sampler_));
  RETURN_IF_NOT_OK(
    ValidateStringValue("QMnistDataset", usage_, {"train", "test", "test10k", "test50k", "nist", "all"}));
  return Status::OK();
}

Status QMnistNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  if (compat_) {
    TensorShape scalar = TensorShape::CreateScalar();
    RETURN_IF_NOT_OK(
      schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));
  } else {
    RETURN_IF_NOT_OK(
      schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
  }

  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));

  auto op = std::make_shared<QMnistOp>(dataset_dir_, usage_, compat_, std::move(schema), std::move(sampler_rt),
                                       num_workers_, connector_que_size_);
  op->SetTotalRepeats(GetTotalRepeats());
  op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(op);

  return Status::OK();
}

// Get the shard id of node
Status QMnistNode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();

  return Status::OK();
}

// Get Dataset size
Status QMnistNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                  int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t num_rows, sample_size;
  RETURN_IF_NOT_OK(QMnistOp::CountTotalRows(dataset_dir_, usage_, &num_rows));
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

Status QMnistNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args, sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["num_parallel_workers"] = num_workers_;
  args["connector_queue_size"] = connector_que_size_;
  args["dataset_dir"] = dataset_dir_;
  args["usage"] = usage_;
  args["compat"] = compat_;
  if (cache_ != nullptr) {
    nlohmann::json cache_args;
    RETURN_IF_NOT_OK(cache_->to_json(&cache_args));
    args["cache"] = cache_args;
  }
  *out_json = args;
  return Status::OK();
}

#ifndef ENABLE_ANDROID
Status QMnistNode::from_json(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "num_parallel_workers", kQMnistNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "connector_queue_size", kQMnistNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "dataset_dir", kQMnistNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "usage", kQMnistNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "compat", kQMnistNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "sampler", kQMnistNode));
  std::string dataset_dir = json_obj["dataset_dir"];
  std::string usage = json_obj["usage"];
  bool compat = json_obj["compat"];
  std::shared_ptr<SamplerObj> sampler;
  RETURN_IF_NOT_OK(Serdes::ConstructSampler(json_obj["sampler"], &sampler));
  std::shared_ptr<DatasetCache> cache = nullptr;
  RETURN_IF_NOT_OK(DatasetCache::from_json(json_obj, &cache));
  *ds = std::make_shared<QMnistNode>(dataset_dir, usage, compat, sampler, cache);
  (void)(*ds)->SetNumWorkers(json_obj["num_parallel_workers"]);
  (void)(*ds)->SetConnectorQueueSize(json_obj["connector_queue_size"]);
  return Status::OK();
}
#endif
}  // namespace dataset
}  // namespace mindspore
