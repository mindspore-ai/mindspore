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
#include "minddata/dataset/engine/ir/datasetops/source/tedlium_node.h"

#include <utility>

#include "minddata/dataset/engine/datasetops/source/tedlium_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// Constructor for TedliumNode.
TedliumNode::TedliumNode(const std::string &dataset_dir, const std::string &release, const std::string &usage,
                         const std::string &extensions, const std::shared_ptr<SamplerObj> &sampler,
                         const std::shared_ptr<DatasetCache> &cache)
    : MappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      release_(release),
      extensions_(extensions),
      usage_(usage),
      sampler_(sampler) {}

std::shared_ptr<DatasetNode> TedliumNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  auto node = std::make_shared<TedliumNode>(dataset_dir_, release_, usage_, extensions_, sampler, cache_);
  (void)node->SetNumWorkers(num_workers_);
  (void)node->SetConnectorQueueSize(connector_que_size_);
  return node;
}

void TedliumNode::Print(std::ostream &out) const {
  out << (Name() + "(cache: " + ((cache_ != nullptr) ? "true" : "false") + ")");
}

Status ValidateExtensionsParam(const std::string &dataset_name, const std::string &extensions) {
  if (extensions != ".sph") {
    std::string err_msg = dataset_name + ": extension " + extensions + " is not supported.";
    MS_LOG(ERROR) << err_msg;
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

Status TedliumNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());

  RETURN_IF_NOT_OK(ValidateDatasetDirParam("TedliumNode", dataset_dir_));

  RETURN_IF_NOT_OK(ValidateStringValue("TedliumNode", release_, {"release1", "release2", "release3"}));

  RETURN_IF_NOT_OK(ValidateExtensionsParam("TedliumNode", extensions_));

  RETURN_IF_NOT_OK(ValidateDatasetSampler("TedliumNode", sampler_));

  if (release_ == "release1" || release_ == "release2") {
    RETURN_IF_NOT_OK(ValidateStringValue("TedliumNode", usage_, {"dev", "train", "test", "all"}));
  } else if (release_ == "release3") {
    RETURN_IF_NOT_OK(ValidateStringValue("TedliumNode", usage_, {"all"}));
  }
  return Status::OK();
}

Status TedliumNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("waveform", DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
  TensorShape sample_rate_scalar = TensorShape::CreateScalar();
  TensorShape trans_scalar = TensorShape::CreateScalar();
  TensorShape talk_id_scalar = TensorShape::CreateScalar();
  TensorShape speaker_id_scalar = TensorShape::CreateScalar();
  TensorShape identi_scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(schema->AddColumn(
    ColDescriptor("sample_rate", DataType(DataType::DE_INT32), TensorImpl::kFlexible, 0, &sample_rate_scalar)));
  RETURN_IF_NOT_OK(schema->AddColumn(
    ColDescriptor("transcript", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 0, &trans_scalar)));
  RETURN_IF_NOT_OK(schema->AddColumn(
    ColDescriptor("talk_id", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 0, &talk_id_scalar)));
  RETURN_IF_NOT_OK(schema->AddColumn(
    ColDescriptor("speaker_id", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 0, &speaker_id_scalar)));
  RETURN_IF_NOT_OK(schema->AddColumn(
    ColDescriptor("identifier", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 0, &identi_scalar)));

  // Argument that is not exposed to user in the API.
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));

  auto tedlium_op = std::make_shared<TedliumOp>(dataset_dir_, release_, usage_, extensions_, num_workers_,
                                                std::move(schema), std::move(sampler_rt), connector_que_size_);
  tedlium_op->SetTotalRepeats(GetTotalRepeats());
  tedlium_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(tedlium_op);
  return Status::OK();
}

Status TedliumNode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();
  return Status::OK();
}

Status TedliumNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                   int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t num_rows = 0, sample_size = 0;
  RETURN_IF_NOT_OK(TedliumOp::CountTotalRows(dataset_dir_, release_, usage_, extensions_, &num_rows));

  // give sampler the total number of files and check if num_samples is smaller.
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));
  sample_size = sampler_rt->CalculateNumSamples(num_rows);
  if (sample_size == -1) {
    RETURN_IF_NOT_OK(size_getter->DryRun(shared_from_this(), &sample_size));
  }
  *dataset_size = sample_size;
  // We cache dataset size so as to not duplicated run.
  dataset_size_ = *dataset_size;
  return Status::OK();
}

Status TedliumNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args, sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["num_parallel_workers"] = num_workers_;
  args["connector_queue_size"] = connector_que_size_;
  args["release"] = release_;
  args["dataset_dir"] = dataset_dir_;
  args["usage"] = usage_;
  args["extensions"] = extensions_;
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
