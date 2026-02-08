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

#include "minddata/dataset/data_source/operation/lj_speech_node.h"

#include <memory>
#include <string>
#include <utility>

#include "minddata/dataset/data_source/lj_speech_op.h"
#include "minddata/dataset/engine/serdes.h"

namespace mindspore {
namespace dataset {
// Constructor for LJSpeechNode.
LJSpeechNode::LJSpeechNode(const std::string &dataset_dir, std::shared_ptr<SamplerObj> sampler,
                           std::shared_ptr<DatasetCache> cache)
    : MappableSourceNode(std::move(cache)), dataset_dir_(dataset_dir), sampler_(sampler) {}

std::shared_ptr<DatasetNode> LJSpeechNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  auto node = std::make_shared<LJSpeechNode>(dataset_dir_, sampler, cache_);
  (void)node->SetNumWorkers(num_workers_);
  (void)node->SetConnectorQueueSize(connector_que_size_);
  return node;
}

void LJSpeechNode::Print(std::ostream &out) const {
  out << (Name() + "(cache: " + ((cache_ != nullptr) ? "true" : "false") + ")");
}

Status LJSpeechNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("LJSpeechNode", dataset_dir_));
  RETURN_IF_NOT_OK(ValidateDatasetSampler("LJSpeechNode", sampler_));
  return Status::OK();
}

// Function to build LJSpeechOp for LJSpeech.
Status LJSpeechNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("waveform", DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
  TensorShape sample_rate_scalar = TensorShape::CreateScalar();
  TensorShape trans_scalar = TensorShape::CreateScalar();
  TensorShape nom_trans_scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(schema->AddColumn(
    ColDescriptor("sample_rate", DataType(DataType::DE_INT32), TensorImpl::kFlexible, 0, &sample_rate_scalar)));
  RETURN_IF_NOT_OK(schema->AddColumn(
    ColDescriptor("transcription", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 0, &trans_scalar)));
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("normalized_transcription", DataType(DataType::DE_STRING),
                                                   TensorImpl::kFlexible, 0, &nom_trans_scalar)));
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));

  auto lj_speech_op = std::make_shared<LJSpeechOp>(dataset_dir_, num_workers_, connector_que_size_, std::move(schema),
                                                   std::move(sampler_rt));
  lj_speech_op->SetTotalRepeats(GetTotalRepeats());
  lj_speech_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(lj_speech_op);

  return Status::OK();
}

// Get the shard id of node.
Status LJSpeechNode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();
  return Status::OK();
}

// Get Dataset size.
Status LJSpeechNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                    int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }

  int64_t num_rows = 0, sample_size = 0;
  RETURN_IF_NOT_OK(LJSpeechOp::CountTotalRows(dataset_dir_, &num_rows));
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

Status LJSpeechNode::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args, sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["num_parallel_workers"] = num_workers_;
  args["connector_queue_size"] = connector_que_size_;
  args["dataset_dir"] = dataset_dir_;
  if (cache_ != nullptr) {
    nlohmann::json cache_args;
    RETURN_IF_NOT_OK(cache_->to_json(&cache_args));
    args["cache"] = cache_args;
  }
  *out_json = args;
  return Status::OK();
}

Status LJSpeechNode::from_json(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  RETURN_UNEXPECTED_IF_NULL(ds);
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "num_parallel_workers", kLJSpeechNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "connector_queue_size", kLJSpeechNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "dataset_dir", kLJSpeechNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "sampler", kLJSpeechNode));
  std::string dataset_dir = json_obj["dataset_dir"];
  std::shared_ptr<SamplerObj> sampler;
  RETURN_IF_NOT_OK(Serdes::ConstructSampler(json_obj["sampler"], &sampler));
  std::shared_ptr<DatasetCache> cache;
  RETURN_IF_NOT_OK(DatasetCache::from_json(json_obj, &cache));
  *ds = std::make_shared<LJSpeechNode>(dataset_dir, sampler, cache);
  (void)((*ds)->SetNumWorkers(json_obj["num_parallel_workers"]));
  (void)((*ds)->SetConnectorQueueSize(json_obj["connector_queue_size"]));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
