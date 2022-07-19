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
#include "minddata/dataset/engine/ir/datasetops/source/speech_commands_node.h"

#include <utility>

#include "minddata/dataset/engine/datasetops/source/speech_commands_op.h"

namespace mindspore {
namespace dataset {
SpeechCommandsNode::SpeechCommandsNode(const std::string &dataset_dir, const std::string &usage,
                                       std::shared_ptr<SamplerObj> sampler, std::shared_ptr<DatasetCache> cache)
    : MappableSourceNode(std::move(cache)), dataset_dir_(dataset_dir), usage_(usage), sampler_(sampler) {}

std::shared_ptr<DatasetNode> SpeechCommandsNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  auto node = std::make_shared<SpeechCommandsNode>(dataset_dir_, usage_, sampler, cache_);
  (void)node->SetNumWorkers(num_workers_);
  (void)node->SetConnectorQueueSize(connector_que_size_);
  return node;
}

void SpeechCommandsNode::Print(std::ostream &out) const {
  out << (Name() + "(cache: " + ((cache_ != nullptr) ? "true" : "false") + ")");
}

Status SpeechCommandsNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("SpeechCommandsNode", dataset_dir_));
  RETURN_IF_NOT_OK(ValidateDatasetSampler("SpeechCommandsNode", sampler_));
  RETURN_IF_NOT_OK(ValidateStringValue("SpeechCommandsNode", usage_, {"train", "valid", "test", "all"}));
  return Status::OK();
}

Status SpeechCommandsNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("waveform", DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
  TensorShape sample_rate_scalar = TensorShape::CreateScalar();
  TensorShape label_scalar = TensorShape::CreateScalar();
  TensorShape speaker_id_scalar = TensorShape::CreateScalar();
  TensorShape utterance_number_scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(schema->AddColumn(
    ColDescriptor("sample_rate", DataType(DataType::DE_INT32), TensorImpl::kFlexible, 0, &sample_rate_scalar)));
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 0, &label_scalar)));
  RETURN_IF_NOT_OK(schema->AddColumn(
    ColDescriptor("speaker_id", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 0, &speaker_id_scalar)));
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("utterance_number", DataType(DataType::DE_INT32),
                                                   TensorImpl::kFlexible, 0, &utterance_number_scalar)));
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));

  auto speech_commands_op = std::make_shared<SpeechCommandsOp>(dataset_dir_, usage_, num_workers_, connector_que_size_,
                                                               std::move(schema), std::move(sampler_rt));
  speech_commands_op->SetTotalRepeats(GetTotalRepeats());
  speech_commands_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(speech_commands_op);
  return Status::OK();
}

Status SpeechCommandsNode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();
  return Status::OK();
}

Status SpeechCommandsNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                          int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t sample_size, num_rows;
  std::vector<std::shared_ptr<DatasetOp>> ops;
  RETURN_IF_NOT_OK(Build(&ops));
  CHECK_FAIL_RETURN_UNEXPECTED(!ops.empty(), "Unable to build SpeechCommandsOp.");
  auto op = std::dynamic_pointer_cast<SpeechCommandsOp>(ops.front());
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

Status SpeechCommandsNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args, sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["usage"] = usage_;
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
}  // namespace dataset
}  // namespace mindspore
