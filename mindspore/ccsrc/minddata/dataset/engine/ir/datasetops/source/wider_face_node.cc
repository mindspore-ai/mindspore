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

#include "minddata/dataset/engine/ir/datasetops/source/wider_face_node.h"

#include <algorithm>
#include <utility>

#include "minddata/dataset/engine/datasetops/source/wider_face_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// Constructor for WIDERFaceNode.
WIDERFaceNode::WIDERFaceNode(const std::string &dataset_dir, const std::string &usage, const bool &decode,
                             const std::shared_ptr<SamplerObj> &sampler, const std::shared_ptr<DatasetCache> &cache)
    : MappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      usage_(usage),
      sampler_(sampler),
      decode_(decode) {}

std::shared_ptr<DatasetNode> WIDERFaceNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  auto node = std::make_shared<WIDERFaceNode>(dataset_dir_, usage_, decode_, sampler, cache_);
  (void)node->SetNumWorkers(num_workers_);
  (void)node->SetConnectorQueueSize(connector_que_size_);
  return node;
}

void WIDERFaceNode::Print(std::ostream &out) const { out << Name(); }

Status WIDERFaceNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("WIDERFaceDataset", dataset_dir_));
  RETURN_IF_NOT_OK(ValidateDatasetSampler("WIDERFaceDataset", sampler_));
  RETURN_IF_NOT_OK(ValidateStringValue("WIDERFaceDataset", usage_, {"all", "train", "valid", "test"}));
  return Status::OK();
}

// Function to build WIDERFaceNode.
Status WIDERFaceNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  if (usage_ == "all" || usage_ == "train" || usage_ == "valid") {
    RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
    RETURN_IF_NOT_OK(
      schema->AddColumn(ColDescriptor(std::string("bbox"), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
    RETURN_IF_NOT_OK(
      schema->AddColumn(ColDescriptor(std::string("blur"), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
    RETURN_IF_NOT_OK(schema->AddColumn(
      ColDescriptor(std::string("expression"), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
    RETURN_IF_NOT_OK(schema->AddColumn(
      ColDescriptor(std::string("illumination"), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
    RETURN_IF_NOT_OK(schema->AddColumn(
      ColDescriptor(std::string("occlusion"), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
    RETURN_IF_NOT_OK(
      schema->AddColumn(ColDescriptor(std::string("pose"), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
    RETURN_IF_NOT_OK(schema->AddColumn(
      ColDescriptor(std::string("invalid"), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
  } else if (usage_ == "test") {
    RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  }
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));
  auto wider_face_op = std::make_shared<WIDERFaceOp>(dataset_dir_, usage_, num_workers_, connector_que_size_, decode_,
                                                     std::move(schema), std::move(sampler_rt));
  wider_face_op->SetTotalRepeats(GetTotalRepeats());
  wider_face_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(wider_face_op);
  return Status::OK();
}

// Get the shard id of node.
Status WIDERFaceNode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();
  return Status::OK();
}

// Get Dataset size.
Status WIDERFaceNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                     int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t num_rows = 0, sample_size;
  std::vector<std::shared_ptr<DatasetOp>> ops;
  RETURN_IF_NOT_OK(Build(&ops));
  CHECK_FAIL_RETURN_UNEXPECTED(!ops.empty(), "Unable to build WIDERFaceOp.");
  auto op = std::dynamic_pointer_cast<WIDERFaceOp>(ops.front());
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

Status WIDERFaceNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args, sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["num_parallel_workers"] = num_workers_;
  args["connector_queue_size"] = connector_que_size_;
  args["dataset_dir"] = dataset_dir_;
  args["decode"] = decode_;
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
