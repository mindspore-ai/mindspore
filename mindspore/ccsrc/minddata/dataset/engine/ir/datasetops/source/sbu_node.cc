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

#include "minddata/dataset/engine/ir/datasetops/source/sbu_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/sbu_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
SBUNode::SBUNode(const std::string &dataset_dir, bool decode, const std::shared_ptr<SamplerObj> &sampler,
                 const std::shared_ptr<DatasetCache> &cache)
    : MappableSourceNode(std::move(cache)), dataset_dir_(dataset_dir), decode_(decode), sampler_(sampler) {}

std::shared_ptr<DatasetNode> SBUNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  auto node = std::make_shared<SBUNode>(dataset_dir_, decode_, sampler, cache_);
  return node;
}

void SBUNode::Print(std::ostream &out) const {
  out << (Name() + "(dataset dir: " + dataset_dir_ + ", decode: " + (decode_ ? "true" : "false") +
          ", cache: " + ((cache_ != nullptr) ? "true" : "false") + ")");
}

Status SBUNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("SBUNode", dataset_dir_));
  RETURN_IF_NOT_OK(ValidateDatasetSampler("SBUNode", sampler_));

  Path root_dir(dataset_dir_);

  Path url_path = root_dir / Path("SBU_captioned_photo_dataset_urls.txt");
  Path caption_path = root_dir / Path("SBU_captioned_photo_dataset_captions.txt");
  Path image_path = root_dir / Path("sbu_images");

  RETURN_IF_NOT_OK(ValidateDatasetFilesParam("SBUNode", {url_path.ToString()}));
  RETURN_IF_NOT_OK(ValidateDatasetFilesParam("SBUNode", {caption_path.ToString()}));
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("SBUNode", {image_path.ToString()}));

  return Status::OK();
}

Status SBUNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("caption", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));

  auto op = std::make_shared<SBUOp>(dataset_dir_, decode_, std::move(schema), std::move(sampler_rt), num_workers_,
                                    connector_que_size_);
  op->SetTotalRepeats(GetTotalRepeats());
  op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(op);

  return Status::OK();
}

// Get the shard id of node
Status SBUNode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();

  return Status::OK();
}

// Get Dataset size
Status SBUNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                               int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t num_rows, sample_size;
  RETURN_IF_NOT_OK(SBUOp::CountTotalRows(dataset_dir_, &num_rows));
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

Status SBUNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args, sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["num_parallel_workers"] = num_workers_;
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
}  // namespace dataset
}  // namespace mindspore
