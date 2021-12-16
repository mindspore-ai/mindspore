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

#include "minddata/dataset/engine/ir/datasetops/source/omniglot_node.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/omniglot_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
OmniglotNode::OmniglotNode(const std::string &dataset_dir, bool background, bool decode,
                           const std::shared_ptr<SamplerObj> &sampler, const std::shared_ptr<DatasetCache> &cache)
    : MappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      background_(background),
      decode_(decode),
      sampler_(sampler) {}

std::shared_ptr<DatasetNode> OmniglotNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  auto node = std::make_shared<OmniglotNode>(dataset_dir_, background_, decode_, sampler, cache_);
  return node;
}

void OmniglotNode::Print(std::ostream &out) const {
  out << (Name() + "(path: " + dataset_dir_ + ", background: " + (background_ ? "true" : "false") +
          ", decode: " + (decode_ ? "true" : "false") + ")");
}

Status OmniglotNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("OmniglotDataset", dataset_dir_));
  RETURN_IF_NOT_OK(ValidateDatasetSampler("OmniglotDataset", sampler_));
  return Status::OK();
}

Status OmniglotNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  // Do internal Schema generation.
  // This arg is exist in OmniglotOp, but not externalized (in Python API).
  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));

  auto op = std::make_shared<OmniglotOp>(num_workers_, dataset_dir_, connector_que_size_, background_, decode_,
                                         std::move(schema), std::move(sampler_rt));
  op->SetTotalRepeats(GetTotalRepeats());
  op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(op);
  return Status::OK();
}

// Get the shard id of node
Status OmniglotNode::GetShardId(int32_t *shard_id) {
  RETURN_UNEXPECTED_IF_NULL(shard_id);
  *shard_id = sampler_->ShardId();
  return Status::OK();
}

// Get Dataset size
Status OmniglotNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                    int64_t *dataset_size) {
  RETURN_UNEXPECTED_IF_NULL(dataset_size);
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t sample_size, num_rows;
  Path dataset_path(dataset_dir_);
  if (background_) {
    dataset_path = dataset_path / "images_background";
  } else {
    dataset_path = dataset_path / "images_evaluation";
  }
  std::string path_str = dataset_path.ToString();

  RETURN_IF_NOT_OK(OmniglotOp::CountRowsAndClasses(path_str, &num_rows, nullptr));
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

Status OmniglotNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args, sampler_args;
  RETURN_UNEXPECTED_IF_NULL(out_json);
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["num_parallel_workers"] = num_workers_;
  args["dataset_dir"] = dataset_dir_;
  args["background"] = background_;
  args["decode"] = decode_;
  args["sampler"] = sampler_args;
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
