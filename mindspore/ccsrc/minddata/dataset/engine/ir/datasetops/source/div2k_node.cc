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
#include "minddata/dataset/engine/ir/datasetops/source/div2k_node.h"

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/div2k_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// Constructor for DIV2KNode
DIV2KNode::DIV2KNode(const std::string &dataset_dir, const std::string &usage, const std::string &downgrade,
                     int32_t scale, bool decode, std::shared_ptr<SamplerObj> sampler,
                     std::shared_ptr<DatasetCache> cache)
    : MappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      usage_(usage),
      downgrade_(downgrade),
      scale_(scale),
      decode_(decode),
      sampler_(sampler) {}

std::shared_ptr<DatasetNode> DIV2KNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  auto node = std::make_shared<DIV2KNode>(dataset_dir_, usage_, downgrade_, scale_, decode_, sampler, cache_);
  return node;
}

void DIV2KNode::Print(std::ostream &out) const {
  out << Name() + "(dataset dir:" + dataset_dir_;
  out << ", usage:" + usage_ << ", scale:" + std::to_string(scale_) << ", downgrade:" + downgrade_;
  if (sampler_ != nullptr) {
    out << ", sampler";
  }
  if (cache_ != nullptr) {
    out << ", cache";
  }
  out << ")";
}

Status DIV2KNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("DIV2KNode", dataset_dir_));
  RETURN_IF_NOT_OK(ValidateStringValue("DIV2KNode", usage_, {"train", "valid", "all"}));
  RETURN_IF_NOT_OK(ValidateStringValue("DIV2KNode", downgrade_, {"bicubic", "unknown", "mild", "difficult", "wild"}));
  RETURN_IF_NOT_OK(ValidateDatasetSampler("DIV2KNode", sampler_));

  std::set<int32_t> scale_arr = {2, 3, 4, 8};
  auto s_it = scale_arr.find(scale_);
  if (s_it == scale_arr.end()) {
    std::string err_msg = "DIV2KNode: " + std::to_string(scale_) + " does not match any mode in [2, 3, 4, 8].";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (scale_ == 8 && downgrade_ != "bicubic") {
    std::string err_msg = "DIV2KNode: scale equal to 8 is allowed only in bicubic downgrade.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  std::set<std::string> downgrade_2018 = {"mild", "difficult", "wild"};
  auto it = downgrade_2018.find(downgrade_);
  if (it != downgrade_2018.end() && scale_ != 4) {
    std::string err_msg = "DIV2KNode: " + downgrade_ + " downgrade requires scale equal to 4.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

// Function to build DIV2KOp for DIV2K
Status DIV2KNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("hr_image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("lr_image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 0, &scalar)));
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));

  auto div2k_op = std::make_shared<DIV2KOp>(num_workers_, dataset_dir_, usage_, downgrade_, scale_, decode_,
                                            connector_que_size_, std::move(schema), std::move(sampler_rt));
  div2k_op->SetTotalRepeats(GetTotalRepeats());
  div2k_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(div2k_op);
  return Status::OK();
}

// Get the shard id of node
Status DIV2KNode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();
  return Status::OK();
}

// Get Dataset size
Status DIV2KNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                 int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }

  int64_t num_rows, sample_size;
  RETURN_IF_NOT_OK(DIV2KOp::CountTotalRows(dataset_dir_, usage_, downgrade_, scale_, &num_rows));
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

Status DIV2KNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args, sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["num_parallel_workers"] = num_workers_;
  args["dataset_dir"] = dataset_dir_;
  args["usage"] = usage_;
  args["downgrade"] = downgrade_;
  args["scale"] = scale_;
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
