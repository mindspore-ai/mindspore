/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/ir/datasetops/concat_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/concat_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/serdes.h"
#endif
#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

// Function to build ConcatOp
ConcatNode::ConcatNode(const std::vector<std::shared_ptr<DatasetNode>> &datasets,
                       const std::shared_ptr<SamplerObj> &sampler,
                       const std::vector<std::pair<int, int>> &children_flag_and_nums,
                       const std::vector<std::pair<int, int>> &children_start_end_index)
    : sampler_(sampler),
      children_flag_and_nums_(children_flag_and_nums),
      children_start_end_index_(children_start_end_index) {
  nary_op_ = true;
  for (auto const &child : datasets) AddChild(child);
}

std::shared_ptr<DatasetNode> ConcatNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  // create an empty vector to copy a concat
  auto node = std::make_shared<ConcatNode>(std::vector<std::shared_ptr<DatasetNode>>(), sampler,
                                           children_flag_and_nums_, children_start_end_index_);
  return node;
}

void ConcatNode::Print(std::ostream &out) const { out << Name(); }

Status ConcatNode::ValidateParams() {
  constexpr size_t kMinChildrenSize = 2;
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  if (children_.size() < kMinChildrenSize) {
    std::string err_msg = "ConcatNode: concatenated datasets are not specified.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (find(children_.begin(), children_.end(), nullptr) != children_.end()) {
    std::string err_msg = "ConcatNode: concatenated datasets should not be null.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  // Either one of children_flag_and_nums_ or children_start_end_index_ should be non-empty.
  if ((children_flag_and_nums_.empty() && !children_start_end_index_.empty()) ||
      (!children_flag_and_nums_.empty() && children_start_end_index_.empty())) {
    std::string err_msg = "ConcatNode: children_flag_and_nums and children_start_end_index should be used together";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

// Get Dataset size
Status ConcatNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                  int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }

  // calculate the total size of all nodes
  int64_t total_dataset_size = 0;
  int64_t child_dataset_size = 0;
  for (int idx = 0; idx < children_.size(); idx++) {
    if (children_flag_and_nums_.empty() || children_flag_and_nums_[idx].second == 0) {
      RETURN_IF_NOT_OK(children_[idx]->GetDatasetSize(size_getter, false, &child_dataset_size));
      total_dataset_size += child_dataset_size;
    } else {
      total_dataset_size += children_flag_and_nums_[idx].second;
    }
  }

  // calculate the size of the shard
  int64_t shard_dataset_size = 0;
  std::shared_ptr<SamplerRT> sampler_rt_base = nullptr;
  if (sampler_) RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt_base));
  std::shared_ptr<DistributedSamplerRT> sampler_rt =
    sampler_ ? std::dynamic_pointer_cast<DistributedSamplerRT>(sampler_rt_base) : nullptr;
  if (sampler_rt != nullptr) {
    RETURN_IF_NOT_OK(sampler_rt->SetNumRowsInDataset(total_dataset_size));
    RETURN_IF_NOT_OK(sampler_rt->InitSampler());

    // (total_size % num_shards != 0) & shard_id >= (remainder) ? CalculateNumSamples()-1 : CalculateNumSamples()
    // example: 23 rows, 10 shards --> shard sizes = {3,3,3,2,2,2,2,2,2,2}
    if ((sampler_rt->GetNumSamples() % sampler_rt->GetDeviceNum()) >= 0 &&
        sampler_rt->GetDeviceID() >= (sampler_rt->GetNumSamples() % sampler_rt->GetDeviceNum())) {
      shard_dataset_size = sampler_rt->GetNumSamples() / sampler_rt->GetDeviceNum();
    } else {
      shard_dataset_size = sampler_rt->GetNumSamples() / sampler_rt->GetDeviceNum() + 1;
    }
  } else {
    shard_dataset_size = total_dataset_size;
  }

  *dataset_size = shard_dataset_size;
  dataset_size_ = *dataset_size;
  return Status::OK();
}

Status ConcatNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  std::shared_ptr<ConcatOp> op;
  if (children_flag_and_nums_.empty() || children_start_end_index_.empty()) {
    op = std::make_shared<ConcatOp>();
  } else {
    std::shared_ptr<SamplerRT> sampler_rt = nullptr;
    RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));
    op = std::make_shared<ConcatOp>(sampler_rt, children_flag_and_nums_, children_start_end_index_);
  }
  op->SetTotalRepeats(GetTotalRepeats());
  op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(op);

  return Status::OK();
}

// Visitor accepting method for IRNodePass
Status ConcatNode::Accept(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->Visit(shared_from_base<ConcatNode>(), modified);
}

// Visitor accepting method for IRNodePass
Status ConcatNode::AcceptAfter(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->VisitAfter(shared_from_base<ConcatNode>(), modified);
}

Status ConcatNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args, sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["children_flag_and_nums"] = children_flag_and_nums_;
  args["children_start_end_index"] = children_start_end_index_;
  *out_json = args;
  return Status::OK();
}

#ifndef ENABLE_ANDROID
Status ConcatNode::from_json(nlohmann::json json_obj, std::vector<std::shared_ptr<DatasetNode>> datasets,
                             std::shared_ptr<DatasetNode> *result) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("sampler") != json_obj.end(), "Failed to find sampler");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("children_flag_and_nums") != json_obj.end(),
                               "Failed to find children_flag_and_nums");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("children_start_end_index") != json_obj.end(),
                               "Failed to find children_start_end_index");
  std::shared_ptr<SamplerObj> sampler;
  RETURN_IF_NOT_OK(Serdes::ConstructSampler(json_obj["sampler"], &sampler));
  std::vector<std::pair<int, int>> children_flag_and_nums = json_obj["children_flag_and_nums"];
  std::vector<std::pair<int, int>> children_start_end_index = json_obj["children_start_end_index"];
  *result = std::make_shared<ConcatNode>(datasets, sampler, children_flag_and_nums, children_start_end_index);
  return Status::OK();
}
#endif
}  // namespace dataset
}  // namespace mindspore
