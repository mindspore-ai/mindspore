/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/ir/datasetops/source/celeba_node.h"

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/celeba_op.h"
#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

// Constructor for CelebANode
CelebANode::CelebANode(const std::string &dataset_dir, const std::string &usage,
                       const std::shared_ptr<SamplerObj> &sampler, const bool &decode,
                       const std::set<std::string> &extensions, const std::shared_ptr<DatasetCache> &cache)
    : MappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      usage_(usage),
      sampler_(sampler),
      decode_(decode),
      extensions_(extensions) {}

std::shared_ptr<DatasetNode> CelebANode::Copy() {
  std::shared_ptr<SamplerObj> sampler = sampler_ == nullptr ? nullptr : sampler_->Copy();
  auto node = std::make_shared<CelebANode>(dataset_dir_, usage_, sampler, decode_, extensions_, cache_);
  return node;
}

void CelebANode::Print(std::ostream &out) const {
  out << Name() + "(cache:" + ((cache_ != nullptr) ? "true" : "false") + ")";
}

Status CelebANode::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("CelebANode", dataset_dir_));

  RETURN_IF_NOT_OK(ValidateDatasetSampler("CelebANode", sampler_));

  RETURN_IF_NOT_OK(ValidateStringValue("CelebANode", usage_, {"all", "train", "valid", "test"}));

  return Status::OK();
}

// Function to build CelebANode
std::vector<std::shared_ptr<DatasetOp>> CelebANode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  RETURN_EMPTY_IF_ERROR(
    schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  // label is like this:0 1 0 0 1......
  RETURN_EMPTY_IF_ERROR(
    schema->AddColumn(ColDescriptor("attr", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));

  build_status = AddCacheOp(&node_ops);  // remove me after changing return val of Build()
  RETURN_EMPTY_IF_ERROR(build_status);

  node_ops.push_back(std::make_shared<CelebAOp>(num_workers_, rows_per_buffer_, dataset_dir_, connector_que_size_,
                                                decode_, usage_, extensions_, std::move(schema),
                                                std::move(sampler_->Build())));

  return node_ops;
}

// Get the shard id of node
Status CelebANode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();

  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
