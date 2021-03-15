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

#include "minddata/dataset/engine/ir/datasetops/cache_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/engine/datasetops/cache_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
CacheNode::CacheNode(std::shared_ptr<DatasetNode> child, std::shared_ptr<SamplerObj> sampler,
                     std::shared_ptr<DatasetCache> cache)
    : DatasetNode(std::move(cache)), sampler_(sampler) {
  this->AddChild(child);
}

void CacheNode::Print(std::ostream &out) const { out << Name(); }

std::shared_ptr<DatasetNode> CacheNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  auto node = std::make_shared<CacheNode>(nullptr, sampler, cache_);
  return node;
}

Status CacheNode::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateDatasetSampler("CacheNode", sampler_));
  return Status::OK();
}

Status CacheNode::Build(std::vector<std::shared_ptr<DatasetOp>> *node_ops) {
  CHECK_FAIL_RETURN_UNEXPECTED(cache_ != nullptr,
                               "Internal error. Attempt to create a cache node without cache client.");
  RETURN_IF_NOT_OK(cache_->Build());
  std::shared_ptr<DatasetOp> cache_op = nullptr;
  RETURN_IF_NOT_OK(cache_->CreateCacheOp(num_workers_, &cache_op));
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));
  cache_op->SetSampler(sampler_rt);
  cache_op->set_total_repeats(GetTotalRepeats());
  cache_op->set_num_repeats_per_epoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(cache_op);
  return Status::OK();
}

// Visitor accepting method for IRNodePass
Status CacheNode::Accept(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->Visit(shared_from_base<CacheNode>(), modified);
}

// Visitor accepting method for IRNodePass
Status CacheNode::AcceptAfter(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->VisitAfter(shared_from_base<CacheNode>(), modified);
}
}  // namespace dataset
}  // namespace mindspore
