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

#include "minddata/dataset/engine/ir/datasetops/cache_lookup_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
CacheLookupNode::CacheLookupNode(std::shared_ptr<DatasetNode> child, std::shared_ptr<SamplerObj> sampler,
                                 std::shared_ptr<DatasetCache> cache)
    : DatasetNode(std::move(cache)), sampler_(sampler), lookup_op_(nullptr), lookup_node_copy_(nullptr) {
  this->AddChild(child);
}

void CacheLookupNode::Print(std::ostream &out) const { out << Name(); }

std::shared_ptr<DatasetNode> CacheLookupNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  auto node = std::make_shared<CacheLookupNode>(nullptr, sampler, cache_);
  lookup_node_copy_ = node;
  return node;
}

Status CacheLookupNode::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateDatasetSampler("CacheNode", sampler_));
  return Status::OK();
}

Status CacheLookupNode::Build(std::vector<std::shared_ptr<DatasetOp>> *node_ops) {
  CHECK_FAIL_RETURN_UNEXPECTED(cache_ != nullptr,
                               "Internal error. Attempt to create a cache lookup node without cache client.");
  RETURN_IF_NOT_OK(cache_->Build());
  RETURN_IF_NOT_OK(cache_->CreateCacheLookupOp(num_workers_, &lookup_op_, sampler_));
  lookup_op_->set_total_repeats(GetTotalRepeats());
  lookup_op_->set_num_repeats_per_epoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(lookup_op_);
  return Status::OK();
}

// Visitor accepting method for IRNodePass
Status CacheLookupNode::Accept(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->Visit(shared_from_base<CacheLookupNode>(), modified);
}

// Visitor accepting method for IRNodePass
Status CacheLookupNode::AcceptAfter(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->VisitAfter(shared_from_base<CacheLookupNode>(), modified);
}

std::shared_ptr<SamplerObj> CacheLookupNode::SamplerCopy() {
  // CacheLookupNode should already been copied, so we just return it here
  return std::static_pointer_cast<SamplerObj>(lookup_node_copy_);
}

Status CacheLookupNode::SamplerBuild(std::shared_ptr<SamplerRT> *const out) {
  // Runtime cache lookup op should already been built, so we just return it here
  auto lookup_op = std::dynamic_pointer_cast<CacheLookupOp>(lookup_op_);
  *out = std::shared_ptr<SamplerRT>(lookup_op);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
