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

#include "minddata/dataset/engine/opt/post/repeat_pass.h"

#include <memory>

#include "minddata/dataset/engine/ir/datasetops/cache_lookup_node.h"
#include "minddata/dataset/engine/ir/datasetops/cache_merge_node.h"
#include "minddata/dataset/engine/ir/datasetops/cache_node.h"
#include "minddata/dataset/engine/ir/datasetops/epoch_ctrl_node.h"
#include "minddata/dataset/engine/ir/datasetops/repeat_node.h"
#include "minddata/dataset/engine/ir/datasetops/transfer_node.h"

namespace mindspore {
namespace dataset {

RepeatPass::RepeatPass()
    : num_repeats_(1), num_epochs_(1), is_merge_(false), is_cached_(false), cache_lookup_(nullptr) {}

// Identifies the subtree below this node as being in a repeated path of the tree.
Status RepeatPass::Visit(std::shared_ptr<RepeatNode> node, bool *const modified) {
  // If this is an infinite repeat under infinite repeat/epoch, adjust current num_repeats_.
  // Otherwise, after multiplication it would become positive and this repeat wouldn't run infinitely.
  if (node->Count() == DatasetOp::kInfiniteRepeat && num_repeats_ < 0) {
    num_repeats_ = -num_repeats_;
  }
  // This RepeatOp and its descendent nodes should be repeated for another num_repeats() times.
  //
  // Consider this example:
  // tfreader --> map --> repeat(2) --> epoch ctrl(3)
  // num_repeats_ is originally 3, after this repeat(2), num_repeats_ becomes 6 (2*3),
  // meaning repeat op should be set to read 6 times (2*3), do does map op and tfreader op.
  //
  // Another example:
  // tfreader --> repeat1(3) --> map --> repeat2(2) --> epoch ctrl(4)
  // num_repeats_ is originally 4, after repeat2(2), num_repeats_ becomes 8 (2*4),
  // meaning repeat2 and map op should be set to read 8 times (2*4).
  // Then, after repeat1(3), num_repeats_ becomes 24 (3*2*4), meaning repeat1 and tfreader op should repeat 24 times.
  num_repeats_ *= node->Count();
  return Status::OK();
}

// Identifies the subtree below this node as being in a repeated path of the tree.
Status RepeatPass::Visit(std::shared_ptr<EpochCtrlNode> node, bool *const modified) {
  // Get the total number of epochs from the EpochCtrlOp parameter
  num_epochs_ = node->Count();
  // Every node below this EpochCtrlOp should be repeated for num_epochs_ times.
  // For example: tfreader --> epoch ctrl(3)
  // num_repeats_ is originally 1 (default initialization), after this epoch ctrl(3), num_repeats_ becomes 3 (1*3),
  // meaning epoch ctrl op should be set to read 3 times (1*3), so does tfreader op.
  num_repeats_ *= num_epochs_;
  return Status::OK();
}

#ifndef ENABLE_ANDROID
// Identifies the subtree below this node as being in a cache merge path
Status RepeatPass::Visit(std::shared_ptr<CacheMergeNode> node, bool *const modified) {
  // Turn on the flag that we're under a merge op
  is_merge_ = true;
  return Status::OK();
}

// Identifies the subtree below this node as being cached
Status RepeatPass::Visit(std::shared_ptr<CacheNode> node, bool *const modified) {
  // Turn on the flag that we're under a merge op
  is_cached_ = true;
  return Status::OK();
}
#endif

// Hooks up any identified eoe nodes under this repeat.
Status RepeatPass::VisitAfter(std::shared_ptr<RepeatNode> node, bool *const modified) {
  // We are a repeat op in the descendant tree of a merge op, then we take the saved lookup up
  // and set its total repeats. It is important that the op is removed from the save area,
  // because the merge op above us may also take action on it later for a different case when
  // there is no repeat in the merge leg.
  if (is_merge_ && cache_lookup_) {
    cache_lookup_->SetTotalRepeats(num_repeats_);
    cache_lookup_->SetNumEpochs(num_epochs_);
    cache_lookup_.reset();
  }

  if (is_cached_) {
    AddToCachedNodeStack(node);
  }
  node->SetTotalRepeats(num_repeats_);
  node->SetNumEpochs(num_epochs_);
  // We finish the walk of this RepeatOp's descendent nodes.
  // The total repeats of nodes above this Repeat(n) have nothing to do with this RepeatOp's parameter n.
  // But num_repeats_ has been multiplied by n during this Repeat(n)'s PreRunOnNode,
  // so we divide num_repeats_ by n to be able to correctly set total repeats for nodes above this RepeatOp.
  num_repeats_ /= node->Count();
  return Status::OK();
}

// Hooks up any identified eoe nodes under this repeat.
Status RepeatPass::VisitAfter(std::shared_ptr<EpochCtrlNode> node, bool *const modified) {
  node->SetTotalRepeats(num_repeats_);
  node->SetNumEpochs(num_epochs_);
  // We finish the walk of this EpochCtrl's descendent nodes.
  num_repeats_ /= node->Count();
  return Status::OK();
}

// All operators have a flag that might be set related to the repeat and any leaf nodes need to be set up
// for use with a controlling repeat above it.
Status RepeatPass::VisitAfter(std::shared_ptr<DatasetNode> node, bool *const modified) {
  // If we are under a cache op, then save ourselves to the cached op stack.
  if (is_cached_) {
    AddToCachedNodeStack(node);
  }
  // Set total repeats and total epochs for the node
  node->SetTotalRepeats(num_repeats_);
  node->SetNumEpochs(num_epochs_);
  return Status::OK();
}

#ifndef ENABLE_ANDROID
// CacheOp removes previous leaf ops and replaces them with itself
Status RepeatPass::VisitAfter(std::shared_ptr<CacheNode> node, bool *const modified) {
  is_cached_ = false;

  // if we are a cache within a repeat path of the tree, then adjust the total repeats and total epochs for cached ops.
  // So that those cached nodes become 1-time use (up to eoe), never repeated.  Instead
  // the repeating behaviours shall be invoked against the cache op.
  std::shared_ptr<DatasetNode> cached_node = PopFromCachedNodeStack();
  while (cached_node != nullptr) {
    int32_t cached_op_total_repeats = cached_node->GetTotalRepeats() / num_repeats_;
    cached_node->SetTotalRepeats(cached_op_total_repeats);
    // Cached ops will only be executed on the first epoch, therefore, num_epochs_ = 1
    cached_node->SetNumEpochs(1);
    cached_node = PopFromCachedNodeStack();
  }

  node->SetTotalRepeats(num_repeats_);
  node->SetNumEpochs(num_epochs_);
  return Status::OK();
}

// Turns off the tracking for operations under merge op
Status RepeatPass::VisitAfter(std::shared_ptr<CacheMergeNode> node, bool *const modified) {
  // If there was not any repeat in the merge cache miss leg, then the cache_lookup
  // would not have been consumed yet.  In that case, we need to set its total repeats for it.
  if (cache_lookup_) {
    cache_lookup_->SetTotalRepeats(num_repeats_);
    cache_lookup_->SetNumEpochs(num_epochs_);
  }
  node->SetTotalRepeats(num_repeats_);
  node->SetNumEpochs(num_epochs_);
  cache_lookup_.reset();  // If we are not repeated then the saved lookup is no longer needed or used
  is_merge_ = false;
  return Status::OK();
}

// Saves the lookup up in case it needs to be referenced by a repeat
Status RepeatPass::VisitAfter(std::shared_ptr<CacheLookupNode> node, bool *const modified) {
  if (!node->IsLeaf()) {
    // By definition, the CacheLookup must be a leaf op.  Make that clear here.
    RETURN_STATUS_UNEXPECTED("CacheLookupOp must be a leaf node!");
  }

  // save the lookup op.  There could be a repeat in the cache miss leg of the merge op, in which case we
  // may still need to be flagged as a repeating leaf.  We can't decide that here though, so save ourself
  // into the pass so that the decision can be made during the processing of the cache miss leg of the merge.
  // Further, if there's a repeat above the merge but no repeat in the cache miss leg, then the merge op will
  // add the lookup to the eoe stack
  cache_lookup_ = std::static_pointer_cast<DatasetNode>(node);

  return Status::OK();
}
#endif

Status RepeatPass::VisitAfter(std::shared_ptr<TransferNode> node, bool *const modified) {
  // Set total repeats and total epochs for the TransferNode
  node->SetTotalRepeats(num_epochs_);
  node->SetNumEpochs(num_epochs_);
  return Status::OK();
}

// Adds an operator to the cached operator stack save area
void RepeatPass::AddToCachedNodeStack(std::shared_ptr<DatasetNode> node) { cached_node_stacks_.push(node); }

// Pops an operator from the cached operator stack save area
std::shared_ptr<DatasetNode> RepeatPass::PopFromCachedNodeStack() {
  std::shared_ptr<DatasetNode> top_node = nullptr;
  if (!cached_node_stacks_.empty()) {
    top_node = cached_node_stacks_.top();
    cached_node_stacks_.pop();
  }
  return top_node;
}
}  // namespace dataset
}  // namespace mindspore
