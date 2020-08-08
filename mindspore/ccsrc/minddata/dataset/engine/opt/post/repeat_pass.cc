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

#include <memory>
#include "minddata/dataset/engine/opt/post/repeat_pass.h"
#include "minddata/dataset/engine/datasetops/repeat_op.h"
#include "minddata/dataset/engine/datasetops/cache_op.h"
#include "minddata/dataset/engine/datasetops/cache_lookup_op.h"
#include "minddata/dataset/engine/datasetops/cache_merge_op.h"
#include "minddata/dataset/engine/datasetops/device_queue_op.h"
#include "minddata/dataset/engine/datasetops/epoch_ctrl_op.h"

namespace mindspore {
namespace dataset {

RepeatPass::RepeatPass()
    : is_repeated_(false),
      nested_repeats_(0),
      num_repeats_(1),
      num_epochs_(1),
      is_merge_(false),
      is_cached_(false),
      cache_lookup_(nullptr) {}

// Identifies the subtree below this node as being in a repeated path of the tree.
Status RepeatPass::PreRunOnNode(std::shared_ptr<RepeatOp> node, bool *modified) {
  // Create a new stack for eoe operators and push onto our stack of stacks.
  std::unique_ptr<op_stack> new_stack = std::make_unique<op_stack>();
  eoe_op_stacks_.push(std::move(new_stack));
  // If we are already repeated, then this is a nested repeat.
  if (is_repeated_) {
    nested_repeats_++;
  }
  is_repeated_ = true;

  // If this is an infinite repeat under infinite repeat/epoch, adjust current num_repeats_.
  // Otherwise, after multiplication it would become positive and this repeat wouldn't run infinitely.
  if (node->num_repeats() == DatasetOp::kInfiniteRepeat && num_repeats_ < 0) {
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
  num_repeats_ *= node->num_repeats();
  return Status::OK();
}

// Identifies the subtree below this node as being in a repeated path of the tree.
Status RepeatPass::PreRunOnNode(std::shared_ptr<EpochCtrlOp> node, bool *modified) {
  // EpochCtrl is derived from RepeatOp.  Generally it should do the identical setup
  // that RepeatOp does.  However, epoch control is actually simpler because it can
  // only exist as the root node so it doesn't need all the nested code.
  // Create a new stack for eoe operators and push onto our stack of stacks.
  std::unique_ptr<op_stack> new_stack = std::make_unique<op_stack>();
  eoe_op_stacks_.push(std::move(new_stack));
  is_repeated_ = true;
  // Get the total number of epochs from the EpochCtrlOp parameter
  num_epochs_ = node->num_repeats();
  // Every node below this EpochCtrlOp should be repeated for num_epochs_ times.
  // For example: tfreader --> epoch ctrl(3)
  // num_repeats_ is originally 1 (default initialization), after this epoch ctrl(3), num_repeats_ becomes 3 (1*3),
  // meaning epoch ctrl op should be set to read 3 times (1*3), so does tfreader op.
  num_repeats_ *= num_epochs_;
  return Status::OK();
}

// Identifies the subtree below this node as being in a cache merge path
Status RepeatPass::PreRunOnNode(std::shared_ptr<CacheMergeOp> node, bool *modified) {
  // Turn on the flag that we're under a merge op
  is_merge_ = true;
  return Status::OK();
}

// Identifies the subtree below this node as being cached
Status RepeatPass::PreRunOnNode(std::shared_ptr<CacheOp> node, bool *modified) {
  // Turn on the flag that we're under a merge op
  is_cached_ = true;
  return Status::OK();
}

// Hooks up any identified eoe nodes under this repeat.
Status RepeatPass::RunOnNode(std::shared_ptr<RepeatOp> node, bool *modified) {
  // Pop the leaf ops from the save-area stack and add them to the repeat op's eoe node tracking
  std::shared_ptr<DatasetOp> leaf_op = PopFromEOEOpStack();

  while (leaf_op != nullptr) {
    node->AddToEoeList(leaf_op);
    leaf_op = PopFromEOEOpStack();
  }

  // At this point, we are done with the save area stack.  It's a unique pointer to an empty stack
  // at this time, so we can pop it to get rid of it.
  op_stack *current_stack = eoe_op_stacks_.top().get();
  if (!current_stack->empty()) {
    RETURN_STATUS_UNEXPECTED("The eoe op stack should be empty right now!");
  }
  eoe_op_stacks_.pop();

  // We are a repeat op in the descendant tree of a merge op, then we take the saved lookup up
  // and add it to the list of eoe/leaf ops for the repeat.  It is important that the op is removed
  // from the save area, because the merge op above us may also take action on it later for a different
  // case when there is no repeat in the merge leg.
  if (is_merge_ && cache_lookup_) {
    cache_lookup_->set_total_repeats(num_repeats_);
    cache_lookup_->set_num_repeats_per_epoch(num_repeats_ / num_epochs_);
    node->AddToEoeList(std::move(cache_lookup_));
  }

  // If we are a nested repeat, then we add ourself to the repeat stack for the next one above us.
  // A nested repeat acts like an eoe/leaf for the repeat in the ascendant tree.
  if (nested_repeats_ > 0) {
    AddToEOEOpStack(node);
    nested_repeats_--;
  } else {
    // If we are not nested, or we were the top-most repeat, now we clear the flag
    if (nested_repeats_ != 0) {
      RETURN_STATUS_UNEXPECTED("Nested repeat counter cannot be negative!");
    }
    is_repeated_ = false;
  }
  if (is_cached_) {
    AddToCachedOpStack(node);
  }
  node->set_total_repeats(num_repeats_);
  node->set_num_repeats_per_epoch(num_repeats_ / num_epochs_);
  // We finish the walk of this RepeatOp's descendent nodes.
  // The total repeats of nodes above this Repeat(n) have nothing to do with this RepeatOp's parameter n.
  // But num_repeats_ has been multiplied by n during this Repeat(n)'s PreRunOnNode,
  // so we devide num_repeats_ by n to be able to correctly set total repeats for nodes above this RepeatOp.
  num_repeats_ /= node->num_repeats();
  return Status::OK();
}

// Hooks up any identified eoe nodes under this repeat.
Status RepeatPass::RunOnNode(std::shared_ptr<EpochCtrlOp> node, bool *modified) {
  // Pop the leaf ops from the save-area stack and add them to the eoe node tracking
  std::shared_ptr<DatasetOp> leaf_op = PopFromEOEOpStack();
  while (leaf_op != nullptr) {
    node->AddToEoeList(leaf_op);
    leaf_op = PopFromEOEOpStack();
  }
  is_repeated_ = false;
  node->set_total_repeats(num_repeats_);
  node->set_num_repeats_per_epoch(num_repeats_ / num_epochs_);
  // We finish the walk of this EpochCtrl's descendent nodes.
  num_repeats_ /= node->num_repeats();
  return Status::OK();
}

// CacheOp removes previous leaf ops and replaces them with itself
Status RepeatPass::RunOnNode(std::shared_ptr<CacheOp> node, bool *modified) {
  is_cached_ = false;
  if (is_repeated_) {
    // if we are a cache within a repeat path of the tree, then there will be
    // eoe-generating ops in the eoe op stack in the tree.  They are flagged as such so that the
    // repeat or epoch ctrl operators can work with them for repeat activity during runtime.
    // However, since a cache is present:
    // - unflag those ops as being repeated ops
    // - remove them from the eoe op stack so that repeat op above in the tree won't know about them
    // - add ourself (the cache op), as an eoe op
    // We do this so that those old leafs become 1-time use (up to eoe), never repeated.  Instead
    // the repeating behaviours shall be invoked against the cache op.
    std::shared_ptr<DatasetOp> leaf_op = PopFromEOEOpStack();
    while (leaf_op != nullptr) {
      leaf_op = PopFromEOEOpStack();
    }
    AddToEOEOpStack(std::static_pointer_cast<DatasetOp>(node));

    // adjust the total epochs and total repeats for ops under this cache op
    std::shared_ptr<DatasetOp> cached_op = PopFromCachedOpStack();
    while (cached_op != nullptr) {
      int32_t cached_op_total_repeats = cached_op->op_total_repeats() / num_repeats_;
      cached_op->set_total_repeats(cached_op_total_repeats);
      // Cached ops will only be executed on the first epoch, therefore, num_epochs_ = 1
      cached_op->set_num_repeats_per_epoch(cached_op_total_repeats);
      cached_op = PopFromCachedOpStack();
    }
  }

  node->set_total_repeats(num_repeats_);
  node->set_num_repeats_per_epoch(num_repeats_ / num_epochs_);
  return Status::OK();
}

// All operators have a flag that might be set related to the repeat and any leaf nodes need to be set up
// for use with a controlling repeat above it.
Status RepeatPass::RunOnNode(std::shared_ptr<DatasetOp> node, bool *modified) {
  // If we are in a repeat path, then set our repeated flag
  if (is_repeated_) {
    // if we are a leaf node then save ourself in a stack for the repeat operator above us
    if (node->IsLeaf()) {
      AddToEOEOpStack(node);
    }
  }
  if (is_cached_) {
    AddToCachedOpStack(node);
  }
  // Set total repeats and total epochs for the node
  node->set_total_repeats(num_repeats_);
  node->set_num_repeats_per_epoch(num_repeats_ / num_epochs_);
  return Status::OK();
}

// Turns off the tracking for operations under merge op
Status RepeatPass::RunOnNode(std::shared_ptr<CacheMergeOp> node, bool *modified) {
  // Setting the flag is needed since we didn't call the base class DatasetOp version
  if (is_repeated_) {
    // If there was not any repeat in the merge cache miss leg, then the cache_lookup
    // would not have been consumed yet.  In that case, we need to assign it to the upper repeat eoe stack
    if (cache_lookup_) {
      cache_lookup_->set_total_repeats(num_repeats_);
      node->set_num_repeats_per_epoch(num_repeats_ / num_epochs_);
      AddToEOEOpStack(std::move(cache_lookup_));
    }
  }
  node->set_total_repeats(num_repeats_);
  node->set_num_repeats_per_epoch(num_repeats_ / num_epochs_);
  cache_lookup_.reset();  // If we are not repeated then the saved lookup is no longer needed or used
  is_merge_ = false;
  return Status::OK();
}

// Saves the lookup up in case it needs to be referenced by a repeat
Status RepeatPass::RunOnNode(std::shared_ptr<CacheLookupOp> node, bool *modified) {
  if (!node->IsLeaf()) {
    // By definition, the CacheLookup must be a leaf op.  Make that clear here.
    RETURN_STATUS_UNEXPECTED("CacheLookupOp must be a leaf node!");
  }

  // save the lookup op.  There could be a repeat in the cache miss leg of the merge op, in which case we
  // may still need to be flagged as a repeating leaf.  We can't decide that here though, so save ourself
  // into the pass so that the decision can be made during the processing of the cache miss leg of the merge.
  // Further, if there's a repeat above the merge but no repeat in the cache miss leg, then the merge op will
  // add the lookup to the eoe stack
  cache_lookup_ = std::static_pointer_cast<DatasetOp>(node);

  return Status::OK();
}

Status RepeatPass::RunOnNode(std::shared_ptr<DeviceQueueOp> node, bool *modified) {
  // Set total repeats and total epochs for the DeviceQueueOp
  node->set_total_repeats(num_epochs_);
  node->set_num_repeats_per_epoch(1);
  return Status::OK();
}

// Adds an operator to the eoe operator stack save area
void RepeatPass::AddToEOEOpStack(std::shared_ptr<DatasetOp> dataset_op) {
  op_stack *current_stack = eoe_op_stacks_.top().get();
  current_stack->push(dataset_op);
}

// Pops an operator from the eoe operator stack save area
std::shared_ptr<DatasetOp> RepeatPass::PopFromEOEOpStack() {
  std::shared_ptr<DatasetOp> top_op = nullptr;
  op_stack *current_stack = eoe_op_stacks_.top().get();
  if (current_stack != nullptr && !current_stack->empty()) {
    top_op = current_stack->top();
    current_stack->pop();
  }
  return top_op;
}

// Adds an operator to the cached operator stack save area
void RepeatPass::AddToCachedOpStack(std::shared_ptr<DatasetOp> dataset_op) { cached_op_stacks_.push(dataset_op); }

// Pops an operator from the cached operator stack save area
std::shared_ptr<DatasetOp> RepeatPass::PopFromCachedOpStack() {
  std::shared_ptr<DatasetOp> top_op = nullptr;
  if (!cached_op_stacks_.empty()) {
    top_op = cached_op_stacks_.top();
    cached_op_stacks_.pop();
  }
  return top_op;
}
}  // namespace dataset
}  // namespace mindspore
