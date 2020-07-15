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
#include "minddata/dataset/engine/datasetops/epoch_ctrl_op.h"

namespace mindspore {
namespace dataset {

RepeatPass::RepeatPass() : is_repeated_(false), nested_repeats_(0), is_merge_(false), cache_lookup_(nullptr) {}

// Identifies the subtree below this node as being in a repeated path of the tree.
Status RepeatPass::PreRunOnNode(std::shared_ptr<RepeatOp> node, bool *modified) {
  // Create a new stack for eoe operators and push onto our stack of stacks.
  std::unique_ptr<eoe_op_stack> new_stack = std::make_unique<eoe_op_stack>();
  eoe_op_stacks_.push(std::move(new_stack));
  // If we are already repeated, then this is a nested repeat.
  if (is_repeated_) {
    nested_repeats_++;
  }
  is_repeated_ = true;
  return Status::OK();
}

// Identifies the subtree below this node as being in a repeated path of the tree.
Status RepeatPass::PreRunOnNode(std::shared_ptr<EpochCtrlOp> node, bool *modified) {
  // EpochCtrl is derived from RepeatOp.  Generally it should do the identical setup
  // that RepeatOp does.  However, epoch control is actually simpler because it can
  // only exist as the root node so it doesn't need all the nested code.
  // Create a new stack for eoe operators and push onto our stack of stacks.
  std::unique_ptr<eoe_op_stack> new_stack = std::make_unique<eoe_op_stack>();
  eoe_op_stacks_.push(std::move(new_stack));
  is_repeated_ = true;
  return Status::OK();
}

// Identifies the subtree below this node as being in a cache merge path
Status RepeatPass::PreRunOnNode(std::shared_ptr<CacheMergeOp> node, bool *modified) {
  // Turn on the flag that we're under a merge op
  is_merge_ = true;
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
  eoe_op_stack *current_stack = eoe_op_stacks_.top().get();
  if (!current_stack->empty()) {
    RETURN_STATUS_UNEXPECTED("The eoe op stack should be empty right now!");
  }
  eoe_op_stacks_.pop();

  // We are a repeat op in the descendant tree of a merge op, then we take the saved lookup up
  // and add it to the list of eoe/leaf ops for the repeat.  It is important that the op is removed
  // from the save area, because the merge op above us may also take action on it later for a different
  // case when there is no repeat in the merge leg.
  if (is_merge_ && cache_lookup_) {
    cache_lookup_->set_control_flag(DatasetOp::kDeOpRepeated);
    node->AddToEoeList(std::move(cache_lookup_));
  }

  // If we are a nested repeat, then we add ourself to the repeat stack for the next one above us.
  // A nested repeat acts like an eoe/leaf for the repeat in the ascendant tree.
  if (nested_repeats_ > 0) {
    node->set_control_flag(DatasetOp::kDeOpRepeated);
    AddToEOEOpStack(node);
    nested_repeats_--;
  } else {
    // If we are not nested, or we were the top-most repeat, now we clear the flag
    if (nested_repeats_ != 0) {
      RETURN_STATUS_UNEXPECTED("Nested repeat counter cannot be negative!");
    }
    is_repeated_ = false;
  }

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
  return Status::OK();
}

// CacheOp removes previous leaf ops and replaces them with itself
Status RepeatPass::RunOnNode(std::shared_ptr<CacheOp> node, bool *modified) {
  if (is_repeated_) {
    node->set_control_flag(DatasetOp::kDeOpRepeated);
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
      leaf_op->ClearControlFlag(DatasetOp::kDeOpLastRepeat);
      leaf_op->ClearControlFlag(DatasetOp::kDeOpRepeated);
      leaf_op = PopFromEOEOpStack();
    }
    AddToEOEOpStack(std::static_pointer_cast<DatasetOp>(node));
  }

  return Status::OK();
}

// All operators have a flag that might be set related to the repeat and any leaf nodes need to be set up
// for use with a controlling repeat above it.
Status RepeatPass::RunOnNode(std::shared_ptr<DatasetOp> node, bool *modified) {
  // If we are in a repeat path, then set our repeated flag
  if (is_repeated_) {
    node->set_control_flag(DatasetOp::kDeOpRepeated);

    // if we are a leaf node then save ourself in a stack for the repeat operator above us
    if (node->IsLeaf()) {
      AddToEOEOpStack(node);
    }
  }
  return Status::OK();
}

// Turns off the tracking for operations under merge op
Status RepeatPass::RunOnNode(std::shared_ptr<CacheMergeOp> node, bool *modified) {
  // Setting the flag is needed since we didn't call the base class DatasetOp version
  if (is_repeated_) {
    node->set_control_flag(DatasetOp::kDeOpRepeated);
    // If there was not any repeat in the merge cache miss leg, then the cache_lookup
    // would not have been consumed yet.  In that case, we need to assign it to the upper repeat eoe stack
    if (cache_lookup_) {
      AddToEOEOpStack(std::move(cache_lookup_));
    }
  }
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

  // If we are in a repeat path already, then there must be a repeat above the merge op
  // In this case, we naturally are a repeating leaf op so add the required setup for leafs under repeat here.
  if (is_repeated_) {
    node->set_control_flag(DatasetOp::kDeOpRepeated);
    // Delay the assigment of this leap to the eoe stack and allow the merge op processing to handle that.
  }

  // save the lookup op.  There could be a repeat in the cache miss leg of the merge op, in which case we
  // may still need to be flagged as a repeating leaf.  We can't decide that here though, so save ourself
  // into the pass so that the decision can be made during the processing of the cache miss leg of the merge.
  // Further, if there's a repeat above the merge but no repeat in the cache miss leg, then the merge op will
  // add the lookup to the eoe stack
  cache_lookup_ = std::static_pointer_cast<DatasetOp>(node);

  return Status::OK();
}

// Adds an operator to the eoe operator stack save area
void RepeatPass::AddToEOEOpStack(std::shared_ptr<DatasetOp> dataset_op) {
  eoe_op_stack *current_stack = eoe_op_stacks_.top().get();
  current_stack->push(dataset_op);
}

// Pops an operator from the eoe operator stack save area
std::shared_ptr<DatasetOp> RepeatPass::PopFromEOEOpStack() {
  std::shared_ptr<DatasetOp> top_op = nullptr;
  eoe_op_stack *current_stack = eoe_op_stacks_.top().get();
  if (current_stack != nullptr && !current_stack->empty()) {
    top_op = current_stack->top();
    current_stack->pop();
  }
  return top_op;
}
}  // namespace dataset
}  // namespace mindspore
