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

#include <memory>
#include "minddata/dataset/engine/opt/post/generator_node_pass.h"
#include "minddata/dataset/engine/ir/datasetops/source/generator_node.h"

namespace mindspore {
namespace dataset {

GeneratorNodePass::GeneratorNodePass() : repeat_ancestors_({}) {}
/*
 * A diagram shows how the code work:
 *   With the tree below as an input
 *
 *             EpochCtrl(-1)
 *              /    \
 *          Repeat1   \
 *           /      Repeat3
 *          ..          \
 *         /          Generator2
 *      Repeat2       Add: Gen2-Rep3
 *       /
 *     Generator1
 *     Add: Gen1-Rep2
 *
 * The sequence of the DFS walk of the tree looks like this:
 *   1) Visit(EpochCtrl): push EpochCtrl, repeat_ancestor_ = { EpochCtrl }
 *   2) Visit(Repeat1): push Repeat1, repeat_ancestors_ = { EpochCtrl, Repeat1 }
 *   3) Visit(Repeat2): push Repeat2, repeat_ancestors_ = { EpochCtrl, Repeat1, Repeat2 }
 *   4) Visit(Generator1): record Repeat2 as its ancestor
 *                         record Repeat1 as Repeat2's ancestor
 *                         record EpochCtrl as Repeat1's ancestor
 *   5) VisitAfter(Repeat2): pop Repeat2, repeat_ancestors_ = { EpochCtrl, Repeat1 }
 *   6) VisitAfter(Repeat1): pop Repeat1, repeat_ancestors_ = { EpochCtrl }
 *   7) Visit(Repeat3): push Repeat3, repeat_ancestors_ = { EpochCtrl, Repeat3 }
 *   8) Visit(Generator2): record Repeat3 as its ancestors
 *                         record EpochCtrl as Repeat3's ancestor
 *   9) VisitAfter(Repeat3): pop Repeat3, repeat_ancestors_ = { EpochCtrl }
 *   10) VisitAfter(EpochCtrl): don't care. We could pop EpochCtrl.
 */

Status GeneratorNodePass::Visit(std::shared_ptr<EpochCtrlNode> node, bool *const modified) {
  // Add this EpochCtrl node as an ancestor of its descendant
  repeat_ancestors_.push_back(node);
  return Status::OK();
}

Status GeneratorNodePass::Visit(std::shared_ptr<RepeatNode> node, bool *const modified) {
  // Add this Repeat node as an ancestor of its descendant
  repeat_ancestors_.push_back(node);
  return Status::OK();
}

Status GeneratorNodePass::Visit(std::shared_ptr<GeneratorNode> node, bool *const modified) {
  // Form a reset relationship with the immediate Repeat/EpochCtrl ancestor node of this leaf Generator Node
  // only when any of its ancestors is an infinite repeat.
  if (repeat_ancestors_.size() > 0) {
    bool infinite_repeat = false;
    for (auto &repeat_ancestor : repeat_ancestors_) {
      if (repeat_ancestor->Count() < 0) {
        infinite_repeat = true;
        break;
      }
    }
    if (infinite_repeat) {
      // Form a pair-wise relationship between this leaf Generator node and its immediate Repeat/EpochCtrl
      // ancestor node, and between the next adjacent pairs in the vector. For example,
      // if we have GeneratorNode -> Repeat1 -> Repeat2 -> EpochCtrl(-1), the pair-wise relationships are:
      // (GeneratorNode, Repeat1), (Repeat1, Repeat2), and (Repeat2, EpochCtrl)
      for (auto i = repeat_ancestors_.size() - 1; i > 0; --i) {
        auto ancestor = repeat_ancestors_[i - 1];
        RETURN_IF_NOT_OK(repeat_ancestors_[i]->AddResetAncestor(ancestor));
      }
      RETURN_IF_NOT_OK(node->AddResetAncestor(repeat_ancestors_.back()));
    }
  }
  return Status::OK();
}

Status GeneratorNodePass::VisitAfter(std::shared_ptr<RepeatNode> node, bool *const modified) {
  // When we backtrack from the same Repeat node, we pop it out from the list of ancestors.
  repeat_ancestors_.pop_back();
  return Status::OK();
}

Status GeneratorNodePass::VisitAfter(std::shared_ptr<EpochCtrlNode> node, bool *const modified) {
  // As EpochCtrl node is a terminal node, the process stops here.
  // Popping it back out of the reset ancestors is unnecessary.
  // This function becomes a no-op function and can be deleted completely.
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
