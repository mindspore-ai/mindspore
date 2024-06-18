/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/opt/pre/skip_pushdown_pass.h"
#include "minddata/dataset/engine/ir/datasetops/batch_node.h"
#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"
#include "minddata/dataset/engine/ir/datasetops/map_node.h"
#include "minddata/dataset/engine/ir/datasetops/project_node.h"
#include "minddata/dataset/engine/ir/datasetops/rename_node.h"
#include "minddata/dataset/engine/ir/datasetops/root_node.h"
#include "minddata/dataset/engine/ir/datasetops/skip_node.h"
#ifdef ENABLE_PYTHON
#include "minddata/dataset/engine/ir/datasetops/source/generator_node.h"
#endif
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/ir/datasetops/source/minddata_node.h"
#endif
#include "minddata/dataset/engine/ir/datasetops/source/samplers/skip_first_epoch_sampler_ir.h"

namespace mindspore {
namespace dataset {
SkipPushdownPass::SkipNodes::SkipNodes() : skip_count_(0), skip_steps_(0) {}

// activate the optimization steps, and increase skip_count_ (if not the first skip node in the pipeline)
Status SkipPushdownPass::SkipNodes::Visit(std::shared_ptr<SkipNode> node, bool *const modified) {
  if (!node->OnceOnly()) {
    return Visit(std::static_pointer_cast<DatasetNode>(node), modified);
  }
  skip_count_ += node->Count();
  nodes_to_remove_.push_back(node);
  return Status::OK();
}

Status SkipPushdownPass::SkipNodes::VisitAfter(std::shared_ptr<SkipNode> node, bool *const modified) {
  if (!node->OnceOnly()) {
    return VisitAfter(std::static_pointer_cast<DatasetNode>(node), modified);
  }
  CHECK_FAIL_RETURN_UNEXPECTED(skip_count_ == 0, "The skip_count_ cannot be non-zero here.");
  return Status::OK();
}

Status SkipPushdownPass::SkipNodes::Visit(std::shared_ptr<BatchNode> node, bool *const modified) {
#ifdef ENABLE_PYTHON
  if (node->BatchSizeFunc()) {
    return Visit(std::static_pointer_cast<DatasetNode>(node), modified);
  }
#endif
  CHECK_FAIL_RETURN_UNEXPECTED(skip_count_ >= 0, "The skip size cannot be negative.");
  if (skip_count_ == 0) {
    return Status::OK();
  }  // no active skip node above. normal flow

  // we have an active skip node above.
  skip_count_ *= node->BatchSize();

  return Status::OK();
}

Status SkipPushdownPass::SkipNodes::Visit(std::shared_ptr<ProjectNode> node, bool *const modified) {
  CHECK_FAIL_RETURN_UNEXPECTED(skip_count_ >= 0, "The skip size cannot be negative.");
  return Status::OK();
}

Status SkipPushdownPass::SkipNodes::Visit(std::shared_ptr<RenameNode> node, bool *const modified) {
  CHECK_FAIL_RETURN_UNEXPECTED(skip_count_ >= 0, "The skip size cannot be negative.");
  return Status::OK();
}

Status SkipPushdownPass::SkipNodes::Visit(std::shared_ptr<MappableSourceNode> node, bool *const modified) {
  CHECK_FAIL_RETURN_UNEXPECTED(skip_count_ >= 0, "The skip size cannot be negative.");
  if (skip_count_ == 0) {
    return Status::OK();
  }  // no active skip node above. normal flow

  // we have an active skip node above.
  auto new_sampler = std::make_shared<SkipFirstEpochSamplerObj>(skip_count_);
  MS_LOG(INFO) << "Adding SkipFirstEpochSampler(" << skip_count_ << ")";
  auto sampler = node->Sampler();
  if (sampler != nullptr) {
    RETURN_IF_NOT_OK(new_sampler->AddChildSampler(sampler));
  }
  node->SetSampler(new_sampler);
  skip_count_ = 0;

  return Status::OK();
}

Status SkipPushdownPass::SkipNodes::Visit(std::shared_ptr<MapNode> node, bool *const modified) {
  CHECK_FAIL_RETURN_UNEXPECTED(skip_count_ >= 0, "The skip size cannot be negative.");
  if (skip_count_ == 0) {
    return Status::OK();
  }  // no active skip node above. normal flow

  // we have an active skip node above.
  MS_LOG(WARNING)
    << "Pushing down skip node below a map node will result in slightly different outputs for random transformations.";
  return Status::OK();
}

Status SkipPushdownPass::SkipNodes::Visit(std::shared_ptr<NonMappableSourceNode> node, bool *const modified) {
  node->SetSkipSteps(skip_steps_);
  return InsertSkipNode(node);
}

#ifdef ENABLE_PYTHON
Status SkipPushdownPass::SkipNodes::Visit(std::shared_ptr<GeneratorNode> node, bool *const modified) {
  if (!node->IsMappableSource()) {
    RETURN_IF_NOT_OK(InsertSkipNode(node));
  } else {
    RETURN_IF_NOT_OK(Visit(std::static_pointer_cast<MappableSourceNode>(node), modified));
  }
  return Status::OK();
}
#endif

// This functions is used for Ops that are random, and the ones in which Visit is Not Implemented yet;
Status SkipPushdownPass::SkipNodes::Visit(std::shared_ptr<DatasetNode> node, bool *const modified) {
  return InsertSkipNode(node);
}

Status SkipPushdownPass::SkipNodes::Visit(std::shared_ptr<RootNode> node, bool *const modified) {
  int64_t dataset_size = node->DatasetSize();
  int64_t step = node->Step();
  // in fast recover mode, we need to know how many steps are actually skipped
  // when we skip `step / dataset_size` epochs
  skip_steps_ = step / dataset_size * dataset_size;
  return Status::OK();
}

// constructor
SkipPushdownPass::SkipPushdownPass() = default;

// Walk the tree to push down the skip node inserted when Reset is called.
Status SkipPushdownPass::RunOnTree(std::shared_ptr<DatasetNode> root_ir, bool *const modified) {
  // Since previous pass may have disabled fast_recovery (due to shuffle node in pipeline),
  // we need to double check here if it is enabled.
  if (!GlobalContext::config_manager()->fast_recovery()) {
    MS_LOG(INFO) << "Pre pass: ignoring skip node pushdown pass (shuffle node was detected).";
    return Status::OK();
  }
  MS_LOG(INFO) << "Pre pass: skip node pushdown pass started.";
  // Assumption: The total skip counts in the first_epoch_only skip node is less than the size of the dataset. This
  // assumption is not validated here.
  // Create the skip node pass which can identify which nodes need to be removed and which ones added.
  std::unique_ptr<SkipPushdownPass::SkipNodes> skip_nodes = std::make_unique<SkipPushdownPass::SkipNodes>();
  if (root_ir->IsSizeDefined()) {
    RETURN_IF_NOT_OK(skip_nodes->Run(root_ir, modified));
  }

  // Update modified flag if there were any nodes identified to be removed
  if (!skip_nodes->nodes_to_remove().empty() || !skip_nodes->insert_skip_above().empty()) {
    *modified = true;
  }

  // Add skip node(s) to the tree (if any)
  for (const auto &iter : skip_nodes->insert_skip_above()) {
    MS_LOG(INFO) << "Inserting a Skip(" << iter.second << ") node above this node: " << iter.first->Name();
    auto new_skip_node = std::make_shared<SkipNode>(iter.second);
    new_skip_node->SetOnceOnly(true);
    RETURN_IF_NOT_OK(iter.first->InsertAbove(new_skip_node));
  }

  // Then, execute the removal of any nodes that were set up for removal
  for (const auto &node : skip_nodes->nodes_to_remove()) {
    RETURN_IF_NOT_OK(node->Drop());
  }

  MS_LOG(INFO) << "Pre pass: skip node pushdown pass is complete.";
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
