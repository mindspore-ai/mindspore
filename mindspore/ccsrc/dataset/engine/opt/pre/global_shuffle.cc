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

#include <vector>
#include <algorithm>
#include "dataset/engine/opt/pre/global_shuffle.h"
#include "dataset/engine/execution_tree.h"
#include "dataset/engine/datasetops/shuffle_op.h"
#include "dataset/engine/datasetops/source/tf_reader_op.h"
#include "dataset/engine/datasetops/source/text_file_op.h"
#include "dataset/engine/datasetops/source/clue_op.h"

namespace mindspore {
namespace dataset {

Status GlobalShufflePass::RunOnTree(ExecutionTree *tree, bool *modified) {
  std::vector<std::shared_ptr<TFReaderOp>> tf_readers;
  std::vector<std::shared_ptr<TextFileOp>> text_files;
  std::vector<std::shared_ptr<ClueOp>> clues;

  // Pass 1, search for all sources which requires global shuffle
  for (auto &op : *tree) {
    if (auto ptr = std::dynamic_pointer_cast<TFReaderOp>(op.shared_from_this())) {
      if (ptr->RequireGlobalShuffle()) {
        tf_readers.push_back(ptr);
        continue;
      }
    }
    if (auto ptr = std::dynamic_pointer_cast<TextFileOp>(op.shared_from_this())) {
      if (ptr->RequireGlobalShuffle()) {
        text_files.push_back(ptr);
        continue;
      }
    }
    if (auto ptr = std::dynamic_pointer_cast<ClueOp>(op.shared_from_this())) {
      if (ptr->RequireGlobalShuffle()) {
        clues.push_back(ptr);
        continue;
      }
    }
  }

  // Pass 2, insert shuffle nodes
  // The following blocks can be implemented with template if we unify the CountTotalRows across all source nodes .
  for (auto node : tf_readers) {
    std::shared_ptr<ShuffleOp::Builder> builder = std::make_shared<ShuffleOp::Builder>();
    int64_t total_rows = 0;
    TFReaderOp::CountTotalRows(&total_rows, node->FileNames(), 8, true);
    int32_t avg_file_size = total_rows / (node->FileNames().size());
    builder->SetShuffleSize(std::max(avg_file_size * 4, 10000));
    std::shared_ptr<ShuffleOp> op;
    RETURN_IF_NOT_OK(builder->Build(&op));
    RETURN_IF_NOT_OK(tree->AssociateNode(op));
    RETURN_IF_NOT_OK(node->InsertAsParent(op));
  }

  for (auto node : text_files) {
    std::shared_ptr<ShuffleOp::Builder> builder = std::make_shared<ShuffleOp::Builder>();
    int64_t total_rows = 0;
    TextFileOp::CountAllFileRows(node->FileNames(), &total_rows);
    int32_t avg_file_size = total_rows / (node->FileNames().size());
    builder->SetShuffleSize(std::max(avg_file_size * 4, 10000));
    std::shared_ptr<ShuffleOp> op;
    RETURN_IF_NOT_OK(builder->Build(&op));
    RETURN_IF_NOT_OK(tree->AssociateNode(op));
    RETURN_IF_NOT_OK(node->InsertAsParent(op));
  }

  for (auto node : clues) {
    std::shared_ptr<ShuffleOp::Builder> builder = std::make_shared<ShuffleOp::Builder>();
    int64_t total_rows = 0;
    ClueOp::CountAllFileRows(node->FileNames(), &total_rows);
    int32_t avg_file_size = total_rows / (node->FileNames().size());
    builder->SetShuffleSize(std::max(avg_file_size * 4, 10000));
    std::shared_ptr<ShuffleOp> op;
    RETURN_IF_NOT_OK(builder->Build(&op));
    RETURN_IF_NOT_OK(tree->AssociateNode(op));
    RETURN_IF_NOT_OK(node->InsertAsParent(op));
  }

  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
