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

#ifndef DATASET_ENGINE_OPT_PASS_PRE_MAPCOLREORDER_H
#define DATASET_ENGINE_OPT_PASS_PRE_MAPCOLREORDER_H

#include <memory>
#include "dataset/engine/opt/pass.h"

namespace mindspore {
namespace dataset {
// Map Column Recorder Pass will insert ProjectOp when MapOp requires a full output columns reorder.
// Example:
// Input Tree:  TFReader -> MapOp(with col_order) -> Batch
// Output Tree: TFReader -> MapOp -> ProjectOp(col_order) -> Batch
class MapColumnReorder : public TreePass {
  Status RunOnTree(ExecutionTree *tree, bool *modified) override;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_OPT_PASS_PRE_MAPCOLREORDER_H
