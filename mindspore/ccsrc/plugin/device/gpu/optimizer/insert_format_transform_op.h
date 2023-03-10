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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GPU_INSERT_FORMAT_TRANSFORM_OP_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GPU_INSERT_FORMAT_TRANSFORM_OP_H_

#include <memory>
#include <string>
#include <vector>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class InsertFormatTransformOp : public PatternProcessPass {
 public:
  explicit InsertFormatTransformOp(bool multigraph = true)
      : PatternProcessPass("insert_format_transform_op", multigraph) {}
  ~InsertFormatTransformOp() override = default;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  void ProcessForTupleItem(const FuncGraphPtr &graph, const AnfNodePtr &node, int node_index,
                           const std::vector<int64_t> &transpose_perm, const std::string &transpose_format) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GPU_INSERT_FORMAT_TRANSFORM_OP_H_
