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

#ifndef MINDSPORE_PREDICT_BATCHNORM_GLOBAL_FORMAT_TRANSFORM_PASS_H
#define MINDSPORE_PREDICT_BATCHNORM_GLOBAL_FORMAT_TRANSFORM_PASS_H

#include <unordered_map>
#include <set>
#include <vector>
#include <memory>
#include <string>
#include <utility>
#include "tools/common/graph_util.h"
#include "tools/converter/optimizer.h"
#include "tools/converter/legacy_optimizer/graph/format_trans_pass.h"

using mindspore::schema::TensorT;
namespace mindspore {
namespace lite {
class GlobalFormatTransformPass : public FormatTransPass {
 public:
  GlobalFormatTransformPass() = default;

  ~GlobalFormatTransformPass() override = default;

  STATUS Run(MetaGraphT *graph) override;

 protected:
  STATUS TransWeightToNhwc(MetaGraphT *graph, const std::set<size_t> &pre_not_trans_nodes);

  STATUS FindPreNh2NcNodes(MetaGraphT *graph, size_t nc2nh_index, std::vector<size_t> *to_do_insert_nodes,
                           std::vector<size_t> *pre_not_trans_nodes);
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_PREDICT_BATCHNORM_GLOBAL_FORMAT_TRANSFORM_PASS_H
