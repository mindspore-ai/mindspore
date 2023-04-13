/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_TRAIN_OPTIMIZER_FUSION_GRU_FUSION_PASS_H_
#define MINDSPORE_LITE_SRC_TRAIN_OPTIMIZER_FUSION_GRU_FUSION_PASS_H_

#include <memory>
#include <vector>
#include "tools/converter/optimizer.h"

namespace mindspore {
namespace lite {
class LinkInfoManager;
class GruFusionPass : public GraphPass {
 public:
  GruFusionPass() = default;
  ~GruFusionPass() override = default;
  STATUS Run(schema::MetaGraphT *graph) override;

 private:
  STATUS FuseToGruCell(schema::MetaGraphT *graph);
  STATUS FuseGruCell(schema::MetaGraphT *graph);
  bool MatchPatten(schema::MetaGraphT *graph, uint32_t stack_index, std::vector<uint32_t> *strided_slices,
                   std::vector<uint32_t> *squeezes, std::vector<uint32_t> *gru_cells);
  bool CheckGruCellConnection(schema::MetaGraphT *graph, const std::vector<uint32_t> &gru_cells);
  STATUS CreateGru(schema::MetaGraphT *graph, uint32_t stack_index, const std::vector<uint32_t> &strided_slices,
                   const std::vector<uint32_t> &squeezes, const std::vector<uint32_t> &gru_cells);
  std::shared_ptr<LinkInfoManager> link_info_manager_{nullptr};
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_TRAIN_OPTIMIZER_FUSION_GRU_FUSION_PASS_H_
