/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <utility>
#include "src/expression/export.h"
#include "src/expression/ops.h"
#include "src/common/utils.h"
#include "nnacl/conv_parameter.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
constexpr static int kFmkVal = 3;

int ExportSession::Init(const std::string model_name, std::string version) {
  meta_graph_ = new (std::nothrow) mindspore::schema::MetaGraphT();
  if (meta_graph_ == nullptr) {
    MS_LOG(ERROR) << "cannot allocate meta_graph";
    return RET_ERROR;
  }
  meta_graph_->fmkType = kFmkVal;
  meta_graph_->name = model_name;
  meta_graph_->version = version;
  return RET_OK;
}

bool ExportSession::IsToDependOnly(EXPR *expr) {
  auto itr = outmap_.find(expr);
  if (itr != outmap_.end() && !itr->second.empty()) {
    for (auto expr : itr->second) {
      auto node = expr->node();
      if (node->primitive() != schema::PrimitiveType_Depend) return false;
    }
    return true;
  }
  return false;
}

int ExportSession::SetInputOutput(const std::vector<EXPR *> &inputs, const std::vector<EXPR *> &outputs) {
  for (auto &in : inputs) {
    auto id = GetOutput(in);
    meta_graph_->inputIndex.push_back(id);
  }
  for (auto &out : outputs) {
    auto id = GetOutput(out);
    meta_graph_->outputIndex.push_back(id);
  }
  auto sub_graph = std::make_unique<mindspore::schema::SubGraphT>();
  if (sub_graph == nullptr) {
    MS_LOG(ERROR) << "cannot allocate SubGraphT";
    return RET_ERROR;
  }
  auto model_name = meta_graph_->name;
  sub_graph->name = model_name + "_subgraph";
  sub_graph->inputIndices = meta_graph_->inputIndex;
  sub_graph->outputIndices = meta_graph_->outputIndex;
  for (size_t i = 0; i < meta_graph_->nodes.size(); i++) sub_graph->nodeIndices.push_back(i);
  for (size_t i = 0; i < meta_graph_->allTensors.size(); i++) sub_graph->tensorIndices.push_back(i);
  meta_graph_->subGraph.emplace_back(std::move(sub_graph));
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
