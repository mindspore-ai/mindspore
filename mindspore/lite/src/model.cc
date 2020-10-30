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
#include "src/ops/primitive_c.h"
#include "include/model.h"
#include "src/common/log_adapter.h"
#include "src/model_common.h"

namespace mindspore::lite {
Model *Model::Import(const char *model_buf, size_t size) { return ImportFromBuffer(model_buf, size, false); }

void Model::Free() {
  if (this->buf != nullptr) {
    free(this->buf);
    this->buf = nullptr;
  }
}

void Model::Destroy() {
  Free();
  auto nodes_size = this->all_nodes_.size();
  for (size_t i = 0; i < nodes_size; ++i) {
    auto node = this->all_nodes_[i];
    MS_ASSERT(node != nullptr);
    MS_ASSERT(node->primitive_ != nullptr);
    delete node->primitive_;
    node->primitive_ = nullptr;
    delete node;
  }
  this->all_nodes_.clear();

  auto sub_graph_size = this->sub_graphs_.size();
  for (size_t i = 0; i < sub_graph_size; ++i) {
    auto sub_graph = this->sub_graphs_[i];
    delete sub_graph;
  }
}

Model::~Model() { Destroy(); }
}  // namespace mindspore::lite
