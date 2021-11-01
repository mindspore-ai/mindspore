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

#include "tools/converter/optimizer.h"
#include "src/common/log_adapter.h"
#include "src/common/log_util.h"

namespace mindspore {
namespace lite {
Optimizer::~Optimizer() {
  for (auto pass : graph_passes_) {
    delete (pass);
  }
  graph_passes_.clear();
  for (auto pass : node_passes_) {
    delete (pass);
  }
  node_passes_.clear();
}

void Optimizer::AddPass(GraphPass *graph_pass) {
  if (graph_pass != nullptr) {
    this->graph_passes_.emplace_back(graph_pass);
  }
}

void Optimizer::AddPass(NodePass *node_pass) {
  if (node_pass != nullptr) {
    this->node_passes_.emplace_back(node_pass);
  }
}

STATUS Optimizer::Run(schema::MetaGraphT *graph_defT) {
  MS_ASSERT(graph_defT != nullptr);
  STATUS status;
  bool ifNotChanged = true;
  // each node should go through all node pass not each node pass go through all node
  for (auto &opDef : graph_defT->nodes) {
    for (auto pass : this->node_passes_) {
      auto graph_node = new (std::nothrow) GraphNode(graph_defT, opDef.get());
      if (graph_node == nullptr) {
        return RET_ERROR;
      }
      if (pass == nullptr) {
        delete graph_node;
        return RET_ERROR;
      }
      status = pass->Run(graph_node);
      delete graph_node;
      if (status != RET_OK && status != RET_NO_CHANGE && status != RET_INFER_INVALID) {
        MS_LOG(ERROR) << "Run NodePass failed";
        return status;
      } else {
        if (status == RET_OK) {
          ifNotChanged = false;
        }
      }
    }
  }

  for (auto pass : this->graph_passes_) {
    CHECK_NULL_RETURN(pass);
    status = pass->Run(graph_defT);
    if (status != RET_OK && status != RET_NO_CHANGE && status != RET_INFER_INVALID) {
      MS_LOG(ERROR) << "Run GraphPass failed";
      return status;
    } else {
      if (status == RET_OK) {
        ifNotChanged = false;
      }
    }
  }
  return ifNotChanged ? RET_NO_CHANGE : RET_OK;
}
}  // namespace lite
}  // namespace mindspore
