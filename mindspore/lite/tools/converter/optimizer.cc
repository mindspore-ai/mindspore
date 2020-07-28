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
#include "utils/log_adapter.h"

namespace mindspore {
namespace lite {
Optimizer::~Optimizer() {
  for (auto pass : graphPasses) {
    if (pass != nullptr) {
      delete (pass);
    }
  }

  for (auto pass : nodePasses) {
    if (pass != nullptr) {
      delete (pass);
    }
  }
}

void Optimizer::AddPass(GraphPass *graphPass) {
  if (graphPass != nullptr) {
    this->graphPasses.emplace_back(graphPass);
  }
}

void Optimizer::AddPass(NodePass *nodePass) {
  if (nodePass != nullptr) {
    this->nodePasses.emplace_back(nodePass);
  }
}

STATUS Optimizer::Run(schema::MetaGraphT *graphDefT) {
  STATUS status;
  bool ifNotChanged = true;
  // each node should go through all node pass not each node pass go through all node
  for (auto &opDef : graphDefT->nodes) {
    for (auto pass : this->nodePasses) {
      status = pass->Run(new GraphNode(graphDefT, opDef.get()));
      if (status != RET_OK && status != RET_NO_CHANGE) {
        MS_LOG(ERROR) << "Run NodePass failed";
        return status;
      } else {
        if (status == RET_OK) {
          ifNotChanged = false;
        }
      }
    }
  }

  for (auto pass : this->graphPasses) {
    status = pass->Run(graphDefT);
    if (status != RET_OK && status != RET_NO_CHANGE) {
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

