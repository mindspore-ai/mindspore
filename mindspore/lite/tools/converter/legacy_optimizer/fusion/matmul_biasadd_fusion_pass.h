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

#ifndef MINDSPORE_PREDICT_MATMUL_BIASADD_FUSION_PASS_H
#define MINDSPORE_PREDICT_MATMUL_BIASADD_FUSION_PASS_H

#include <string>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <utility>
#include "tools/converter/legacy_optimizer/fusion/fusion_pass.h"
#include "tools/common/graph_util.h"

namespace mindspore {
namespace lite {
constexpr const char *MATMUL_NAME = "MATMUL";

class MatMulBiasAddFusionPass : public FusionPass {
 public:
  MatMulBiasAddFusionPass() = default;

  ~MatMulBiasAddFusionPass() override;

  STATUS DefinePattern() override;

  STATUS DoFusion(MetaGraphT *graph, const std::string &patternName,
                  std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) override;

  STATUS Run(MetaGraphT *graph) override;

 protected:
  static STATUS AddFullConnectionBiasTensor(const std::shared_ptr<Path> &matMulPath,
                                            const std::shared_ptr<Path> &dstPath, MetaGraphT *subGraph);
  STATUS InsertTransposeNode(MetaGraphT *subGraph, const std::shared_ptr<Path> &matMulPath);

 protected:
  bool transA = false;
  bool transB = false;
  size_t id = 0;

  OpDefCopyer TransposeOpCopyer = [](CNodeT *inOpDef) -> std::unique_ptr<CNodeT> {
    auto newOpDef = std::make_unique<schema::CNodeT>();
    if (newOpDef == nullptr) {
      MS_LOG(ERROR) << "new CNodeT failed";
      return nullptr;
    }
    newOpDef->name = inOpDef->name;
    newOpDef->quantType = inOpDef->quantType;
    newOpDef->primitive = std::make_unique<schema::PrimitiveT>();
    if (newOpDef->primitive == nullptr) {
      MS_LOG(ERROR) << "new PrimitiveT failed";
      return nullptr;
    }
    newOpDef->primitive->value.type = schema::PrimitiveType_Transpose;
    return newOpDef;
  };
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_PREDICT_MATMUL_BIASADD_FUSION_PASS_H
