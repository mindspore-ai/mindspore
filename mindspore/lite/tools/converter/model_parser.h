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

#ifndef MS_MODEL_PARSER_H
#define MS_MODEL_PARSER_H
#include <google/protobuf/message.h>
#include <string>
#include <memory>
#include "schema/inner/model_generated.h"
#include "src/common/anf_importer/import_from_meta_graphT.h"
#include "ir/anf.h"
#include "include/errorcode.h"

namespace mindspore::lite {
using namespace schema;
class ModelParser {
 public:
  ModelParser() {}

  virtual ~ModelParser() {}

  virtual FuncGraphPtr ParseToAnf(const std::string &modelFile, const std::string &weightFile) {
    auto *meta_graph = Parse(modelFile, weightFile);
    if (meta_graph == nullptr) {
      MS_LOG(ERROR) << "Parse to metaGraph return nullptr";
      return nullptr;
    }
    return Fb2Anf(Parse(modelFile, weightFile));
  }
  virtual schema::MetaGraphT *Parse(const std::string &modelFile, const std::string &weightFile,
                                    const QuantType &quantType = QuantType_QUANT_NONE) = 0;

 public:
  static FuncGraphPtr Fb2Anf(schema::MetaGraphT *meta_graph) {
    MS_EXCEPTION_IF_NULL(meta_graph);
    auto func_graph = std::make_shared<FuncGraph>();
    auto importer = new AnfImporterFromMetaGraphT(meta_graph, func_graph);
    auto ret = importer->Import();
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "Import anf_graph from meta_graphT failed, ret: " << ret;
      return nullptr;
    }
    return func_graph;
  }
};
}  // namespace mindspore::lite

#endif


