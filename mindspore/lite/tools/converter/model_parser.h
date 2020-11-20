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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MODEL_PARSER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_MODEL_PARSER_H
#include <google/protobuf/message.h>
#include <string>
#include <memory>
#include "schema/inner/model_generated.h"
#include "tools/anf_importer/import_from_meta_graphT.h"
#include "ir/anf.h"
#include "tools/converter/converter_context.h"

namespace mindspore::lite {
using namespace schema;
class ModelParser {
 public:
  ModelParser() = default;

  virtual ~ModelParser() = default;

  virtual FuncGraphPtr Parse(const std::string &model_file, const std::string &weight_file,
                             const QuantType &quant_type) {
    auto *meta_graph = ParseToFb(model_file, weight_file, quant_type);
    if (meta_graph == nullptr) {
      MS_LOG(ERROR) << "parse model to fb failed";
      return nullptr;
    }
    auto func_graph = this->Fb2Anf(meta_graph);
    delete (meta_graph);
    return func_graph;
  }

 protected:
  virtual schema::MetaGraphT *ParseToFb(const std::string &model_file, const std::string &weight_file,
                                        const QuantType &quant_type = QuantType_QUANT_NONE) = 0;

 public:
  static FuncGraphPtr Fb2Anf(schema::MetaGraphT *meta_graph) {
    if (meta_graph == nullptr) {
      MS_LOG(ERROR) << "meta_graph is null";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_NULL_PTR);
      return nullptr;
    }
    auto func_graph = std::make_shared<FuncGraph>();
    AnfImporterFromMetaGraphT importer(meta_graph, func_graph);
    auto status = importer.Import();
    if (RET_OK != status) {
      MS_LOG(ERROR) << "Import anf_graph from meta_graphT failed, ret: " << status;
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
      return nullptr;
    }
    return func_graph;
  }
};
}  // namespace mindspore::lite

#endif
