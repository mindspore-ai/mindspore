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

#ifndef MINDSPORE_LITE_SRC_ANF_IMPORTER_IMPORTER_FROM_META_GRAPHT_H_
#define MINDSPORE_LITE_SRC_ANF_IMPORTER_IMPORTER_FROM_META_GRAPHT_H_

#include <utility>

#include "schema/inner/model_generated.h"
#include "src/common/anf_importer/anf_importer.h"

namespace mindspore::lite {
class AnfImporterFromMetaGraphT : public AnfImporter {
 public:
  explicit AnfImporterFromMetaGraphT(schema::MetaGraphT *meta_graph, FuncGraphPtr func_graph)
      : meta_graph_(meta_graph), func_graph_(std::move(func_graph)) {}

  ~AnfImporterFromMetaGraphT() override = default;

  FuncGraphPtr GetResult() override;

 private:
  void ConverterConstTensor() override;

  int ConverterCNode() override;

  void AddReturnCNode() override;

 private:
  schema::MetaGraphT *meta_graph_;
  FuncGraphPtr func_graph_;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_ANF_IMPORTER_IMPORTER_FROM_META_GRAPHT_H_

