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

#ifndef MINDSPORE_LITE_TOOLS_COMMON_ANF_IMPORTER_IMPORTER_FROM_META_GRAPHT_H_
#define MINDSPORE_LITE_TOOLS_COMMON_ANF_IMPORTER_IMPORTER_FROM_META_GRAPHT_H_

#include <utility>
#include <memory>
#include <unordered_map>
#include "schema/inner/model_generated.h"
#include "abstract/abstract_value.h"
#include "ir/func_graph.h"

namespace mindspore::lite {
class AnfImporterFromMetaGraphT {
 public:
  virtual ~AnfImporterFromMetaGraphT() = default;

  static FuncGraphPtr Fb2Anf(schema::MetaGraphT *meta_graph);

 private:
  explicit AnfImporterFromMetaGraphT(schema::MetaGraphT *meta_graph) : meta_graph_(meta_graph) {
    this->func_graph_ = std::make_shared<FuncGraph>();
  }

  int ConverterConstTensor();

  int ConverterCNode();

  ValueNodePtr ConvertPrimitive(const std::unique_ptr<schema::CNodeT> &cNode);

  static abstract::AbstractTensorPtr ConvertTensorToAbstractTensor(const std::unique_ptr<schema::TensorT> &tensor);

  int ConvertAbstract(const std::unique_ptr<schema::CNodeT> &src_cnode, const CNodePtr &dst_cnode);

  int AddReturnCNode();

  AnfNodePtr GetNode(int tensor_id);

  void AddNode(int tensor_id, AnfNodePtr node);

 private:
  std::unordered_map<int, AnfNodePtr> nodes_;
  schema::MetaGraphT *meta_graph_;
  FuncGraphPtr func_graph_;
};

FuncGraphPtr Fb2Anf(schema::MetaGraphT *meta_graph);
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_TOOLS_COMMON_ANF_IMPORTER_IMPORTER_FROM_META_GRAPHT_H_
