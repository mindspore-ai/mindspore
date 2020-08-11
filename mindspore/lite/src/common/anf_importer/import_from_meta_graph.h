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

#ifndef MINDSPORE_LITE_SRC_ANF_IMPORTER_IMPORTER_FROM_META_GRAPH_H_
#define MINDSPORE_LITE_SRC_ANF_IMPORTER_IMPORTER_FROM_META_GRAPH_H_

#include <memory>
#include <map>
#include "src/train/model_impl.h"
#include "schema/model_generated.h"
#include "src/common/anf_importer/anf_importer.h"

namespace mindspore::lite {
class AnfImporterFromMetaGraph : public AnfImporter {
 public:
  explicit AnfImporterFromMetaGraph(std::shared_ptr<ModelImpl> model) : model_(model) {}

  ~AnfImporterFromMetaGraph() override = default;

  FuncGraphPtr GetResult() override;

 private:
  void ConverterConstTensor() override;

  int ConverterCNode() override;

  void AddReturnCNode() override;

 private:
  std::shared_ptr<ModelImpl> model_ = nullptr;
  std::map<int, AnfNodePtr> originator_;
  int num_of_tensors_ = 0;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_ANF_IMPORTER_IMPORTER_FROM_META_GRAPH_H_
