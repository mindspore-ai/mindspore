/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_ANF_EXPORTER_ANF_EXPORTER_H_
#define MINDSPORE_LITE_SRC_ANF_EXPORTER_ANF_EXPORTER_H_

#include <map>
#include <string>
#include <vector>
#include <memory>
#include "schema/inner/model_generated.h"
#include "ir/func_graph.h"

namespace mindspore::lite {
class AnfExporter {
 public:
  AnfExporter() = default;
  virtual ~AnfExporter() = default;
  schema::MetaGraphT *Export(const FuncGraphPtr &funcGraph);
  void SetOpOutputNode(const CNodePtr &cnode, const std::vector<schema::TensorT *> &outputTensors,
                       schema::MetaGraphT *graph, schema::CNodeT *fbnode);
  void SetOpInputNode(const CNodePtr &cnode, schema::MetaGraphT *meta_graph, schema::CNodeT *fbNode);
  void RemoveIfMakeTuple(const CNodePtr &cnode);
  bool RemoveIfTupleGetItem(const CNodePtr &cnode);
  bool AddOutPutIfReturn(const std::unique_ptr<schema::MetaGraphT> &metaGraphT, const CNodePtr &cnode);
 private:
  std::map<std::string, int> nodeIdMap;
  std::vector<schema::CNodeT *> graphInputNodes;
  std::map<std::string, int>  mapRemoveGetItem_;
};

schema::MetaGraphT *Export(const FuncGraphPtr &funcGraph);
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_ANF_EXPORTER_ANF_EXPORTER_H_
