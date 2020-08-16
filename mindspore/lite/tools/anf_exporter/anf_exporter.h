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
#include "src/ir/primitive_t_value.h"
#include "ir/func_graph.h"

namespace mindspore::lite {
class AnfExporter {
 public:
  AnfExporter() = default;
  virtual ~AnfExporter() = default;
  schema::MetaGraphT *Export(const FuncGraphPtr &func_graph);
  void SetOpOutputNode(const CNodePtr &cnode, const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                       schema::CNodeT *fb_node);
  int SetOpInputNode(const CNodePtr &cnode, const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                     schema::CNodeT *fb_node);
  void RemoveIfMakeTuple(const CNodePtr &cnode);
  bool RemoveIfTupleGetItem(const CNodePtr &cnode);
  bool AddOutPutIfReturn(const std::unique_ptr<schema::MetaGraphT> &meta_graphT, const CNodePtr &cnode);

 protected:
  void ConvertInputCNode(const std::shared_ptr<AnfNode> input_anode, schema::CNodeT *output_cnode);
  int ConvertInputParameter(const std::shared_ptr<AnfNode> input_anode, size_t anode_index,
                            const std::unique_ptr<schema::MetaGraphT> &meta_graphT, schema::CNodeT *output_cnode);
  int ConvertInputValueNode(std::shared_ptr<AnfNode> input_anode,
                            const std::unique_ptr<schema::MetaGraphT> &meta_graphT, schema::CNodeT *output_cnode);
  void SetGraphInputIndex(const std::unique_ptr<schema::MetaGraphT> &meta_graphT);
  bool IsPrimitiveCNode(const AnfNodePtr &node, schema::PrimitiveType type);
  int ConvertQuantParam(const std::unique_ptr<schema::MetaGraphT> &meta_graph,
                        const std::shared_ptr<PrimitiveTValue> primitive,
                        const std::unique_ptr<schema::CNodeT> &dst_node);

 private:
  std::map<std::string, int> node_id_map_;
  std::vector<schema::CNodeT *> graph_input_nodes_;
  std::map<std::string, int> map_remove_get_item_;
};

schema::MetaGraphT *Export(const FuncGraphPtr &func_graph);
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_ANF_EXPORTER_ANF_EXPORTER_H_
