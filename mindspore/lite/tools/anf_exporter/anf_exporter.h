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

#ifndef MINDSPORE_LITE_TOOLS_ANF_EXPORTER_ANF_EXPORTER_H_
#define MINDSPORE_LITE_TOOLS_ANF_EXPORTER_ANF_EXPORTER_H_

#include <map>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <set>
#include <list>
#include "schema/inner/model_generated.h"
#include "ops/primitive_c.h"
#include "ir/func_graph.h"
#include "tools/anf_exporter/fetch_content.h"
#include "tools/converter/converter_context.h"
#include "tools/converter/converter_flags.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/common/node_util.h"

using mindspore::ops::PrimitiveC;

namespace mindspore::lite {

constexpr const int kMainGraphIndex = 0;
constexpr const int kFirstDataIndex = 1;
constexpr const int kSecondDataIndex = 2;
constexpr const int kPrimIndex = 0;
constexpr const int kSwitchFalseIndex = 3;

class AnfExporter {
 public:
  AnfExporter() = default;
  virtual ~AnfExporter() = default;
  schema::MetaGraphT *Export(const FuncGraphPtr &func_graph, bool keep_graph = false, bool copy_primitive = false,
                             bool train_flag = false);
  int SetOpOutputNode(const CNodePtr &cnode, const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                      schema::CNodeT *fb_node);
  int SetOpInputNode(const CNodePtr &cnode, const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                     schema::CNodeT *fb_node);

 protected:
  int ConvertInputCNode(const std::shared_ptr<AnfNode> &input_anode, schema::CNodeT *output_cnode);
  int ConvertInputCNodeCommonOp(const AnfNodePtr &input_anode, schema::CNodeT *output_cnode);
  int ConvertInputParameter(const CNodePtr &cnode, size_t index, const PrimitivePtr &primitive,
                            const std::unique_ptr<schema::MetaGraphT> &meta_graphT, schema::CNodeT *op_node);
  int ConvertInputValueNode(const CNodePtr &cnode, size_t index, const PrimitivePtr &primitive,
                            const std::unique_ptr<schema::MetaGraphT> &meta_graphT, schema::CNodeT *op_node);
  int SetSubGraphInputIndex(const std::unique_ptr<schema::MetaGraphT> &meta_graphT, const size_t &subgraph_index);
  int SetSubGraphOutputIndex(const CNodePtr &cnode, size_t subgraph_index,
                             const std::unique_ptr<schema::MetaGraphT> &meta_graphT, schema::CNodeT *return_node);
  static int SetPostTrainOutputTensorType(const std::unique_ptr<schema::MetaGraphT> &meta_graph,
                                          const std::shared_ptr<mindspore::Primitive> &primitive,
                                          const std::unique_ptr<schema::CNodeT> &dst_node);
  static int ConvertQuantParam(const std::unique_ptr<schema::MetaGraphT> &meta_graph,
                               const std::shared_ptr<mindspore::Primitive> &primitive,
                               const std::unique_ptr<schema::CNodeT> &dst_node);
  int Anf2Fb(const FuncGraphPtr &func_graph, const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
             const size_t &subgraph_index, const bool &keep_graph, const bool &copy_primitive);
  int ExportSubgraph(const FuncGraphPtr &func_graph, const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                     bool keep_graph, bool copy_primitive, const std::shared_ptr<AnfNode> &partial_anode = nullptr);
  static CNodePtr CreateCallCnode(const FuncGraphPtr &fg, const AnfNodePtr &cnode);
  static CNodePtr CreatePartialCnode(const FuncGraphPtr &fg, const AnfNodePtr &node);
  bool HasExported(const FuncGraphPtr &func_graph);
  int ExportPartialNode(const std::unique_ptr<schema::MetaGraphT> &meta_graphT, const bool &keep_graph,
                        const bool &copy_primitive, const CNodePtr &partial_cnode,
                        const std::unique_ptr<schema::CNodeT> &schema_cnode);
  std::list<CNodePtr> InsertCallNode(const FuncGraphPtr &func_graph);
  int SetMetaGraphInput(const FuncGraphPtr &func_graph, const std::unique_ptr<schema::MetaGraphT> &meta_graphT);
  int SetMetaGraphOutput(const FuncGraphPtr &func_graph, const std::unique_ptr<schema::MetaGraphT> &meta_graphT);
  int CreateNewTensorForParameter(const std::unique_ptr<schema::MetaGraphT> &meta_graphT, const AnfNodePtr &input);

 private:
  // Key is a pair of node and its output id. Value is the mapped tensor id of meta_graph.
  std::map<std::pair<AnfNodePtr, int>, int> node_id_map_;
  // The first item is FuncGraph which has been exported, the second item is the subgraph index in meta_graph
  std::map<FuncGraphPtr, size_t> fg_subgraph_map_;
  std::vector<AnfNodePtr> graph_inputs_;
  std::set<AnfNodePtr> graph_inputs_has_exported_;
  std::map<AnfNodePtr, int> graph_inputs_map_;
  uint32_t node_idx_ = 0;
  bool train_flag_ = false;
};
// by default, copy_primitive is false, which means that the MetaGraph and func_graph share the same schema::PrimitiveT.
// but in PostQuantization, the func_graph need to transfer to MetaGraph first and do MetaGraph pass, which may modify
// the schema::PrimitiveT and cause bug; If all the passes have been done in func_graph, every thing would be simple
// and clear.
schema::MetaGraphT *Export(const FuncGraphPtr &func_graph, bool keep_graph = false, bool copy_primitive = false,
                           bool train_flag = false);
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_TOOLS_ANF_EXPORTER_ANF_EXPORTER_H_
