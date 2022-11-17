/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_LITE_EXPORTER_FETCH_CONTENT_H_
#define MINDSPORE_LITE_TOOLS_LITE_EXPORTER_FETCH_CONTENT_H_

#include <string>
#include <vector>
#include <map>
#include "ir/primitive.h"
#include "ir/func_graph.h"
#include "src/common/utils.h"
#include "nnacl/op_base.h"
#include "include/registry/converter_context.h"

namespace mindspore {
namespace lite {
struct DataInfo {
  bool enable_huffman_code_;
  int compress_type_;
  int format_;
  int data_type_;
  int node_type_;
  std::vector<int> shape_;
  std::vector<uint8_t> data_;
  void *data_ptr_;
  DataInfo()
      : enable_huffman_code_(false),
        compress_type_(kNoCompression),
        format_(0),
        data_type_(0),
        node_type_{0},
        data_ptr_(nullptr) {}
};

int FetchFromDefaultParam(const ParameterPtr &param_node, const converter::FmkType &fmk_type, DataInfo *data_info,
                          bool copy_data);

int FetchDataFromParameterNode(const CNodePtr &cnode, size_t index, converter::FmkType fmk_type, DataInfo *data_info,
                               bool copy_data);

int FetchDataFromValueNode(const CNodePtr &cnode, size_t index, converter::FmkType fmk_type, bool train_flag,
                           DataInfo *data_info, bool copy_data);

int FetchDataFromCNode(const CNodePtr &cnode, size_t index, DataInfo *data_info);

int FetchConstData(const CNodePtr &cnode, size_t index, converter::FmkType fmk_type, DataInfo *data_info,
                   bool copy_data);

int RemoveIfDepend(const CNodePtr &cnode);

int RemoveIfMakeTuple(const CNodePtr &cnode);
int GetFlattenInputsIfMakeTuple(const CNodePtr &cnode, std::vector<AnfNodePtr> *inputs, bool *has_make_tuple);

// Notes:The op_parameter allocates memory through malloc, and may need to manually free op_parameter.
int FetchOpParameterFromNode(const AnfNodePtr &node, OpParameter **op_parameter);

int FetchOpParameterFromFuncGraph(const FuncGraphPtr &func_graph, std::map<std::string, OpParameter *> *op_parameters);
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_LITE_EXPORTER_FETCH_CONTENT_H_
