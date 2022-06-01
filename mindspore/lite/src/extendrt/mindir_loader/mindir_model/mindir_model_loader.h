/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_MINDIR_MODEL_MINDIR_MODEL_LOADER_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_MINDIR_MODEL_MINDIR_MODEL_LOADER_H_

#include <memory>
#include <string>

#include "extendrt/mindir_loader/mindir_model/mindir_model.h"
#include "extendrt/mindir_loader/model_loader.h"
#include "proto/mind_ir.pb.h"
#include "ops/base_operator.h"

namespace mindspore::infer::mindir {
class MindirModelLoader : public ModelLoader {
 public:
  MindirModelLoader() {}
  ~MindirModelLoader() = default;

  AbstractBaseModel *ImportModel(const char *model_buf, size_t size, bool take_buf) override;

 private:
  // int InitModelBuffer(const char *model_buf, size_t size, bool take_buf);
  bool ConvertModel(const mind_ir::ModelProto &model_proto);
  bool ConvertPrimitives(const mind_ir::ModelProto &model_proto);
  bool ConvertGraph(const mind_ir::GraphProto &graph_proto, LiteGraph::SubGraph *sub_graph = nullptr,
                    bool is_main_graph = false);
  bool ConvertNodes(const mind_ir::GraphProto &graph_proto, LiteGraph::SubGraph *sub_graph = nullptr,
                    bool is_main_graph = false);
  bool ConvertTensors(const mind_ir::GraphProto &graph_proto, LiteGraph::SubGraph *sub_graph = nullptr,
                      bool is_main_graph = false);
  std::shared_ptr<void> MakePrimitiveC(const std::string &node_type);

 private:
  MindirModel *model_;
  mindspore::HashMap<std::string, std::shared_ptr<mindspore::ops::BaseOperator>> all_operators_;
  mindspore::HashMap<std::string, int32_t> tensor_index_map_;
  int tensor_count_;
  int node_count_;
};
}  // namespace mindspore::infer::mindir

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_MINDIR_MODEL_MINDIR_MODEL_LOADER_H_
