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

#ifndef MINDSPORE_LITE_SRC_TRAIN_MODEL_IMPL_H_
#define MINDSPORE_LITE_SRC_TRAIN_MODEL_IMPL_H_

#include <string>
#include <map>
#include <memory>
#include <vector>
#include "schema/model_generated.h"
#include "src/ops/ops.h"
#include "ir/func_graph.h"

namespace mindspore::lite {
namespace train {
class ModelImpl : public FuncGraph {
 public:
  static std::shared_ptr<ModelImpl> Import(const char *model_buf, size_t size);  // { return NULL; };
  ModelImpl() = default;
  explicit ModelImpl(const schema::MetaGraph *graph) : meta_graph(graph) {}
  ~ModelImpl() override = default;
  const lite::Primitive *GetOp(const std::string &name) const;
  const schema::MetaGraph *GetMetaGraph() const;
  void FreeMetaGraph();
  int BuildOps();

  void AddCNodeInputOutput(std::string name, const std::vector<int> &input, const std::vector<int> &output) {
    std::vector<int> *tuple = new std::vector<int>[2];
    tuple[0] = input;
    tuple[1] = output;
    connectivity_[name] = tuple;
  }
  std::vector<int> *GetCNodeInputOutputIndices(std::string name) { return connectivity_[name]; }
  void AddAnfNode(int id, AnfNodePtr anf_ptr) { tensors_[id] = anf_ptr; }
  AnfNodePtr GetAnfNode(int id) { return tensors_[id]; }

 protected:
  lite::Primitive *CopyPrimitive(const schema::Primitive *srcPrim);

 protected:
  const schema::MetaGraph *meta_graph = nullptr;
  std::map<int, AnfNodePtr> tensors_;
  std::map<std::string, std::vector<int> *> connectivity_;
  std::map<std::string, lite::Primitive *> ops;
};
}  // namespace train
using ModelImpl = mindspore::lite::train::ModelImpl;
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_TRAIN_MODEL_IMPL_H_
