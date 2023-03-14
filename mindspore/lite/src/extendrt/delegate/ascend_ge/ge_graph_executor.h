/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_GRAPH_EXECUTOR_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_GRAPH_EXECUTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <map>

#include "include/api/context.h"
#include "include/model.h"
#include "include/transform/graph_ir/types.h"
#include "extendrt/session/lite_graph_executor.h"
#include "common/config_infos.h"

namespace mindspore {
class GeGraphExecutor : public LiteGraphExecutor {
 public:
  GeGraphExecutor() = default;
  ~GeGraphExecutor() = default;
  GeGraphExecutor(const std::shared_ptr<mindspore::Context> &context, const ConfigInfos &config_infos)
      : context_(context), config_infos_(config_infos) {}

  bool CompileGraph(const FuncGraphPtr &graph, const std::map<string, string> &compile_options) override;

  bool RunGraph(const FuncGraphPtr &graph, const std::vector<tensor::Tensor> &inputs,
                std::vector<tensor::Tensor> *outputs, const std::map<string, string> &compile_options) override;

  bool Resize(const FuncGraphPtr &, const std::vector<tensor::Tensor> &inputs,
              const std::vector<ShapeVector> &dims) override {
    return true;
  }

  std::vector<tensor::Tensor> GetInputInfos(const FuncGraphPtr &) override;

  std::vector<tensor::Tensor> GetOutputInfos(const FuncGraphPtr &) override;

  static FuncGraphPtr BuildDFGraph(const FuncGraphPtr &anf_graph, const transform::TensorOrderMap &init_inputs_map,
                                   bool export_air, const ConfigInfos config_infos = {});

 private:
  const std::shared_ptr<mindspore::Context> context_;
  ConfigInfos config_infos_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_GRAPH_EXECUTOR_H_
