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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_NODE_INFERSHAPE_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_NODE_INFERSHAPE_H_

#include <vector>
#include <memory>
#include <string>
#include <map>
#include "schema/inner/model_generated.h"
#include "src/tensor.h"
#include "tools/anf_exporter/fetch_content.h"
#include "tools/converter/converter_flags.h"
#include "tools/optimizer/common/format_utils.h"

using mindspore::converter::FmkType;
namespace mindspore {
namespace opt {
class NodeInferShape {
 public:
  explicit NodeInferShape(FmkType fmk_type = converter::kFmkTypeMs, bool train_flag = false)
      : fmk_type_(fmk_type), train_flag_(train_flag) {}
  virtual ~NodeInferShape() = default;
  void Init(FmkType fmk_type, bool train_flag) {
    fmk_type_ = fmk_type;
    train_flag_ = train_flag;
  }
  STATUS InferShape(const CNodePtr &cnode);
  bool JudgeOpSupportInfer(const CNodePtr &cnode);
  std::vector<int> GetInputShape(const CNodePtr &cnode, size_t index);
  std::vector<int> GetIntVecInput(const CNodePtr &cnode, size_t index);
  STATUS GetCNodeInputTensors(const CNodePtr &cnode, std::vector<lite::Tensor *> *inputs);

 private:
  STATUS GetCNodeConstInput(const CNodePtr &cnode, std::vector<lite::Tensor *> *const_ms_inputs);
  STATUS GetCNodeVarInput(const CNodePtr &cnode, std::vector<lite::Tensor *> *var_ms_inputs);
  lite::Tensor *GetCNodeTensorListVarInput(const lite::DataInfo &data_info);
  STATUS GetCNodeOutputTensors(const CNodePtr &cnode, std::vector<lite::Tensor *> *outputs);
  STATUS ConvertToLiteTensor(const std::vector<lite::DataInfo> &data_infos, std::vector<lite::Tensor *> *tensors);
  STATUS SetCNodeAbstract(const std::shared_ptr<CNode> &cnode, const std::vector<lite::Tensor *> &outputs, int status);
  abstract::AbstractBasePtr ConvertLiteTensorToAbstract(lite::Tensor *tensor);
  abstract::AbstractBasePtr ConvertTensorListToAbstract(lite::Tensor *tensor);
  FmkType fmk_type_{converter::kFmkTypeMs};
  bool train_flag_{false};
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_NODE_INFERSHAPE_H_
