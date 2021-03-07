/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_INFERSHAPE_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_INFERSHAPE_PASS_H_
#include <vector>
#include <memory>
#include <string>
#include "tools/converter/converter_flags.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "backend/optimizer/common/pass.h"
#include "mindspore/lite/src/tensor.h"
#include "mindspore/lite/src/tensorlist.h"
#include "mindspore/lite/include/errorcode.h"
using mindspore::lite::STATUS;
using mindspore::lite::converter::FmkType;
namespace mindspore::opt {
class InferShapePass : public Pass {
 public:
  InferShapePass() : Pass("infershape_pass") {}
  ~InferShapePass() override = default;
  bool Run(const FuncGraphPtr &graph) override;
  void SetFmkType(FmkType fmkType) { this->fmk_type = fmkType; }

 private:
  void FreeTensors(std::vector<lite::Tensor *> *tensors);
  abstract::AbstractTensorPtr ConvertLiteTensorToAbstractTensor(lite::Tensor *tensor);
  STATUS GetCNodeInputTensors(const CNodePtr &cnode, std::vector<lite::Tensor *> *input_tensors);
  STATUS GetCNodeOutputTensors(const CNodePtr &cnode, std::vector<lite::Tensor *> *output_tensors);
  STATUS SetParameterAbstract(const ParameterPtr &parameter);
  STATUS SetCNodeAbstract(const std::vector<lite::Tensor *> &output_tensors, const std::shared_ptr<CNode> &cnode);
  int StrIsContain(const std::vector<std::string> &total, const std::string &aim);
  int SetSubGraphInputsAbstract(const CNodePtr &cnode, const FuncGraphPtr &func_graph);

 private:
  FmkType fmk_type = lite::converter::FmkType_ONNX;
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_INFERSHAPE_PASS_H_
