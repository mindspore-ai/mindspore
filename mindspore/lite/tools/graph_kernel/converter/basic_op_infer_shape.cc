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

#include "tools/graph_kernel/converter/basic_op_infer_shape.h"

#include <utility>
#include <algorithm>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

#include "common/graph_kernel/core/graph_kernel_callback.h"
#include "utils/anf_utils.h"
#include "src/common/ops/anf_utils.h"
#include "src/common/primitive_t_utils.h"
#include "src/common/ops/populate/populate_register.h"
#include "tools/optimizer/graph/lite_tensor_extractor.h"
#include "src/litert/infer_manager.h"

namespace mindspore::graphkernel {
void BasicOpInferShape::InferShape(const CNodePtr &cnode) {
  std::unordered_map<std::string, mindspore::Format> format_converter = {{kOpFormat_NHWC, mindspore::NHWC},
                                                                         {kOpFormat_NCHW, mindspore::NCHW}};
  auto anf_prim = GetValueNode<std::shared_ptr<Primitive>>(cnode->input(0));
  if (anf_prim == nullptr) {
    MS_LOG(DEBUG) << "primitive is nullptr";
    return;
  }
  (void)anf_prim->AddAttr(opt::kInferDone, MakeValue<bool>(false));
  auto cb = Callback::Instance();
  std::vector<TensorPtr> inputs_ptr;
  std::vector<TensorPtr> const_inputs;
  size_t const_index = 0;
  if (opt::LiteTensorExtractor::GetCNodeConstInput(cnode, &const_inputs, converter::kFmkTypeMs, false, false) !=
      lite::RET_OK) {
    MS_LOG(ERROR) << "get const inputs failed.";
    return;
  }
  for (size_t index = 1; index < cnode->inputs().size(); index++) {
    if (cnode->input(index)->isa<CNode>()) {
      std::vector<int> shape;
      ShapeVector shp = cb->GetInputShape(cnode, index - 1);
      (void)std::transform(shp.begin(), shp.end(), std::back_inserter(shape), LongToInt);
      auto format = cb->GetInputFormat(cnode, index - 1);
      if (format != kOpFormat_NHWC && format != kOpFormat_NCHW) {
        MS_LOG(ERROR) << "Graph Kernel only support NHWC and NCHW";
        return;
      }
      (void)inputs_ptr.emplace_back(
        std::make_shared<lite::Tensor>(cb->GetInputType(cnode, index - 1), shape, format_converter[format]));
    } else {
      (void)inputs_ptr.emplace_back(const_inputs[const_index++]);
    }
  }

  std::vector<TensorPtr> outputs_ptr;
  for (size_t index = 0; index < AnfUtils::GetOutputTensorNum(cnode); index++) {
    auto format = cb->GetOutputFormat(cnode, index);
    if (format != kOpFormat_NHWC && format != kOpFormat_NCHW) {
      MS_LOG(ERROR) << "Graph Kernel only support NHWC and NCHW";
      return;
    }
    (void)outputs_ptr.emplace_back(
      std::make_shared<lite::Tensor>(cb->GetOutputType(cnode, index), std::vector<int>(), format_converter[format]));
  }

  auto prim_t = lite::GetPrimitiveT(cnode->input(0));
  if (prim_t == nullptr) {
    MS_LOG(DEBUG) << "prim_t is nullptr";
    return;
  }
  const size_t INITIAL_SIZE = 1024;
  flatbuffers::FlatBufferBuilder fbb(INITIAL_SIZE);
  auto prim = lite::ConvertToPrimitive(prim_t.get(), &fbb);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "get primitive failed.";
    fbb.Clear();
    return;
  }
  std::vector<lite::Tensor *> inputs;
  (void)std::transform(inputs_ptr.begin(), inputs_ptr.end(), std::back_inserter(inputs),
                       [](const TensorPtr &input) { return input.get(); });
  std::vector<lite::Tensor *> outputs;
  (void)std::transform(outputs_ptr.begin(), outputs_ptr.end(), std::back_inserter(outputs),
                       [](const TensorPtr &output) { return output.get(); });
  auto ret = lite::KernelInferShape(inputs, outputs, prim, {}, lite::SCHEMA_CUR);
  if (ret == lite::RET_NOT_SUPPORT) {
    auto parameter_gen = lite::PopulateRegistry::GetInstance()->GetParameterCreator(
      static_cast<int>(prim->value_type()), lite::SCHEMA_CUR);
    if (parameter_gen == nullptr) {
      MS_LOG(ERROR) << "PopulateParameter return nullptr, type: " << schema::EnumNamePrimitiveType(prim->value_type());
      fbb.Clear();
      return;
    }
    auto parameter = parameter_gen(prim);
    if (parameter == nullptr) {
      MS_LOG(ERROR) << "parameter is nullptr.";
      fbb.Clear();
      return;
    }
    ret = lite::KernelInferShape(inputs, outputs, parameter);
    if (parameter->destroy_func_ != nullptr) {
      parameter->destroy_func_(parameter);
    }
    free(parameter);
    parameter = nullptr;
  }
  fbb.Clear();
  if (ret == lite::RET_OK) {
    (void)anf_prim->AddAttr(opt::kInferDone, MakeValue<bool>(true));
  }
  if (ret == lite::RET_OK || ret == lite::RET_INFER_INVALID) {
    (void)SetCNodeAbstract(cnode, outputs, ret);
  } else {
    MS_LOG(WARNING) << "infer shape failed.";
  }
}
}  // namespace mindspore::graphkernel
