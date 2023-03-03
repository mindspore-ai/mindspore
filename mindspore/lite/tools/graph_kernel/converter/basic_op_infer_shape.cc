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

#include "ops/core_ops.h"
#include "backend/common/graph_kernel/core/graph_kernel_callback.h"
#include "utils/anf_utils.h"
#include "src/common/ops/anf_utils.h"
#include "src/common/primitive_t_utils.h"
#include "src/common/ops/populate/populate_register.h"
#include "tools/optimizer/graph/lite_tensor_extractor.h"
#include "src/litert/infer_manager.h"

namespace mindspore::graphkernel {
inline mindspore::Format FormatStringToEnum(const std::string &format) {
  std::unordered_map<std::string, mindspore::Format> format_converter = {{kOpFormat_NHWC, mindspore::NHWC},
                                                                         {kOpFormat_NCHW, mindspore::NCHW}};
  auto iter = format_converter.find(format);
  if (iter == format_converter.end()) {
    MS_LOG(WARNING) << "Unsupported format [" << format << "] in GraphKernel";
    return mindspore::DEFAULT_FORMAT;
  }
  return iter->second;
}

void ExtractInputs(const CNodePtr &cnode, std::vector<TensorPtr> *inputs_holder, std::vector<lite::Tensor *> *inputs) {
  std::vector<TensorPtr> const_inputs;
  size_t const_index = 0;
  if (opt::LiteTensorExtractor::GetCNodeConstInput(cnode, &const_inputs, converter::kFmkTypeMs, false, false) !=
      lite::RET_OK) {
    MS_LOG(ERROR) << "get const inputs failed.";
    return;
  }
  auto cb = Callback::Instance();
  for (size_t index = 1; index < cnode->inputs().size(); index++) {
    if (cnode->input(index)->isa<CNode>()) {
      std::vector<int> shape;
      ShapeVector shp = cb->GetInputShape(cnode, index - 1);
      (void)std::transform(shp.begin(), shp.end(), std::back_inserter(shape), LongToInt);
      auto format = cb->GetInputFormat(cnode, index - 1);
      (void)inputs_holder->emplace_back(
        std::make_shared<lite::Tensor>(cb->GetInputType(cnode, index - 1), shape, FormatStringToEnum(format)));
    } else {
      if (const_index >= const_inputs.size()) {
        MS_LOG(WARNING) << "const_index " << const_index << " is out of range of const_inputs " << const_inputs.size();
      } else {
        (void)inputs_holder->emplace_back(const_inputs[const_index++]);
      }
    }
  }
  (void)std::transform(inputs_holder->cbegin(), inputs_holder->cend(), std::back_inserter(*inputs),
                       [](const TensorPtr &input) { return input.get(); });
}

void ExtractOutputs(const CNodePtr &cnode, std::vector<TensorPtr> *out_holder, std::vector<lite::Tensor *> *outputs) {
  auto cb = Callback::Instance();
  size_t output_num = AnfUtils::GetOutputTensorNum(cnode);
  for (size_t index = 0; index < output_num; index++) {
    auto format = cb->GetOutputFormat(cnode, index);
    (void)out_holder->emplace_back(
      std::make_shared<lite::Tensor>(cb->GetOutputType(cnode, index), std::vector<int>(), FormatStringToEnum(format)));
  }
  (void)std::transform(out_holder->cbegin(), out_holder->cend(), std::back_inserter(*outputs),
                       [](const TensorPtr &output) { return output.get(); });
}

void BasicOpInferShape::InferShapeRealKernel(const CNodePtr &cnode) {
  auto anf_prim = GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(anf_prim);
  (void)anf_prim->AddAttr(opt::kInferDone, MakeValue<bool>(false));

  std::vector<TensorPtr> inputs_holder;
  std::vector<lite::Tensor *> inputs;
  ExtractInputs(cnode, &inputs_holder, &inputs);

  std::vector<TensorPtr> outputs_holder;
  std::vector<lite::Tensor *> outputs;
  ExtractOutputs(cnode, &outputs_holder, &outputs);

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
    MS_LOG(WARNING) << "infer shape failed. node: " << cnode->fullname_with_scope();
  }
}

void BasicOpInferShape::InsertAbstract(const CNodePtr &cnode) {
  if (IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem)) {
    auto input2 = cnode->input(kInputNodeOutputIndexInTupleGetItem);
    auto item_idx = LongToSize(AnfUtils::GetIntValue(input2));
    auto abs_tuple = dyn_cast<abstract::AbstractTuple>(AnfUtils::VisitKernel(cnode, item_idx).first->abstract());
    MS_EXCEPTION_IF_NULL(abs_tuple);
    cnode->set_abstract(abs_tuple->elements()[item_idx]);
    return;
  }
  if (IsOneOfPrimitiveCNode(cnode, {prim::kPrimDepend, prim::kPrimLoad, prim::kPrimUpdateState})) {
    cnode->set_abstract(cnode->input(1)->abstract());
    return;
  }
}

void BasicOpInferShape::InferShape(const CNodePtr &cnode) {
  if (AnfUtils::IsRealKernel(cnode)) {
    InferShapeRealKernel(cnode);
  } else {
    InsertAbstract(cnode);
  }
}
}  // namespace mindspore::graphkernel
