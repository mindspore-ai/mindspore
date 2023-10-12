/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "backend/common/graph_kernel/core/graph_kernel_callback.h"
#include "utils/anf_utils.h"
#include "src/common/ops/anf_utils.h"
#include "src/common/primitive_t_utils.h"
#include "src/common/ops/populate/populate_register.h"
#include "tools/optimizer/graph/lite_tensor_extractor.h"
#include "src/litert/infer_manager.h"

namespace mindspore::graphkernel {
namespace {
void SetAbstractShape(const abstract::AbstractBasePtr &abs, const BaseShapePtr &shape) {
  MS_EXCEPTION_IF_NULL(abs);
  abs->set_shape(shape);
}

void SetAbstract(const CNodePtr &cnode) {
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

BaseShapePtr AllGatherInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  if (input_args.empty()) {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(primitive);
  auto shape_ptr = CheckAndConvertUtils::GetTensorInputShape(primitive->name(), input_args, 0);
  MS_EXCEPTION_IF_NULL(shape_ptr);
  auto x_shape = shape_ptr->shape();
  auto rank_list = primitive->GetAttr("rank_list");
  if (rank_list->isa<ValueSequence>()) {
    auto rank_list_ptr = rank_list->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(rank_list_ptr);
    auto out_shape = x_shape;
    if (!out_shape.empty() && out_shape[0] > 0) {
      out_shape[0] *= SizeToLong(rank_list_ptr->size());
    }
    return std::make_shared<abstract::Shape>(out_shape);
  }
  return nullptr;
}

BaseShapePtr ShapeInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  if (input_args.empty()) {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(primitive);
  auto shape_ptr = CheckAndConvertUtils::GetTensorInputShape(primitive->name(), input_args, 0);
  MS_EXCEPTION_IF_NULL(shape_ptr);
  auto x_shape = shape_ptr->shape();
  int64_t rank = IsDynamicRank(x_shape) ? -1 : SizeToLong(x_shape.size());
  return std::make_shared<abstract::Shape>(ShapeVector{rank});
}

AbstractBasePtrList RectifyBatchMatMul(const AbstractBasePtrList &orig_abs_list) {
  AbstractBasePtrList abs_list = orig_abs_list;
  if (abs_list.size() < kIndex2) {
    return abs_list;
  }
  auto shape0_ptr = CheckAndConvertUtils::GetTensorInputShape("BatchMatMul", orig_abs_list, 0);
  MS_EXCEPTION_IF_NULL(shape0_ptr);
  auto x0_shape = shape0_ptr->shape();
  auto shape1_ptr = CheckAndConvertUtils::GetTensorInputShape("BatchMatMul", orig_abs_list, 1);
  MS_EXCEPTION_IF_NULL(shape1_ptr);
  auto x1_shape = shape1_ptr->shape();
  if (x0_shape.size() < x1_shape.size()) {
    ShapeVector new_x0_shape(x1_shape.size() - x0_shape.size(), 1);
    new_x0_shape.insert(new_x0_shape.end(), x0_shape.begin(), x0_shape.end());
    abs_list[0] = orig_abs_list[0]->Clone();
    SetAbstractShape(abs_list[0], std::make_shared<abstract::Shape>(new_x0_shape));
  } else if (x0_shape.size() > x1_shape.size()) {
    ShapeVector new_x1_shape(x0_shape.size() - x1_shape.size(), 1);
    new_x1_shape.insert(new_x1_shape.end(), x1_shape.begin(), x1_shape.end());
    abs_list[1] = orig_abs_list[1]->Clone();
    SetAbstractShape(abs_list[1], std::make_shared<abstract::Shape>(new_x1_shape));
  }
  return abs_list;
}

using OpInferFunc = std::function<BaseShapePtr(const PrimitivePtr &, const std::vector<AbstractBasePtr> &)>;
using OpRectifyFunc = std::function<AbstractBasePtrList(const AbstractBasePtrList &)>;
}  // namespace

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
  if (opt::LiteTensorExtractor::GetCNodeConstInputs(cnode, converter::kFmkTypeMs, false, false, &const_inputs) !=
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

void BasicOpInferShape::InsertAbstract(const CNodePtr &cnode) { SetAbstract(cnode); }

void BasicOpInferShape::InferShape(const CNodePtr &cnode) {
  if (AnfUtils::IsRealKernel(cnode)) {
    InferShapeRealKernel(cnode);
  } else {
    InsertAbstract(cnode);
  }
}

bool DynOpInferShape::HasDynamicShapeInput(const FuncGraphPtr &func_graph) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &params = func_graph->parameters();
  for (const auto &param : params) {
    if (param == nullptr) {
      continue;
    }
    auto param_shape = param->Shape();
    if (param_shape != nullptr && param_shape->IsDynamic()) {
      return true;
    }
  }
  return false;
}

bool DynOpInferShape::InferShapeRealKernel(const CNodePtr &cnode) const {
  auto prim = GetCNodePrimitive(cnode);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr for cnode: " << cnode->fullname_with_scope();
    return false;
  }
  // collect op inputs abstract
  AbstractBasePtrList abs_list;
  abs_list.reserve(cnode->size());
  for (size_t i = 1; i < cnode->size(); ++i) {
    const auto &input = cnode->input(i);
    if (input == nullptr) {
      continue;
    }
    auto abs = input->abstract();
    if (abs == nullptr && input->isa<ValueNode>()) {
      abs = input->cast<ValueNodePtr>()->value()->ToAbstract();
    }
    if (abs == nullptr) {
      MS_LOG(ERROR) << "inputs[" << i << "] has no abstract for cnode: " << cnode->fullname_with_scope();
      return false;
    }
    abs_list.push_back(abs);
  }
  // some op has no C++ infer
  static std::unordered_map<std::string, OpInferFunc> infer_func_map{{"AllGather", AllGatherInferShape}};
  auto prim_name = prim->name();
  auto iter = infer_func_map.find(prim_name);
  if (iter != infer_func_map.end()) {
    SetAbstractShape(cnode->abstract(), iter->second(prim, abs_list));
    return true;
  }
  // core/ops 'Shape' returns AbstractTuple, which will change the original abstract type
  if (prim_name == "Shape" && cnode->abstract()->isa<abstract::AbstractTensor>()) {
    SetAbstractShape(cnode->abstract(), ShapeInferShape(prim, abs_list));
    return true;
  }
  // some op's abstract does not satisfy core/ops infer
  if (prim_name == "StridedSlice" || prim_name == "PromptFlashAttention") {
    return true;
  }
  static std::unordered_map<std::string, OpRectifyFunc> rectify_map{{"BatchMatMul", RectifyBatchMatMul}};
  auto rec_iter = rectify_map.find(prim_name);
  if (rec_iter != rectify_map.end()) {
    abs_list = rec_iter->second(abs_list);
  }
  auto found = abstract::GetPrimitiveInferImpl(prim);
  if (found.has_value() && found.value().IsImplInferShapeAndType()) {
    auto infer_impl = found.value();
    SetAbstractShape(cnode->abstract(), infer_impl.InferShape(prim, abs_list));
    return true;
  }
  MS_LOG(ERROR) << "Can not find infer shape function for " << prim_name;
  return false;
}

bool DynOpInferShape::InferShape(const CNodePtr &cnode) const {
  if (AnfUtils::IsRealKernel(cnode)) {
    if (!InferShapeRealKernel(cnode)) {
      MS_LOG(ERROR) << "infer shape failed for cnode: " << cnode->fullname_with_scope();
      return false;
    }
  } else {
    if (IsPrimitiveCNode(cnode, prim::kPrimLoad)) {
      return true;
    }
    SetAbstract(cnode);
  }
  return true;
}

bool DynOpInferShape::Run(const FuncGraphPtr &func_graph) {
  if (!HasDynamicShapeInput(func_graph)) {
    return false;
  }
  MS_LOG(INFO) << "Dynamic shape infer for func graph: " << func_graph->ToString();
  auto nodes = TopoSort(func_graph->output());
  for (const auto &node : nodes) {
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      if (!InferShape(cnode)) {
        break;
      }
    }
  }
  return true;
}
}  // namespace mindspore::graphkernel
