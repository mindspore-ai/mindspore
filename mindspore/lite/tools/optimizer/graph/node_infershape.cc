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

#define USE_DEPRECATED_API
#include "tools/optimizer/graph/node_infershape.h"
#include <memory>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/comparison_ops.h"
#include "mindspore/core/ops/random_ops.h"
#include "src/common/primitive_t_utils.h"
#include "tools/common/node_util.h"
#include "tools/common/tensor_util.h"
#include "src/common/utils.h"
#include "src/common/ops/populate/populate_register.h"
#include "src/common/ops/anf_utils.h"
#include "src/litert/infer_manager.h"
#include "src/tensorlist.h"
#include "src/registry/kernel_interface_registry.h"
#include "tools/optimizer/graph/lite_tensor_extractor.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"
#include "tools/optimizer/format/to_nchw_format.h"
#include "tools/optimizer/format/to_nhwc_format.h"
#include "tools/common/graph_util.h"
#include "src/common/common.h"

namespace mindspore {
namespace opt {
static const std::unordered_set<PrimitivePtr> kNNACLToOpsInfer = {
  // arithmetic_self
  prim::kPrimAbs,
  prim::kPrimAsin,
  prim::kPrimAsinh,
  prim::kPrimACos,
  prim::kPrimAcosh,
  prim::kPrimAtanh,
  prim::kPrimCos,
  prim::kPrimCosh,
  prim::kPrimCeLU,
  prim::kPrimSeLU,
  prim::kPrimHSwish,
  prim::kPrimMatrixDeterminant,
  prim::kPrimLog,
  prim::kPrimLog1p,
  prim::kPrimSquare,
  prim::kPrimSqrt,
  prim::kPrimRsqrt,
  prim::kPrimSin,
  prim::kPrimSinh,
  prim::kPrimFloor,
  prim::kPrimCeil,
  prim::kPrimRound,
  prim::kPrimNeg,
  prim::kPrimReciprocal,
  prim::kPrimErf,
  prim::kPrimSign,
  prim::kPrimSoftsign,
  prim::kPrimMultinomial,
  // arithmetic
  prim::kPrimFloorDiv,
  prim::kPrimFloorMod,
  prim::kPrimLogicalAnd,
  prim::kPrimLogicalNot,
  prim::kPrimLogicalOr,
  prim::kPrimLogicalXor,
  prim::kPrimMaximum,
  prim::kPrimMinimum,
  prim::kPrimMod,
  prim::kPrimSquaredDifference,
  prim::kPrimLeftShift,
  prim::kPrimRightShift,
  prim::kPrimROIAlign,
  // tuple op
  prim::kPrimTupleGetItem,
  prim::kPrimMakeTuple,
  prim::kPrimMakeTupleV2,
  // nnacl/infer/common_infer.c
  prim::kPrimClip,
  prim::kPrimElu,
  prim::kPrimLeakyRelu,
  prim::kPrimLrn,
  prim::kPrimOnesLike,
  prim::kPrimReverseSequence,
  prim::kPrimReverseV2,
  prim::kPrimSmoothL1Loss,
  prim::kPrimZerosLike,
  // format op
  prim::kPrimResize,
  // compare op
  prim::kPrimEqual,
  prim::kPrimGreater,
  prim::kPrimGreaterEqual,
  prim::kPrimLess,
  prim::kPrimLessEqual,
  prim::kPrimNotEqual,

  prim::kPrimActivation,
  prim::kPrimArgMaxFusion,
  prim::kPrimArgMinFusion,
  prim::kPrimGLU,
  prim::kPrimGridSampler2D,
  prim::kPrimGridSampler3D,
  prim::kPrimDeformableConv2d,
  // grad op
  prim::kPrimActivationGrad,
  prim::kPrimAbsGrad,
  prim::kPrimBinaryCrossEntropyGrad,
  prim::kPrimLogGrad,
  prim::kPrimMaximumGrad,
  prim::kPrimMinimumGrad,
  prim::kPrimNegGrad,
  prim::kPrimRsqrtGrad,
  prim::kPrimSqrtGrad,
  prim::kPrimSmoothL1LossGrad,
  prim::kPrimGridSampler2D,
};

namespace {
constexpr int kInputChannal = 3;
constexpr size_t INITIAL_SIZE = 1024;
void RectifyFormat(const std::vector<lite::Tensor *> &inputs, FmkType fmk_type) {
  MS_ASSERT(cnode != nullptr);
  if (fmk_type != converter::kFmkTypeOnnx) {
    return;
  }
  for (auto &input : inputs) {
    auto shape = input->shape();
    if (shape.size() == kInputSizeFour && shape[kInputIndexThree] == kInputChannal && shape[1] == -1) {
      input->set_format(mindspore::NHWC);
    }
  }
}

tensor::TensorPtr NewTensorInfo(const lite::Tensor *tensor) {
  std::vector<int> shape(tensor->shape());
  std::vector<int64_t> shape_vector(shape.begin(), shape.end());
  auto tensor_info = std::make_shared<tensor::Tensor>(tensor->data_type(), shape_vector);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "new tensor::Tensor failed";
    return nullptr;
  }
  return tensor_info;
}

STATUS ConvertAbstract(const AbstractBasePtr &src_abs, AbstractBasePtr *dst_abs, bool change,
                       FormatTransNodeType perm) {
  if (SetAbstractTensorInfo(src_abs) != RET_OK) {
    MS_LOG(ERROR) << "SetAbstractTensorInfo failed";
    return lite::RET_ERROR;
  }
  *dst_abs = src_abs;
  if (change) {
    if (ConvertAbstractFormatShape(*dst_abs, perm) != RET_OK) {
      MS_LOG(ERROR) << "ConvertAbstractFormatShape failed";
      return lite::RET_ERROR;
    }
  }

  // change core/ops dynamic rank {-2} to Lite dynamic shape {-1}, will be removed after calling core/infer
  ShapeVector shape;
  if (opt::FetchShapeFromAbstract(*dst_abs, &shape) != RET_OK) {
    MS_LOG(ERROR) << "FetchShapeFromAbstract failed.";
    return RET_ERROR;
  }
  if (IsDynamicRank(shape)) {
    auto nnacl_dynamic_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeDimAny});
    (*dst_abs)->set_shape(nnacl_dynamic_shape);
  }
  return RET_OK;
}
}  // namespace

bool JudgeOpSupportNNACLInfer(const CNodePtr &cnode) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, false, "cnode is nullptr.");
  if (CheckPrimitiveType(cnode, prim::kPrimCustom)) {
    return true;
  }
  auto prim_t = lite::GetPrimitiveT(cnode->input(0));
  if (prim_t == nullptr) {
    return false;
  }
  auto parameter_gen =
    lite::PopulateRegistry::GetInstance()->GetParameterCreator(static_cast<int>(prim_t->value.type), lite::SCHEMA_CUR);
  if (parameter_gen == nullptr) {
    prim_t.reset();
    return false;
  }
  return true;
}

bool JudgeOpSupportOpsInfer(const CNodePtr &cnode) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, false, "cnode is nullptr.");
  for (const auto &type : kNNACLToOpsInfer) {
    if (CheckPrimitiveType(cnode, type)) {
      return true;
    }
  }
  return false;
}

bool NodeInferShape::JudgeOpSupportInfer(const CNodePtr &cnode) {
  return JudgeOpSupportOpsInfer(cnode) || JudgeOpSupportNNACLInfer(cnode);
}

STATUS NodeInferShape::InferShapeByNNACL(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  auto anf_prim = GetValueNode<std::shared_ptr<Primitive>>(cnode->input(0));
  if (anf_prim == nullptr) {
    MS_LOG(DEBUG) << cnode->fullname_with_scope() << "'s cnode primitive is nullptr";
    return lite::RET_ERROR;
  }
  (void)anf_prim->AddAttr(kInferDone, MakeValue<bool>(false));
  std::vector<TensorPtr> inputs_ptr;
  if (LiteTensorExtractor::GetCNodeInputTensors(cnode, &inputs_ptr, fmk_type_, train_flag_, false) != lite::RET_OK) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " get inputs failed.";
    return lite::RET_ERROR;
  }
  std::vector<TensorPtr> outputs_ptr;
  if (LiteTensorExtractor::GetCNodeOutputTensors(cnode, &outputs_ptr, train_flag_) != lite::RET_OK) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " get outputs failed.";
    return lite::RET_ERROR;
  }
  auto prim_t = lite::GetPrimitiveT(cnode->input(0));
  if (prim_t == nullptr) {
    MS_LOG(DEBUG) << cnode->fullname_with_scope() << " get lite prim_t is nullptr";
    return lite::RET_ERROR;
  }
  flatbuffers::FlatBufferBuilder fbb(INITIAL_SIZE);
  auto prim = lite::ConvertToPrimitive(prim_t.get(), &fbb);
  if (prim == nullptr) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " get primitive failed.";
    fbb.Clear();
    return lite::RET_ERROR;
  }
  std::vector<lite::Tensor *> inputs;
  (void)std::transform(inputs_ptr.begin(), inputs_ptr.end(), std::back_inserter(inputs),
                       [](const TensorPtr &input) { return input.get(); });
  std::vector<lite::Tensor *> outputs;
  (void)std::transform(outputs_ptr.begin(), outputs_ptr.end(), std::back_inserter(outputs),
                       [](const TensorPtr &output) { return output.get(); });
  auto ret = KernelInferShape(inputs, outputs, prim, {}, lite::SCHEMA_CUR);
  if (ret == lite::RET_NOT_SUPPORT) {
    auto parameter_gen = lite::PopulateRegistry::GetInstance()->GetParameterCreator(
      static_cast<int>(prim->value_type()), lite::SCHEMA_CUR);
    if (parameter_gen == nullptr) {
      MS_LOG(ERROR) << "PopulateParameter return nullptr, type: " << schema::EnumNamePrimitiveType(prim->value_type());
      fbb.Clear();
      return lite::RET_ERROR;
    }
    auto parameter = parameter_gen(prim);
    if (parameter == nullptr) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << " generate nullptr lite op parameter.";
      fbb.Clear();
      return lite::RET_ERROR;
    }
    RectifyFormat(inputs, fmk_type_);
    ret = KernelInferShape(inputs, outputs, parameter);
    if (parameter->destroy_func_ != nullptr) {
      parameter->destroy_func_(parameter);
    }
    free(parameter);
    parameter = nullptr;
  }
  fbb.Clear();
  if (ret == lite::RET_OK) {
    (void)anf_prim->AddAttr(kInferDone, MakeValue<bool>(true));
  }
  if (ret == lite::RET_OK || ret == lite::RET_INFER_INVALID) {
    auto set_status = SetCNodeAbstract(cnode, outputs, ret);
    (void)anf_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(static_cast<int64_t>(inputs[0]->format())));
    if (set_status != lite::RET_OK) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << " set CNode abstract failed.";
      return set_status;
    }
  } else {
    MS_LOG(WARNING) << "InferShapeByNNACL for op: " << cnode->fullname_with_scope() << " failed.";
  }
  std::vector<int64_t> outputs_format;
  (void)std::transform(outputs.begin(), outputs.end(), std::back_inserter(outputs_format),
                       [](const lite::Tensor *output) { return output->format(); });
  (void)anf_prim->AddAttr(kOutputsFormat, MakeValue(outputs_format));
  return ret;
}

STATUS NodeInferShape::InferShape(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  STATUS status;
  if (JudgeOpSupportOpsInfer(cnode)) {
    status = InferShapeByOps(cnode, true);
  } else if (JudgeOpSupportNNACLInfer(cnode)) {
    status = InferShapeByNNACL(cnode);
  } else {
    MS_LOG(ERROR) << "Unsupported node: " << cnode->fullname_with_scope() << " for infershape.";
    return RET_ERROR;
  }
  return status;
}

STATUS NodeInferShape::OpsInferShape(const PrimitivePtr &anf_prim, const AbstractBasePtrList &abs_list,
                                     AbstractBasePtr *result, bool invalid) {
  auto found = abstract::GetPrimitiveInferImpl(anf_prim);
  if (!found.has_value()) {
    MS_LOG(ERROR) << "Can't find the infer impl for ops: " << anf_prim->name();
    return lite::RET_ERROR;
  }
  auto infer = found.value();
  if (!infer.IsImplInferShapeAndType()) {
    MS_LOG(ERROR) << "For ops: " << anf_prim->name() << ", the InferShapeAndType is not implemented.";
    return lite::RET_ERROR;
  }

  *result = found->InferShapeAndType(nullptr, anf_prim, abs_list);
  if (*result == nullptr) {
    MS_LOG(ERROR) << "For ops: " << anf_prim->name() << ", call InferShapeAndType failed.";
    return lite::RET_ERROR;
  }
  return RET_OK;
}

STATUS NodeInferShape::ConvertAbstractListToNCOrNH(const CNodePtr &cnode, AbstractBasePtrList abs_list,
                                                   FormatTransNodeType perm, bool *changed) {
  MS_ERROR_IF_NULL_W_RET_VAL(cnode, lite::RET_ERROR);
  MS_ERROR_IF_NULL_W_RET_VAL(changed, lite::RET_ERROR);
  std::vector<size_t> insert_index;
  *changed = false;
  if (GetFormatSensitiveOpInsertIndex(cnode, &insert_index) != RET_OK) {
    MS_LOG(ERROR) << "GetFormatSensitiveOpInsertIndex failed.";
    return RET_ERROR;
  }
  if (insert_index.size() == 0) {
    MS_LOG(DEBUG) << "op don't meet condition.";
    return lite::RET_OK;
  }
  *changed = true;
  for (auto &index : insert_index) {
    if ((index < 1) || index > abs_list.size()) {
      MS_LOG(ERROR) << "index is invalid.";
      return lite::RET_ERROR;
    }
    if (ConvertAbstractFormatShape(abs_list[index - 1], perm) != lite::RET_OK) {
      MS_LOG(ERROR) << "ConvertAbstract failed.";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

STATUS NodeInferShape::SetCNodeAbstractByConvert(const CNodePtr &cnode, const AbstractBasePtr &result, STATUS infer_ret,
                                                 bool change, FormatTransNodeType perm, const Format &format) {
  AbstractBasePtr abs = result;
  if (abs == nullptr) {
    abs = cnode->abstract();
    if (abs == nullptr) {
      MS_LOG(ERROR) << "abstract is nullptr.";
      return lite::RET_ERROR;
    }
  }
  size_t output_size;
  if (utils::isa<abstract::AbstractTuple>(abs)) {
    auto abs_tuple = abs->cast_ptr<abstract::AbstractTuple>();
    AbstractBasePtrList abstract_list;
    output_size = abs_tuple->size();
    if (output_size == 0 || (*abs_tuple)[0]->isa<abstract::AbstractScalar>()) {
      ShapeVector ori_shape = {static_cast<int64_t>(output_size)};
      BaseShapePtr new_shape = std::make_shared<abstract::Shape>(ori_shape);
      TypeId type_id = static_cast<TypeId>(kNumberTypeFloat32);

      if (output_size != 0) {
        auto scalar_type_ptr = (*abs_tuple)[0]->cast<abstract::AbstractScalarPtr>()->GetTypeTrack();
        MS_CHECK_TRUE_MSG(scalar_type_ptr != nullptr, RET_ERROR, "type_ptr is nullptr");
        type_id = scalar_type_ptr->type_id();
      }
      auto type_ptr = TypeIdToType(type_id);
      auto out_abs = std::make_shared<abstract::AbstractTensor>(type_ptr, new_shape);
      AbstractBasePtr new_result;
      if (ConvertAbstract(out_abs, &new_result, change, perm) != RET_OK) {
        MS_LOG(ERROR) << "ConvertAbstract failed.";
        return lite::RET_ERROR;
      }
      cnode->set_abstract(new_result);
      output_size = 1;
    } else {
      for (size_t it = 0; it < output_size; ++it) {
        auto abs_temp = (*abs_tuple)[it];
        AbstractBasePtr new_result;
        if (ConvertAbstract(abs_temp, &new_result, change, perm) != RET_OK) {
          MS_LOG(ERROR) << "ConvertAbstract failed.";
          return lite::RET_ERROR;
        }
        abstract_list.emplace_back(new_result);
      }
      auto new_abstract_list = std::make_shared<abstract::AbstractTuple>(abstract_list);
      CHECK_NULL_RETURN(new_abstract_list);
      cnode->set_abstract(new_abstract_list);
    }
  } else if (utils::isa<abstract::AbstractTensor>(abs)) {
    AbstractBasePtr new_result;
    if (ConvertAbstract(abs, &new_result, change, perm) != RET_OK) {
      MS_LOG(ERROR) << "ConvertAbstract failed.";
      return lite::RET_ERROR;
    }
    cnode->set_abstract(new_result);
    output_size = 1;
  } else if (utils::isa<abstract::AbstractScalar>(abs)) {
    ShapeVector ori_shape = {1};
    BaseShapePtr new_shape = std::make_shared<abstract::Shape>(ori_shape);
    auto scalar_type_ptr = abs->cast<abstract::AbstractScalarPtr>()->GetTypeTrack();
    MS_CHECK_TRUE_MSG(scalar_type_ptr != nullptr, RET_ERROR, "type_ptr is nullptr");
    auto out_abs = std::make_shared<abstract::AbstractTensor>(scalar_type_ptr, new_shape);
    AbstractBasePtr new_result;
    if (ConvertAbstract(out_abs, &new_result, change, perm) != RET_OK) {
      MS_LOG(ERROR) << "ConvertAbstract failed.";
      return lite::RET_ERROR;
    }
    cnode->set_abstract(new_result);
    output_size = 1;
  } else {
    MS_LOG(ERROR) << "Unknown abstract type :" << abs;
    return lite::RET_ERROR;
  }

  std::vector<int64_t> outputs_format(output_size, format);
  auto anf_prim = GetValueNode<std::shared_ptr<Primitive>>(cnode->input(0));
  if (anf_prim == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return lite::RET_ERROR;
  }
  (void)anf_prim->AddAttr(kOutputsFormat, MakeValue(outputs_format));
  return lite::RET_OK;
}

STATUS NodeInferShape::InferShapeByOps(const CNodePtr &cnode, bool invalid) {
  CHECK_NULL_RETURN(cnode);
  STATUS infer_ret = RET_OK;
  AbstractBasePtrList abs_list;
  if (LiteTensorExtractor::GetCNodeInputAbstractLists(cnode, &abs_list) != RET_OK) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " GetCNodeInputAbstractLists failed.";
    return lite::RET_ERROR;
  }

  auto anf_prim = GetValueNode<std::shared_ptr<Primitive>>(cnode->input(0));
  if (anf_prim == nullptr) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " primitive is nullptr";
    return lite::RET_ERROR;
  }
  (void)anf_prim->AddAttr(kInferDone, MakeValue<bool>(false));

  if (LiteTensorExtractor::GetCNodeConstInputToAbstract(cnode, abs_list, fmk_type_, train_flag_) != RET_OK) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " GetCNodeConstInputToAbstract failed.";
    return RET_ERROR;
  }
  Format ori_format = Format::NHWC;
  if (anf_prim->GetAttr(mindspore::ops::kFormat) != nullptr) {
    ori_format = static_cast<Format>(GetValue<int64_t>(anf_prim->GetAttr(mindspore::ops::kFormat)));
  }
  bool changed = false;
  if (ori_format == Format::NHWC) {
    if (ConvertAbstractListToNCOrNH(cnode, abs_list, kNHWC2NCHW, &changed) != RET_OK) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << " ConvertAbstractToNCOrNH failed.";
      return RET_ERROR;
    }
  }
  (void)anf_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(static_cast<int>(Format::NCHW)));
  AbstractBasePtr result;
  try {
    infer_ret = OpsInferShape(anf_prim, abs_list, &result, invalid);
  } catch (const std::exception &e) {
    MS_LOG(WARNING) << "InferShapeByOps for op: " << cnode->fullname_with_scope() << " failed. " << e.what();
    throw;
  }
  (void)anf_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(static_cast<int64_t>(ori_format)));
  if (infer_ret == lite::RET_OK) {
    (void)anf_prim->AddAttr(kInferDone, MakeValue<bool>(true));
    auto input_format = NHWC;
    (void)opt::DetermineCertainVarInputFormat(cnode, 1, &input_format);
    auto set_status = SetCNodeAbstractByConvert(cnode, result, infer_ret, changed, kNCHW2NHWC, input_format);
    if (set_status != lite::RET_OK) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << " SetCNodeAbstractByConvert failed.";
      return set_status;
    }
  }

  return infer_ret;
}

std::vector<int> NodeInferShape::GetInputShape(const CNodePtr &cnode, size_t index) {
  MS_ASSERT(cnode != nullptr);
  if (index >= cnode->size()) {
    return {};
  }
  lite::DataInfo data_info;
  int status = lite::RET_OK;
  CNodePtr base_node = cnode;
  size_t position = index;
  if (CheckPrimitiveType(cnode->input(index), prim::kPrimMakeTuple) ||
      CheckPrimitiveType(cnode->input(index), prim::kPrimMakeTupleV2)) {
    base_node = cnode->input(index)->cast<CNodePtr>();
    position = 1;
  }
  if (utils::isa<CNode>(base_node->input(position))) {
    status = lite::FetchDataFromCNode(base_node, position, &data_info);
  } else if (utils::isa<Parameter>(base_node->input(position))) {
    status = lite::FetchDataFromParameterNode(base_node, position, fmk_type_, &data_info, false);
  } else if (utils::isa<ValueNodePtr>(base_node->input(position))) {
    status = lite::FetchDataFromValueNode(base_node, position, fmk_type_, train_flag_, &data_info, false);
  } else {
    MS_LOG(ERROR) << "input node is invalid.";
    return {};
  }
  if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
    MS_LOG(ERROR) << "fetch data failed.";
    return {};
  }
  return data_info.shape_;
}

std::vector<int> NodeInferShape::GetIntVecInput(const CNodePtr &cnode, size_t index) {
  MS_ASSERT(cnode != nullptr);
  if (index >= cnode->size()) {
    return {};
  }
  auto origin_inputs = cnode->inputs();
  std::vector<AnfNodePtr> specify_inputs = {origin_inputs[0], origin_inputs[index]};
  cnode->set_inputs(specify_inputs);
  std::vector<TensorPtr> specify_tensors;
  if (LiteTensorExtractor::GetCNodeInputTensors(cnode, &specify_tensors, fmk_type_, train_flag_, false) !=
        lite::RET_OK ||
      specify_tensors.empty()) {
    cnode->set_inputs(origin_inputs);
    return {};
  }
  cnode->set_inputs(origin_inputs);
  std::vector<int> tensor_data;
  if (specify_tensors.front()->data_type() != kNumberTypeInt32 &&
      specify_tensors.front()->data_type() != kNumberTypeInt) {
    return {};
  }
  if (specify_tensors.front()->shape().size() != 1) {
    return {};
  }
  MS_CHECK_GE(specify_tensors.front()->shape()[0], 0, {});
  tensor_data.resize(static_cast<size_t>(specify_tensors.front()->shape()[0]));
  if (memcpy_s(tensor_data.data(), tensor_data.size() * sizeof(int), specify_tensors.front()->data(),
               specify_tensors.front()->Size()) != EOK) {
    return {};
  }
  return tensor_data;
}

STATUS NodeInferShape::SetCNodeAbstract(const std::shared_ptr<CNode> &cnode, const std::vector<lite::Tensor *> &outputs,
                                        int status) {
  MS_ASSERT(cnode != nullptr);
  if (outputs.size() == 0) {
    MS_LOG(ERROR) << "empty output_tensors";
    return RET_ERROR;
  }
  auto origin_abstract = cnode->abstract();
  MS_ASSERT(origin_abstract != nullptr);
  if (outputs.size() == 1 && !utils::isa<abstract::AbstractTuple>(origin_abstract)) {
    auto tensor = outputs.front();
    auto new_abstract = ConvertLiteTensorToAbstract(tensor);
    if (new_abstract == nullptr) {
      MS_LOG(ERROR) << "new abstract failed.";
      return RET_ERROR;
    }
    if (status == lite::RET_INFER_INVALID) {
      if (tensor->data_type() == kObjectTypeTensorType) {
        ShapeVector shape = {0};
        auto abstract_shape = std::make_shared<abstract::Shape>(shape);
        CHECK_NULL_RETURN(abstract_shape);
        new_abstract->set_shape(abstract_shape);
      }
    }
    cnode->set_abstract(new_abstract);
  } else {
    AbstractBasePtrList abstract_list;
    for (size_t i = 0; i < outputs.size(); i++) {
      auto tensor = outputs.at(i);
      auto new_abstract = ConvertLiteTensorToAbstract(tensor);
      if (new_abstract == nullptr) {
        MS_LOG(ERROR) << "new abstract failed.";
        return RET_ERROR;
      }
      if (status == lite::RET_INFER_INVALID) {
        if (tensor->data_type() == kObjectTypeTensorType) {
          ShapeVector shape = {0};
          auto abstract_shape = std::make_shared<abstract::Shape>(shape);
          CHECK_NULL_RETURN(abstract_shape);
          new_abstract->set_shape(abstract_shape);
        }
      }
      abstract_list.emplace_back(new_abstract);
    }
    auto new_abstract_list = std::make_shared<abstract::AbstractTuple>(abstract_list);
    CHECK_NULL_RETURN(new_abstract_list);
    cnode->set_abstract(new_abstract_list);
  }
  return RET_OK;
}

abstract::AbstractBasePtr NodeInferShape::ConvertLiteTensorToAbstract(lite::Tensor *tensor) {
  MS_ASSERT(tensor != nullptr);
  if (tensor->data_type() == kObjectTypeTensorType) {
    return ConvertTensorListToAbstract(tensor);
  }
  auto tensor_info = NewTensorInfo(tensor);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "new tensor::Tensor failed";
    return nullptr;
  }
  return tensor_info->ToAbstract();
}

// stract save tensorlist's type and shape. tensor_info save tensorlist's data and data type.
// both of them is different in term of shape and type.
abstract::AbstractBasePtr NodeInferShape::ConvertTensorListToAbstract(lite::Tensor *tensor) {
  MS_ASSERT(tensor != nullptr);
  auto tensor_list = reinterpret_cast<lite::TensorList *>(tensor);
  if (tensor_list == nullptr) {
    MS_LOG(ERROR) << "cast tensor_list failed";
    return nullptr;
  }
  std::vector<int> shape(tensor->shape());
  std::vector<int64_t> shape_vector(shape.begin(), shape.end());
  auto tensor_list_abstract =
    std::make_shared<abstract::AbstractTensor>(TypeIdToType(tensor_list->data_type()), shape_vector);
  if (tensor_list_abstract == nullptr) {
    MS_LOG(ERROR) << "new AbstractTensor failed";
    return nullptr;
  }
  auto elememt_shape = tensor_list->element_shape();
  std::vector<int> data_info;
  data_info.push_back(tensor_list->tensors_data_type());
  data_info.push_back(elememt_shape.size());
  std::copy(elememt_shape.begin(), elememt_shape.end(), std::back_inserter(data_info));
  data_info.push_back(tensor_list->tensors().size());
  for (size_t i = 0; i < tensor_list->tensors().size(); ++i) {
    auto tensor_mem = tensor_list->tensors()[i];
    auto tensor_mem_shape = tensor_mem->shape();
    data_info.push_back(tensor_mem_shape.size());
    std::copy(tensor_mem_shape.begin(), tensor_mem_shape.end(), std::back_inserter(data_info));
  }
  std::vector<int64_t> data_shape;
  data_shape.push_back(data_info.size());
  auto tensor_info = std::make_shared<tensor::Tensor>(kNumberTypeInt32, data_shape, data_info.data(), kNumberTypeInt32);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "new tensor::Tensor failed";
    return nullptr;
  }
  tensor_list_abstract->set_value(tensor_info);
  return tensor_list_abstract;
}
}  // namespace opt
}  // namespace mindspore
