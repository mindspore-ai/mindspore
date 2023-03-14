/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "abstract/ops/primitive_infer_map.h"
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <cstdint>
#include <iterator>

#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "ops/dropout.h"

namespace mindspore {
namespace abstract {
using ops::InferImplDropout;

PrimShapeDependMap &GetInferDependsMap() {
  // Registration directly by the host_depends map will be deprecated and
  // should be registered by the GetValueDependArgIndices
  using ShapeSet = std::set<int64_t>;
  static const auto &kMirrorPad = prim::kPrimMirrorPad->name();
  static const auto &kAdaptiveAvgPool3D = prim::kPrimAdaptiveAvgPool3D->name();
  static const auto &kAdaptiveAvgPool3DGrad = prim::kPrimAdaptiveAvgPool3DGrad->name();
  static const auto &kAvgPoolGradV1 = prim::kPrimAvgPoolGradV1->name();
  static const auto &kOneHot = prim::kPrimOneHot->name();
  static const auto &kDropoutGenMask = prim::kPrimDropoutGenMask->name();
  static const auto &kStridedSlice = prim::kPrimStridedSlice->name();
  static const auto &kStridedSliceGrad = prim::kPrimStridedSliceGrad->name();
  static const auto &kStridedSliceV2 = prim::kPrimStridedSliceV2->name();
  static const auto &kStridedSliceV2Grad = prim::kPrimStridedSliceV2Grad->name();
  static const auto &kSparseToDenseV2 = prim::kPrimSparseToDenseV2->name();
  static const auto &kResizeBicubic = prim::kPrimResizeBicubic->name();
  static const auto &kRandomCategorical = prim::kPrimRandomCategorical->name();
  static const auto &kMatrixDiagV3 = prim::kPrimMatrixDiagV3->name();
  static const auto &kMatrixDiagPartV3 = prim::kPrimMatrixDiagPartV3->name();
  static const auto &kMatrixSetDiagV3 = prim::kPrimMatrixSetDiagV3->name();
  static const auto &kRaggedRange = prim::kPrimRaggedRange->name();
  static const auto &kDynamicBroadcastTo = prim::kPrimDynamicBroadcastTo->name();
  static const auto &kUnsortedSegmentSum = prim::kPrimUnsortedSegmentSum->name();
  static const auto &kUnsortedSegmentProd = prim::kPrimUnsortedSegmentProd->name();
  static const auto &kUnsortedSegmentMin = prim::kPrimUnsortedSegmentMin->name();
  static const auto &kUnsortedSegmentMax = prim::kPrimUnsortedSegmentMax->name();
  static const auto &kGather = prim::kPrimGather->name();
  static const auto &kGatherV2 = prim::kPrimGatherV2->name();
  static const auto &kGatherD = prim::kPrimGatherD->name();
  static const auto &kRangeV2 = prim::kPrimRangeV2->name();
  static const auto &kConv2DBackpropFilter = prim::kPrimConv2DBackpropFilter->name();
  static const auto &kConv2DBackpropInput = prim::kPrimConv2DBackpropInput->name();
  static const auto &kCol2Im = prim::kPrimCol2Im->name();
  static const auto &kTile = prim::kPrimTile->name();
  static const auto &kTopK = prim::kPrimTopK->name();
  static const auto &kNonDeterministicInts = prim::kPrimNonDeterministicInts->name();
  static const auto &kSliceGrad = prim::kPrimSliceGrad->name();
  static const auto &kReshape = prim::kPrimReshape->name();
  static const auto &kResizeNearestNeighborV2 = prim::kPrimResizeNearestNeighborV2->name();
  static const auto &kResizeNearestNeighborV2Grad = prim::kPrimResizeNearestNeighborV2Grad->name();
  static const auto &kScatterNd = prim::kPrimScatterNd->name();
  static const auto &kTruncatedNormal = prim::kPrimTruncatedNormal->name();
  static const auto &kRandomGamma = prim::kPrimRandomGamma->name();
  static const auto &kAffineGrid = prim::kPrimAffineGrid->name();
  static const auto &kFillV2 = prim::kPrimFillV2->name();
  static const auto &kFill = prim::kPrimFill->name();
  static const auto &kFractionalAvgPoolGrad = prim::kPrimFractionalAvgPoolGrad->name();
  static const auto &kTranspose = prim::kPrimTranspose->name();
  static const auto &kResizeLinear1D = prim::kPrimResizeLinear1D->name();
  static const auto &kSegmentMax = prim::kPrimSegmentMax->name();
  static const auto &kSegmentMin = prim::kPrimSegmentMin->name();
  static const auto &kSegmentSum = prim::kPrimSegmentSum->name();
  static const auto &kBlackmanWindow = prim::kPrimBlackmanWindow->name();
  static const auto &kHammingWindow = prim::kPrimHammingWindow->name();
  static const auto &kResizeArea = prim::kPrimResizeArea->name();
  static const auto &kExpand = prim::kPrimExpand->name();
  static const auto &kSspaddmm = prim::kPrimSspaddmm->name();
  static const auto &kBartlettWindow = prim::kPrimBartlettWindow->name();
  static const auto &kExtractGlimpse = prim::kPrimExtractGlimpse->name();
  static const auto &kTensorCopySlices = prim::kPrimTensorCopySlices->name();
  static const auto &kResizeNearestNeighborGrad = prim::kPrimResizeNearestNeighborGrad->name();
  static const auto &kSegmentMean = prim::kPrimSegmentMean->name();
  static const auto &kSegmentProd = prim::kPrimSegmentProd->name();
  static const auto &kStandardNormal = prim::kPrimStandardNormal->name();
  static const auto &kStandardLaplace = prim::kPrimStandardLaplace->name();
  static const auto &kCropAndResizeGradImage = prim::kPrimCropAndResizeGradImage->name();
  static const auto &kTraceGrad = prim::kPrimTraceGrad->name();
  static const auto &kSetSize = prim::kPrimSetSize->name();
  static const auto &kDynamicStitch = prim::kPrimDynamicStitch->name();
  static const auto &kSparseTensorDenseMatmul = prim::kPrimSparseTensorDenseMatmul->name();
  static const auto &kSparseSegmentMeanWithNumSegments = prim::kPrimSparseSegmentMeanWithNumSegments->name();
  static const auto &kSparseSegmentSqrtN = prim::kPrimSparseSegmentSqrtN->name();
  static const auto &kSparseSegmentSqrtnWithNumSegments = prim::kPrimSparseSegmentSqrtNWithNumSegments->name();
  static const auto &kSparseMatrixTranspose = prim::kPrimSparseMatrixTranspose->name();
  static const auto &kSparseToDense = prim::kPrimSparseToDense->name();
  static const auto &kParameterizedTruncatedNormal = prim::kPrimParameterizedTruncatedNormal->name();
  // Common host depends.
  static PrimShapeDependMap depends{{prim::kPrimArgMax->name(), ShapeSet{1}},
                                    {prim::kPrimArgmin->name(), ShapeSet{1}},
                                    {kExtractGlimpse, ShapeSet{1}},
                                    {kMirrorPad, ShapeSet{1}},
                                    {kSegmentMax, ShapeSet{1}},
                                    {kSegmentMin, ShapeSet{1}},
                                    {kSegmentSum, ShapeSet{1}},
                                    {kSegmentMean, ShapeSet{1}},
                                    {kSegmentProd, ShapeSet{1}},
                                    {kUnsortedSegmentSum, ShapeSet{2}},
                                    {kFractionalAvgPoolGrad, ShapeSet{0}},
                                    {kUnsortedSegmentMin, ShapeSet{2}},
                                    {kUnsortedSegmentMax, ShapeSet{2}},
                                    {kUnsortedSegmentProd, ShapeSet{2}},
                                    {kMatrixDiagV3, ShapeSet{1, 2, 3, 4}},
                                    {kMatrixDiagPartV3, ShapeSet{1, 2}},
                                    {kMatrixSetDiagV3, ShapeSet{2}},
                                    {kGather, ShapeSet{2}},
                                    {kGatherV2, ShapeSet{2}},
                                    {kGatherD, ShapeSet{1}},
                                    {kRangeV2, ShapeSet{0, 1, 2}},
                                    {kResizeBicubic, ShapeSet{1}},
                                    {kConv2DBackpropFilter, ShapeSet{2}},
                                    {kConv2DBackpropInput, ShapeSet{2}},
                                    {kCol2Im, ShapeSet{1}},
                                    {kAvgPoolGradV1, ShapeSet{0}},
                                    {kOneHot, ShapeSet{1, 3}},
                                    {kDropoutGenMask, ShapeSet{0}},
                                    {prim::kStatelessDropOutGenMask, ShapeSet{0}},
                                    {kStridedSlice, ShapeSet{1, 2, 3}},
                                    {kStridedSliceGrad, ShapeSet{1, 2, 3, 4}},
                                    {kStridedSliceV2, ShapeSet{1, 2, 3}},
                                    {kStridedSliceV2Grad, ShapeSet{0}},
                                    {kTensorCopySlices, ShapeSet{2, 3, 4}},
                                    {kTile, ShapeSet{1}},
                                    {kTopK, ShapeSet{1}},
                                    {kReshape, ShapeSet{1}},
                                    {kResizeNearestNeighborV2, ShapeSet{1}},
                                    {kResizeNearestNeighborV2Grad, ShapeSet{1}},
                                    {kScatterNd, ShapeSet{2}},
                                    {kSparseToDenseV2, ShapeSet{1}},
                                    {prim::kPrimSparseTensorDenseMatmul->name(), ShapeSet{2}},
                                    {kSliceGrad, ShapeSet{2, 3}},
                                    {kFillV2, ShapeSet{0}},
                                    {kFill, ShapeSet{0, 2}},
                                    {kRandomCategorical, ShapeSet{1}},
                                    {kRandomGamma, ShapeSet{0, 1}},
                                    {kDynamicBroadcastTo, ShapeSet{1}},
                                    {kNonDeterministicInts, ShapeSet{0}},
                                    {prim::kPrimArgminV2->name(), ShapeSet{1}},
                                    {prim::kPrimArgMin->name(), ShapeSet{1}},
                                    {kAffineGrid, ShapeSet{1}},
                                    {prim::kPrimInplaceUpdateV2->name(), ShapeSet{1}},
                                    {kTruncatedNormal, ShapeSet{0}},
                                    {kRaggedRange, ShapeSet{0, 1, 2}},
                                    {kTranspose, ShapeSet{1}},
                                    {kAdaptiveAvgPool3D, ShapeSet{1}},
                                    {kAdaptiveAvgPool3DGrad, ShapeSet{1}},
                                    {kResizeLinear1D, ShapeSet{1}},
                                    {kBlackmanWindow, ShapeSet{0}},
                                    {kHammingWindow, ShapeSet{0}},
                                    {kResizeArea, ShapeSet{1}},
                                    {kExpand, ShapeSet{1}},
                                    {kSspaddmm, ShapeSet{0, 2, 3, 5, 7}},
                                    {kBartlettWindow, ShapeSet{0}},
                                    {kResizeNearestNeighborGrad, ShapeSet{1}},
                                    {kTraceGrad, ShapeSet{1}},
                                    {kStandardNormal, ShapeSet{0}},
                                    {kStandardLaplace, ShapeSet{0}},
                                    {kCropAndResizeGradImage, ShapeSet{3}},
                                    {prim::kPrimCumSum->name(), ShapeSet{1}},
                                    {prim::kPrimCumsum->name(), ShapeSet{1}},
                                    {kSetSize, ShapeSet{2}},
                                    {kDynamicStitch, ShapeSet{0}},
                                    {kSparseTensorDenseMatmul, ShapeSet{2}},
                                    {kSparseSegmentMeanWithNumSegments, ShapeSet{3}},
                                    {kSparseSegmentSqrtN, ShapeSet{2}},
                                    {kSparseSegmentSqrtnWithNumSegments, ShapeSet{3}},
                                    {kSparseMatrixTranspose, ShapeSet{0}},
                                    {kParameterizedTruncatedNormal, ShapeSet{0}},
                                    {prim::kPrimMirrorPadGrad->name(), ShapeSet{1}},
                                    {kSparseToDense, ShapeSet{2}}};
  return depends;
}

int64_t GetDependValueSize(const ValuePtr &value) {
  if (value->isa<Int64Imm>()) {
    return GetValue<int64_t>(value);
  }
  if (!value->isa<ValueTuple>()) {
    MS_LOG(EXCEPTION) << "the element of attr[dyn_input_size] should be all int64 of ValueTuple but got"
                      << value->ToString() << ", type :" << value->type_name();
  }
  int64_t size = 0;
  auto value_tuple = value->cast_ptr<ValueTuple>();
  MS_EXCEPTION_IF_NULL(value_tuple);
  for (size_t i = 0; i < value_tuple->size(); ++i) {
    size += GetDependValueSize((*value_tuple)[i]);
  }
  return size;
}

std::set<int64_t> RectifyDependListFromDynamicInputAttr(const CNodePtr &cnode, const PrimitivePtr &primitive,
                                                        const std::set<int64_t> &ori_depend_list) {
  std::set<int64_t> rec_depend_list = {};
  constexpr auto all_tensor_inputs = -1;
  if (ori_depend_list.size() == 1 && *(ori_depend_list.cbegin()) == all_tensor_inputs) {
    for (size_t i = 1; i < cnode->size(); ++i) {
      const auto &input = cnode->inputs()[i];
      const auto &input_abstract = input->abstract();
      if (input_abstract != nullptr) {
        auto is_tensor = input_abstract->isa<abstract::AbstractTensor>();
        bool is_integer = false;
        if (input_abstract->isa<abstract::AbstractScalar>()) {
          // Now, only scalar with int32/int64 will be used as the output of operator, so only add them to list.
          auto scalar_id = input_abstract->BuildType()->type_id();
          is_integer = (scalar_id == kNumberTypeInt32 || scalar_id == kNumberTypeInt64);
        }
        if (is_tensor || is_integer) {
          (void)rec_depend_list.emplace(SizeToLong(i - 1));
        }
      }
    }
    return rec_depend_list;
  }

  const auto &inputs = cnode->inputs();
  auto attr = primitive->GetAttr(kAttrDynInputSizes);
  if (attr == nullptr) {
    return ori_depend_list;
  }
  MS_EXCEPTION_IF_NULL(attr);
  auto dyn_input_list = attr->cast_ptr<ValueTuple>();
  MS_EXCEPTION_IF_NULL(dyn_input_list);
  for (const auto i : ori_depend_list) {
    if (LongToSize(i) > dyn_input_list->size()) {
      MS_LOG(EXCEPTION) << "The index is out of range.";
    }
    int64_t start_index = 0;
    for (int64_t index = 0; index < i; ++index) {
      auto place_holder_size = GetValue<int64_t>((*dyn_input_list)[index]);
      if (place_holder_size < 0) {
        start_index += 1;
      }
      start_index += place_holder_size;
    }
    auto dyn_size = GetValue<int64_t>((*dyn_input_list)[i]);
    MS_LOG(DEBUG) << "The input " << i << " dynamic input size is " << dyn_size;
    while (dyn_size >= 0) {
      auto depend_index = start_index + dyn_size;
      // skip primitive input
      const auto &input = inputs.at(depend_index + 1);
      MS_EXCEPTION_IF_NULL(input);
      const auto &input_abs = input->abstract();
      if (input_abs != nullptr && input_abs->isa<abstract::AbstractTensor>()) {
        rec_depend_list.emplace(depend_index);
        MS_LOG(DEBUG) << "Rectify dynamic input " << start_index + dyn_size;
      }
      --dyn_size;
    }
  }
  return rec_depend_list;
}

std::set<int64_t> GetValueDependArgIndices(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().empty()) {
    MS_LOG(EXCEPTION) << "Invalid inputs";
  }
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::set<int64_t> ori = {};
  auto device = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  // Special dynamic shape depends for Ascend.
  if (device == kAscendDevice && prim_name == prim::kPrimTranspose->name()) {
    ori.emplace(1);
  }

  auto iter = GetInferDependsMap().find(prim_name);
  if (iter != GetInferDependsMap().end()) {
    ori = iter->second;
  }

  auto op_infer_opt = GetPrimitiveInferImpl(std::make_shared<Primitive>(prim_name));
  if (op_infer_opt.has_value()) {
    auto op_infer = op_infer_opt.value().Get();
    if (op_infer != nullptr && ori.empty()) {
      ori = op_infer->GetValueDependArgIndices();
    }
  }
  if (ori.empty()) {
    return ori;
  }
  size_t input_num = cnode->inputs().size() - 1;
  std::set<int64_t> res = {};

  (void)std::copy_if(ori.begin(), ori.end(), std::inserter(res, res.begin()),
                     [&](int64_t idx) { return idx < SizeToLong(input_num); });
  return RectifyDependListFromDynamicInputAttr(cnode, primitive, res);
}

RegisterInferDependsHelper::RegisterInferDependsHelper(const std::string &name, const std::set<int64_t> &depends) {
  auto &depends_map = GetInferDependsMap();
  depends_map[name] = depends;
}

PrimitiveEvalImplMap *GetPrimitiveInferMapPtr() {
  // using R = PrimitiveEvalImplMap::mapped_type;
  static PrimitiveEvalImplMap prim_eval_implement_map{
    // core/ops infer
    // Do not add anything in this initializer anymore since it will be removed soon, core/ops prim should register its
    // infer in its cc file.
  };
  return &prim_eval_implement_map;
}
const PrimitiveEvalImplMap &GetPrimitiveInferMap() { return *GetPrimitiveInferMapPtr(); }

std::optional<StandardPrimitiveImplReg> GetPrimitiveInferImpl(const PrimitivePtr &primitive) {
  auto iter = GetPrimitiveInferMap().find(primitive);
  if (iter != GetPrimitiveInferMap().end()) {
    return iter->second;
  }

  iter = GetDeprecatedPrimitiveInferMap().find(primitive);
  if (iter != GetDeprecatedPrimitiveInferMap().end()) {
    return iter->second;
  }
  return std::optional<StandardPrimitiveImplReg>();
}

class OpInferCommon : public OpInferBase {
 public:
  OpInferCommon() = delete;
  OpInferCommon(const InferAbstractImpl &infer_impl, const InferValueImpl &infer_value_impl)
      : infer_impl_(infer_impl), infer_value_impl_(infer_value_impl) {}
  ~OpInferCommon() = default;

  BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override;

 private:
  InferAbstractImpl infer_impl_{nullptr};
  InferValueImpl infer_value_impl_{nullptr};
};

BaseShapePtr OpInferCommon::InferShape(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  if (!infer_impl_) {
    return nullptr;
  }

  auto inferred_res = infer_impl_(nullptr, primitive, input_args);
  if (inferred_res == nullptr) {
    return nullptr;
  }

  return inferred_res->BuildShape();
}

TypePtr OpInferCommon::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  if (!infer_impl_) {
    return nullptr;
  }

  auto inferred_res = infer_impl_(nullptr, primitive, input_args);
  if (inferred_res == nullptr) {
    return nullptr;
  }

  return inferred_res->BuildType();
}

ValuePtr OpInferCommon::InferValue(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  if (!infer_value_impl_) {
    return nullptr;
  }
  return infer_value_impl_(primitive, input_args);
}

AbstractBasePtr OpInferCommon::InferShapeAndType(const abstract::AnalysisEnginePtr &engine,
                                                 const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  if (!infer_impl_) {
    return nullptr;
  }

  return infer_impl_(engine, primitive, input_args);
}

StandardPrimitiveImplReg::StandardPrimitiveImplReg(const InferAbstractImpl &infer_abstract,
                                                   const InferValueImpl &infer_value, bool in_white_list) {
  op_infer_ = std::make_shared<OpInferCommon>(infer_abstract, infer_value);
  is_impl_infer_shape_and_type_ = infer_abstract != nullptr;
  is_impl_infer_value_ = infer_value != nullptr;
  in_white_list_ = in_white_list;
}

AbstractBasePtr StandardPrimitiveImplReg::InferShapeAndType(const abstract::AnalysisEnginePtr &engine,
                                                            const PrimitivePtr &primitive,
                                                            const std::vector<AbstractBasePtr> &input_args) const {
  if (op_infer_ == nullptr) {
    return nullptr;
  }

  return op_infer_->InferShapeAndType(engine, primitive, input_args);
}

BaseShapePtr StandardPrimitiveImplReg::InferShape(const PrimitivePtr &prim, const AbstractBasePtrList &args) const {
  if (op_infer_ == nullptr) {
    return nullptr;
  }

  return op_infer_->InferShape(prim, args);
}

TypePtr StandardPrimitiveImplReg::InferType(const PrimitivePtr &prim, const AbstractBasePtrList &args) const {
  if (op_infer_ == nullptr) {
    return nullptr;
  }

  return op_infer_->InferType(prim, args);
}

ValuePtr StandardPrimitiveImplReg::InferValue(const PrimitivePtr &prim, const AbstractBasePtrList &args) const {
  if (op_infer_ == nullptr) {
    return nullptr;
  }

  return op_infer_->InferValue(prim, args);
}
}  // namespace abstract
}  // namespace mindspore
