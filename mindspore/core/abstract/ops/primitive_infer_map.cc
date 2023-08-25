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

#include "mindspore/core/ops/sparse_ops.h"
#include "mindspore/core/ops/random_ops.h"
#include "mindspore/core/ops/conv_pool_ops.h"
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/image_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/shape_calc.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace abstract {
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

bool CheckScalarValid(const AbstractBasePtr &input_abstract) {
  // Now, only scalar with int/float/uint will be used as the output of operator, so only add them to list.
  if (input_abstract->isa<abstract::AbstractScalar>()) {
    auto scalar_id = NormalizeTypeId(input_abstract->BuildType()->type_id());
    return (scalar_id == kNumberTypeBool || scalar_id == kNumberTypeInt || scalar_id == kNumberTypeFloat ||
            scalar_id == kNumberTypeUInt);
  }
  return false;
}

bool CheckNeedAddToDependList(const AbstractBasePtr &input_abstract) {
  auto is_tensor = input_abstract->isa<abstract::AbstractTensor>();
  bool is_integer = false;
  bool is_tuple_scalar_or_tensor = false;
  is_integer = CheckScalarValid(input_abstract);
  if (input_abstract->isa<abstract::AbstractTuple>()) {
    auto tuple_abs = input_abstract->cast_ptr<abstract::AbstractTuple>();
    auto elements = tuple_abs->elements();
    is_tuple_scalar_or_tensor = std::all_of(elements.begin(), elements.end(), [](const AbstractBasePtr &element) {
      return (CheckScalarValid(element)) || element->isa<abstract::AbstractTensor>();
    });
  }
  return is_tensor || is_integer || is_tuple_scalar_or_tensor;
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
        auto need_add_to_depend_list = CheckNeedAddToDependList(input_abstract);
        if (need_add_to_depend_list) {
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
    if (dyn_size == -1) {
      (void)rec_depend_list.emplace(i);
      continue;
    }
    while (dyn_size >= 0) {
      auto depend_index = start_index + dyn_size;
      // skip primitive input
      const auto &input = inputs.at(LongToSize(depend_index + 1));
      MS_EXCEPTION_IF_NULL(input);
      const auto &input_abs = input->abstract();
      if (input_abs != nullptr && input_abs->isa<abstract::AbstractTensor>()) {
        (void)rec_depend_list.emplace(depend_index);
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
  if (primitive == nullptr) {
    return {};
  }
  auto prim_name = primitive->name();
  std::set<int64_t> ori = {};

  auto op_infer_opt = GetPrimitiveInferImpl(primitive);
  if (!op_infer_opt.has_value()) {
    // some operator will be mapped to new operator on Ascend like GatherV2, however they use same Infer information
    if (primitive->HasAttr(kAttrMeOpName)) {
      auto ori_prim_name = GetValue<std::string>(primitive->GetAttr(kAttrMeOpName));
      op_infer_opt = GetPrimitiveInferImpl(std::make_shared<Primitive>(ori_prim_name));
    }
  }

  if (op_infer_opt.has_value()) {
    auto op_infer = op_infer_opt.value().Get();
    if (op_infer != nullptr && ori.empty()) {
      ori = op_infer->GetValueDependArgIndices();
    }
    if (prim_name == ops::kNameShapeCalc) {
      auto value_depend_vector = GetValue<std::vector<bool>>(primitive->GetAttr(ops::kAttrValueDepend));
      for (size_t i = 0; i < value_depend_vector.size(); i++) {
        if (value_depend_vector[i]) {
          ori.insert(i);
        }
      }
    }
  } else if (ori.empty()) {
    MS_LOG(DEBUG) << "Not find infer function GetValueDependArgIndices, prim name: " << prim_name;
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

PrimitiveEvalImplMap *GetPrimitiveInferMapPtr() {
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
