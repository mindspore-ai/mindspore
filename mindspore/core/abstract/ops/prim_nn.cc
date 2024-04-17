/**
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

#include "abstract/ops/infer_functions.h"
#include "abstract/utils.h"
#include "abstract/param_validator.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace abstract {
int64_t GetAndCheckFormat(const ValuePtr &value) {
  int64_t data_format;
  bool result = CheckAndConvertUtils::GetDataFormatEnumValue(value, &data_format);
  if (!result ||
      (data_format != static_cast<int64_t>(Format::NHWC) && data_format != static_cast<int64_t>(Format::NCHW) &&
       data_format != static_cast<int64_t>(Format::NCDHW))) {
    MS_LOG(EXCEPTION) << "data format is invalid, only support NCHW, NHWC and NCDHW";
  }
  return data_format;
}

AbstractBasePtr InferImplBpropCut(const AnalysisEnginePtr &, const PrimitivePtr &,
                                  const AbstractBasePtrList &args_abs_list) {
  // Inputs: a tensor.
  AbstractBasePtrList args_list;
  constexpr size_t out_and_dout_size = 2;
  for (size_t i = 0; i < args_abs_list.size() - out_and_dout_size; i++) {
    args_list.push_back(args_abs_list[i]->Broaden());
  }
  return std::make_shared<AbstractTuple>(args_list);
}

AbstractBasePtr InferImplSparseApplyProximalAdagrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                    const AbstractBasePtrList &args_abs_list) {
  const size_t size_expected = 7;
  CheckRequiredArgsSize(primitive->name(), args_abs_list, size_expected);
  AbstractBasePtrList elements;
  const size_t args_size = 2;
  for (size_t i = 0; i < args_size; ++i) {
    elements.push_back(args_abs_list[i]->Clone()->Broaden());
  }
  return std::make_shared<AbstractTuple>(elements);
}

AbstractBasePtr InferImplSGD(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const AbstractBasePtrList &args_abs_list) {
  const size_t size_expected = 6;
  CheckRequiredArgsSize(primitive->name(), args_abs_list, size_expected);
  AbstractBasePtrList elements;
  elements.push_back(args_abs_list[0]->Clone()->Broaden());
  return std::make_shared<AbstractTuple>(elements);
}

AbstractBasePtr InferImplPad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const AbstractBasePtrList &args_abs_list) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_abs_list, 1);
  auto arg = CheckArg<AbstractTensor>(op_name, args_abs_list, 0);
  auto input_shp = arg->shape()->shape();
  auto padding_attr = primitive->GetAttr("paddings");
  MS_EXCEPTION_IF_NULL(padding_attr);
  if (!padding_attr->isa<ValueTuple>()) {
    MS_LOG(EXCEPTION) << "Paddings is not a ValueTuple";
  }
  std::vector<ValuePtr> paddings = padding_attr->cast<ValueTuplePtr>()->value();
  std::vector<std::vector<int64_t>> paddings_vec;
  for (ValuePtr paddings_elements : paddings) {
    auto item = paddings_elements->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(item);
    std::vector<ValuePtr> paddings_elements_tuple = item->value();
    std::vector<int64_t> paddings_vec_item;
    (void)std::transform(std::begin(paddings_elements_tuple), std::end(paddings_elements_tuple),
                         std::back_inserter(paddings_vec_item),
                         [](const ValuePtr &e) -> int64_t { return GetValue<int64_t>(e); });
    paddings_vec.push_back(paddings_vec_item);
  }

  ShapeVector result_shp;
  size_t length = paddings_vec.size();
  for (size_t i = 0; i < length; ++i) {
    if (paddings_vec[i].size() != 2) {
      MS_LOG(EXCEPTION) << "Paddings 's second dim size is not 2";
    }
    result_shp.push_back(input_shp[i] + paddings_vec[i][0] + paddings_vec[i][1]);
  }
  return std::make_shared<AbstractTensor>(arg->element(), std::make_shared<Shape>(result_shp));
}
}  // namespace abstract
}  // namespace mindspore
