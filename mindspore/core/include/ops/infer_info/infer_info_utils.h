/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_INFER_INFO_UTILS_H_
#define MINDSPORE_CORE_OPS_INFER_INFO_UTILS_H_

#include <memory>
#include "ops/infer_info/infer_info.h"
#include "ops/op_def.h"
#include "utils/simple_info.h"

namespace mindspore::ops {
InferInfoPtrList ConvertAbstractListToInferInfoList(const AbstractBasePtrList &abstract_list, const OpDefPtr op_def);

class OpFrontendFuncImpl;
using OpFrontendFuncImplPtr = std::shared_ptr<OpFrontendFuncImpl>;
MS_CORE_API AbstractBasePtr DoGeneralInfer(const PrimitivePtr prim, const AbstractBasePtrList &abstract_list,
                                           const OpFrontendFuncImplPtr frontend_func_impl = nullptr);

MS_CORE_API ValueSimpleInfoPtr DoGeneralInfer(const PrimitivePtr &prim, const ValuePtrList &values);

template <typename... T>
inline ValueSimpleInfoPtr DoGeneralInfer(const PrimitivePtr &prim, const T &...t) {
  ValuePtrList values;
  (values.push_back(ConvertValuePtr(t)), ...);
  return DoGeneralInfer(prim, values);
}
}  // namespace mindspore::ops
#endif  // MINDSPORE_CORE_OPS_INFER_INFO_UTILS_H_
