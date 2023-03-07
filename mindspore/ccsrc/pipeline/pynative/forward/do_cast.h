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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_FORWARD_DO_CAST_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_FORWARD_DO_CAST_H_

#include <vector>
#include <string>
#include <memory>
#include "pipeline/pynative/base.h"
#include "pipeline/pynative/pynative_cache.h"

namespace mindspore {
namespace pynative {
class CastOperation {
 public:
  CastOperation() = default;
  ~CastOperation() = default;
  void DoCast(const FrontendOpRunInfoPtr &op_run_info);

 private:
  bool IsValueTypeInvalid(const ValuePtr &v) const;
  ValuePtr GetDstType(const TypeId &type_id) const;
  TypeId JudgeMaxType(TypeId max_type, bool has_scalar_float32, bool has_scalar_int64, bool has_tensor_int8) const;
  void GetDstType(const FrontendOpRunInfoPtr &op_run_info,
                  const mindspore::HashMap<SignatureEnumDType, std::vector<size_t>> &type_indexes,
                  mindspore::HashMap<SignatureEnumDType, TypeId> *dst_type) const;
  const std::string &TypeIdToMsTypeStr(const TypeId &type_id) const;
  bool GetSignatureType(const std::vector<Signature> &signatures, std::vector<SignatureEnumDType> *dtypes) const;
  void GetTypeIndex(const std::vector<SignatureEnumDType> &dtypes,
                    mindspore::HashMap<SignatureEnumDType, std::vector<size_t>> *type_indexes) const;
  void SetTensorMixPrecisionCast(const FrontendOpRunInfoPtr &op_run_info) const;
  ValuePtr DoParamMixPrecisionCastTuple(const FrontendOpRunInfoPtr &op_run_info, bool *is_cast,
                                        const ValueSequencePtr &value_seq, const std::string &op_name,
                                        size_t index) const;
  ValuePtr DoParamMixPrecisionCast(const FrontendOpRunInfoPtr &op_run_info, bool *is_cast, const ValuePtr &v,
                                   const std::string &op_name, size_t index) const;
  ValuePtr DoAutoCast(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v, const TypeId &type_id,
                      const std::string &op_name, size_t index) const;
  void DoSignatureCast(const FrontendOpRunInfoPtr &op_run_info,
                       const mindspore::HashMap<SignatureEnumDType, TypeId> &dst_type,
                       const std::vector<SignatureEnumDType> &dtypes) const;
  void SetImplicitCast(const FrontendOpRunInfoPtr &op_run_info);

 private:
  PrimitivePtr cast_prim_{nullptr};
  ImplicitCastCache implicit_cast_map_;
};
using CastOperationPtr = std::shared_ptr<CastOperation>;
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_FORWARD_DO_CAST_H_
