/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_GROUPED_MATMUL_BASE_H_
#define MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_GROUPED_MATMUL_BASE_H_

#include <memory>
#include <vector>
#include <utility>
#include <string>

#include "mindapi/base/types.h"
#include "ops/ops_func_impl/op_func_impl.h"

namespace mindspore {
namespace ops {
class OPS_API GroupedMatmulBaseFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;

  int32_t CheckValidation(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;

  bool GeneralInferRegistered() const override { return true; }

 protected:
  struct Indexes {
    int64_t x = 0;
    int64_t weight = 1;
    int64_t split_item_offset = -4;
    int64_t group_type_offset = -3;
    int64_t transpose_a_offset = -2;
    int64_t transpose_b_offset = -1;
  } idxes_;

  virtual void FetchGroupInfo(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
    MS_LOG(EXCEPTION) << "Not implement exception";
  }

  virtual int64_t FetchGroupListIndex(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
    MS_LOG(EXCEPTION) << "Not implement exception";
  }

  virtual int64_t FetchGroupListSize(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
    MS_LOG(EXCEPTION) << "Not implement exception";
  }

  std::pair<int32_t, int64_t> CommonCheckValidation(const PrimitivePtr &primitive,
                                                    const InferInfoPtrList &input_infos) const;

  virtual int32_t PrivateCheckValidation(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos,
                                         int64_t group_type) const {
    return OP_CHECK_SUCCESS;
  }

  virtual bool GetTransposeValue(const InferInfoPtrList &input_infos, int64_t transpose_index) const { return false; }

 private:
  std::pair<ShapeArray, ShapeArray> FetchInputAndWeightShapes(const PrimitivePtr &primitive,
                                                              const InferInfoPtrList &input_infos) const;

  void CheckInputAndWeightShapeForSingleOutput(const PrimitivePtr &primitive, const ShapeVector &x_shape,
                                               const ShapeVector &w_shape, int64_t group_type, bool transpose_b) const;

  ShapeArray InferShapeForSingleOutput(const PrimitivePtr &primitive, const ShapeArray &x_shapes,
                                       const ShapeArray &w_shapes, int64_t group_list_size, int64_t group_type,
                                       bool transpose_b, bool is_int4 = false) const;

  void CheckInputAndWeightShapeForMultiOutput(const PrimitivePtr &primitive, const ShapeVector &x_shape,
                                              const ShapeVector &w_shape, size_t i) const;

  ShapeArray InferShapeForMultiOutput(const PrimitivePtr &primitive, const ShapeArray &x_shapes,
                                      const ShapeArray &w_shapes) const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_GROUPED_MATMUL_BASE_H_
