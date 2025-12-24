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

#ifndef MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_CONV_PADDING_H_
#define MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_CONV_PADDING_H_

#include <memory>
#include <vector>
#include <string>
#include <utility>
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/types.h"
#include "ops/ops_func_impl/op_func_impl.h"

namespace mindspore {
namespace ops {
class OPS_API ConvPaddingFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;
  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;
  bool GeneralInferRegistered() const override { return true; };

 protected:
  std::pair<ShapeVector, bool> Batchify(const ShapeVector &input_shape, int64_t num_spatial_dims,
                                        const std::string &prim_name) const;
  ShapeArray ConvNdInferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos,
                              const ShapeVector &input_shape, const ShapeVector &weight_shape) const;
  struct Indexes {
    size_t input_idx{0};
    size_t weight_idx{1};
    size_t bias_idx{2};
    size_t stride_idx{3};
    size_t padding_idx{4};
    size_t dilation_idx{5};
    size_t transposed_idx{6};
    size_t output_padding_idx{7};
    size_t groups_idx{8};
  } idxes_;

 private:
  void IndicesCheckPositiveVec(const string &arg_name, const ArrayValue<int64_t> &array, const string &prim_name,
                               bool exclude_zeros, bool padding_stride) const;
  ShapeArray ConvNdPaddingCommonInferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos,
                                           const ShapeVector &input_shape, const ShapeVector &weight_shape,
                                           const ShapeVector &output_shpe) const;
  int64_t GetOutputHWPadding(const ShapeVector &input_shape, const ShapeVector &weight_shape, size_t shape_pos,
                             size_t i, const ArrayValue<int64_t> &stride, const mindspore::PadMode &padding_enum,
                             const ArrayValue<int64_t> &dilation) const;
  ShapeArray DynamicRankInfer(const InferInfoPtrList &input_infos) const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_CONV_PADDING_H_
