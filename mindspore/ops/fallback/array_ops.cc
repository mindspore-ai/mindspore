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

#include <functional>
#include <vector>
#include <memory>
#include "backend/common/expander/fallback/fallback_irbuilder.h"
#include "utils/shape_utils.h"
#include "utils/check_convert_utils.h"
#include "ops_utils/op_utils.h"
#include "view/view_strides_calculator.h"

namespace mindspore {
namespace expander {
REG_FALLBACK_BUILDER("OneHotExt").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto depth = ib->GetInput(kIndex1);
  auto on_value = ib->GetInput(kIndex2);
  auto off_value = ib->GetInput(kIndex3);
  auto axis = ib->Value<int64_t>(-1);
  auto out = ib->Emit("OneHot", {x, depth, on_value, off_value, axis});
  return {out};
});

DEF_PURE_SHAPE_CALC(g_flatten_ext_fallback_shapecalc)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto &input_shape = inputs.at(kIndex0);
    auto start_dim = inputs.at(kIndex1)[0];
    auto end_dim = inputs.at(kIndex2)[0];
    int64_t dim_size = SizeToLong(input_shape.size());
    if (dim_size == 0) {
      return {{1}};
    }
    auto start_dim_fix = start_dim < 0 ? start_dim + dim_size : start_dim;
    auto end_dim_fix = end_dim < 0 ? end_dim + dim_size : end_dim;
    if (start_dim_fix == end_dim_fix) {
      return {input_shape};
    }

    auto begin = input_shape.begin() + start_dim_fix;
    auto end = input_shape.begin() + end_dim_fix + 1;
    auto slice_numel = std::accumulate(begin, end, static_cast<int64_t>(1), std::multiplies<int64_t>());
    ShapeVector shape;
    shape.reserve(dim_size - end_dim_fix + start_dim_fix);
    for (int64_t i = 0; i < start_dim_fix; i++) {
      shape.push_back(input_shape[i]);
    }
    shape.push_back(slice_numel);
    for (int64_t i = end_dim_fix + 1; i < dim_size; i++) {
      shape.push_back(input_shape[i]);
    }
    return {shape};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    if (IsDynamicRank(inputs[0])) {
      return {-1};
    }
    int64_t dim_size = SizeToLong(inputs[0].size());
    auto start_vec = inputs.at(kIndex1);
    auto end_vec = inputs.at(kIndex2);
    if (start_vec.empty() || end_vec.empty()) {
      return {-1};
    }
    auto start_dim = start_vec[0];
    auto end_dim = end_vec[0];
    if (dim_size == 0) {
      return {1};
    }
    auto start_dim_fix = start_dim < 0 ? start_dim + dim_size : start_dim;
    auto end_dim_fix = end_dim < 0 ? end_dim + dim_size : end_dim;
    auto res = dim_size - end_dim_fix + start_dim_fix;
    return {res};
  });

REG_FALLBACK_BUILDER("InplaceScatterSrc").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto dim = ib->GetInput(kIndex1);
  auto index = ib->GetInput(kIndex2);
  auto src = ib->GetInput(kIndex3);
  auto reduce = ib->Value(static_cast<int64_t>(Reduce::REDUCE_NONE));
  auto dim_val = dim->BuildValue();
  auto reduce_val = reduce->BuildValue();
  if (!IsValueKnown(dim_val) || !IsValueKnown(reduce_val)) {
    MS_EXCEPTION(ValueError) << "For `InplaceScatterSrc` op, the `dim` and `reduce` must be a constant!";
  }
  auto idx_shape = ib->GetShape(index);
  if (IsShapeNone(idx_shape)) {
    return {input};
  }
  auto out = ib->Emit("TensorScatterElements", {input, index, src, dim, reduce});
  return {out};
});

REG_FALLBACK_BUILDER("FlattenExt").SetBody(BODYFUNC(ib) {
  NodePtr input = ib->GetInput(kIndex0);
  NodePtr start_dim = ib->GetInput(kIndex1);
  NodePtr end_dim = ib->GetInput(kIndex2);
  auto shape = ib->ShapeCalc(g_flatten_ext_fallback_shapecalc, {input, start_dim, end_dim}, {kIndex1, kIndex2})[0];
  auto out = ib->Reshape(input, shape);
  return {out};
});

REG_FALLBACK_BUILDER("InplaceMaskedFillTensor").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto mask = ib->GetInput(kIndex1);
  auto value = ib->GetInput(kIndex2);
  auto out = ib->MaskedFill(input, mask, value);
  return {out};
});

REG_FALLBACK_BUILDER("ViewAs").SetBody(BODYFUNC(ib) {
  NodePtr input = ib->GetInput(kIndex0);
  NodePtr other = ib->GetInput(kIndex1);
  auto shape = ib->Shape(other);
  auto out = ib->Reshape(input, shape);
  return {out};
});

REG_FALLBACK_BUILDER("Ones").SetBody(BODYFUNC(ib) {
  auto size = ib->GetInput(kIndex0);
  auto dtype = ib->GetInput(kIndex1);
  auto dtype_ptr = dtype->BuildValue();
  auto dtype_val = GetValueWithCheck<int64_t>(dtype_ptr);
  auto out_type = TypeIdToType(static_cast<TypeId>(dtype_val));
  auto value = ib->Tensor(1, out_type);
  auto out = ib->Emit("FillV2", {size, value});
  return {out};
});

REG_FALLBACK_BUILDER("Zeros").SetBody(BODYFUNC(ib) {
  auto size = ib->GetInput(kIndex0);
  auto dtype = ib->GetInput(kIndex1);
  auto dtype_ptr = dtype->BuildValue();
  auto dtype_val = GetValueWithCheck<int64_t>(dtype_ptr);
  auto out_type = TypeIdToType(static_cast<TypeId>(dtype_val));
  auto value = ib->Tensor(0, out_type);
  auto out = ib->Emit("FillV2", {size, value});
  return {out};
});

REG_FALLBACK_BUILDER("OnesLikeExt").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto dtype = ib->GetInput(kIndex1);
  auto org_out = ib->Emit("OnesLike", {input});
  if (ib->GetDtype(dtype)->isa<TypeNone>()) {
    auto input_type = ib->GetDtype(input)->type_id();
    dtype = ib->Value(static_cast<int64_t>(input_type));
  }
  auto dtype_type = GetValue<int64_t>(dtype->BuildValue());
  auto out = ib->Cast(org_out, static_cast<TypeId>(dtype_type));
  return {out};
});

REG_FALLBACK_BUILDER("ZerosLikeExt").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto dtype = ib->GetInput(kIndex1);
  auto org_out = ib->Emit("ZerosLike", {input});
  if (ib->GetDtype(dtype)->isa<TypeNone>()) {
    auto input_type = ib->GetDtype(input)->type_id();
    dtype = ib->Value(static_cast<int64_t>(input_type));
  }
  auto dtype_type = GetValue<int64_t>(dtype->BuildValue());
  auto dtype_id = static_cast<TypeId>(dtype_type);
  if (dtype_id == kNumberTypeUInt64) {
    return {};
  }
  auto out = ib->Cast(org_out, dtype_id);
  return {out};
});

REG_FALLBACK_BUILDER("NewOnes").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto size = ib->GetInput(kIndex1);
  auto dtype = ib->GetInput(kIndex2);
  int64_t dtype_val;
  if (ib->GetDtype(dtype)->isa<TypeNone>()) {
    auto input_type = ib->GetDtype(input)->type_id();
    dtype_val = static_cast<int64_t>(input_type);
  } else {
    auto dtype_ptr = dtype->BuildValue();
    dtype_val = GetValueWithCheck<int64_t>(dtype_ptr);
  }
  auto out_type = TypeIdToType(static_cast<TypeId>(dtype_val));
  auto value = ib->Tensor(1, out_type);
  auto out = ib->Emit("FillV2", {size, value});
  return {out};
});

REG_FALLBACK_BUILDER("NewZeros").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto size = ib->GetInput(kIndex1);
  auto dtype = ib->GetInput(kIndex2);
  int64_t dtype_val;
  if (ib->GetDtype(dtype)->isa<TypeNone>()) {
    auto input_type = ib->GetDtype(input)->type_id();
    dtype_val = static_cast<int64_t>(input_type);
  } else {
    auto dtype_ptr = dtype->BuildValue();
    dtype_val = GetValueWithCheck<int64_t>(dtype_ptr);
  }
  auto out_type = TypeIdToType(static_cast<TypeId>(dtype_val));
  auto value = ib->Tensor(0, out_type);
  auto out = ib->Emit("FillV2", {size, value});
  return {out};
});

REG_FALLBACK_BUILDER("FillScalar").SetBody(BODYFUNC(ib) {
  auto size = ib->GetInput(kIndex0);
  auto fill_value = ib->GetInput(kIndex1);
  auto dtype = ib->GetInput(kIndex2);
  NodePtr value = nullptr;
  if (ib->GetDtype(dtype)->isa<TypeNone>()) {
    value = ib->ScalarToTensor(fill_value, ib->GetDtype(fill_value));
  } else {
    auto dtype_value_ptr = dtype->BuildValue();
    MS_EXCEPTION_IF_NULL(dtype_value_ptr);
    auto dtype_val = GetValueWithCheck<int64_t>(dtype_value_ptr);
    auto dtype_ptr = TypeIdToType(static_cast<TypeId>(dtype_val));
    MS_EXCEPTION_IF_NULL(dtype_ptr);
    value = ib->ScalarToTensor(fill_value, dtype_ptr);
  }
  auto out = ib->Emit("FillV2", {size, value});
  return {out};
});

REG_FALLBACK_BUILDER("FillTensor").SetBody(BODYFUNC(ib) {
  auto size = ib->GetInput(kIndex0);
  auto fill_value = ib->GetInput(kIndex1);
  auto dtype = ib->GetInput(kIndex2);
  auto org_out = ib->Emit("FillV2", {size, fill_value});
  if (ib->GetDtype(dtype)->isa<TypeNone>()) {
    auto input_type = ib->GetDtype(fill_value)->type_id();
    dtype = ib->Value(static_cast<int64_t>(input_type));
  }
  auto out = ib->Emit("Cast", {org_out, dtype});
  return {out};
});

NodePtrList SplitTensorFallbackFunc(FallbackIRBuilder *ib, const NodePtr &input, const NodePtr &split_size,
                                    int64_t dim) {
  constexpr int64_t SPLIT_LOOP_SIZE = 32;
  auto split_size_val = GetValue<int64_t>(split_size->BuildValue());
  const auto &input_shape = input->shape();
  auto dimShape = input_shape[dim];
  if (static_cast<int64_t>(split_size_val) == dimShape) {
    return {input};
  } else {
    int64_t numSplit = (split_size_val + dimShape - 1) / split_size_val;
    int64_t lastSplitSize = split_size_val - (static_cast<int64_t>(split_size_val) * numSplit - dimShape);
    std::vector<int64_t> splitVector(numSplit, static_cast<int64_t>(split_size_val));
    splitVector[numSplit - 1] = lastSplitSize;
    // Using loop splitting in the AiCore scene of the SplitV operator or when the number of outputs exceeds 32
    if (splitVector.size() > SPLIT_LOOP_SIZE) {
      std::vector<mindspore::expander::NodePtr> subTensors;
      const int64_t loopSize = (SizeToLong(splitVector.size()) + SPLIT_LOOP_SIZE - 1) / SPLIT_LOOP_SIZE;
      const int64_t lastSize = SizeToLong(splitVector.size()) % SPLIT_LOOP_SIZE;
      const size_t dimSize = input->shape().size();
      // Construct splitsize as a new splitSize based on loopSize and lastSize
      std::vector<int64_t> newSplitSize;
      std::vector<int64_t> splitTmp;
      std::vector<std::vector<int64_t>> splitList;
      for (int64_t loopIndex = 0; loopIndex < loopSize; loopIndex++) {
        int64_t newSplit = 0;
        int64_t currentSplitVal = 0;
        if (loopIndex != loopSize - 1) {
          for (int64_t noLastIndex = 0; noLastIndex < SPLIT_LOOP_SIZE; noLastIndex++) {
            currentSplitVal = *(splitVector.data() + loopIndex * SPLIT_LOOP_SIZE + noLastIndex);
            splitTmp.emplace_back(currentSplitVal);
            newSplit += currentSplitVal;
          }
        } else {
          for (int64_t lastIndex = 0; lastIndex < lastSize; lastIndex++) {
            currentSplitVal = *(splitVector.data() + loopIndex * SPLIT_LOOP_SIZE + lastIndex);
            splitTmp.emplace_back(currentSplitVal);
            newSplit += currentSplitVal;
          }
        }
        splitList.emplace_back(splitTmp);
        newSplitSize.emplace_back(newSplit);
        splitTmp.clear();
      }
      // Loop call Slice to split self into N large blocks, and use SplitV to split each block again
      int64_t offsetVal = 0;
      for (size_t sliceIndex = 0; sliceIndex < newSplitSize.size(); sliceIndex++) {
        // Calculate offset, increasing offset block by block
        std::vector<int64_t> offsetVector(dimSize, 0);
        offsetVal += sliceIndex == 0 ? 0 : newSplitSize[sliceIndex - 1];
        offsetVector[static_cast<size_t>(dim)] = offsetVal;
        // Calculate size, which is consistent with the output block size
        std::vector<int64_t> sizeVector;
        for (size_t selfIndex = 0; selfIndex < dimSize; selfIndex++) {
          int64_t sizeValue =
            selfIndex == static_cast<size_t>(dim) ? newSplitSize[sliceIndex] : input->shape()[selfIndex];
          sizeVector.emplace_back(sizeValue);
        }
        // Using Slice to process each block
        auto slice = ib->Emit("Slice", {input, ib->Value(offsetVector), ib->Value(sizeVector)});

        // Using SPLitV to slice
        auto out = ib->Emit("SplitV", {slice},
                            {{"size_splits", MakeValue(splitList[sliceIndex])},
                             {"split_dim", MakeValue<int64_t>(dim)},
                             {"num_split", MakeValue<int64_t>(splitList[sliceIndex].size())}});
        for (size_t i = 0; i < splitList[sliceIndex].size(); i++) {
          subTensors.emplace_back(ib->TupleGetItem(out, i));
        }
      }
      return {ib->MakeTuple(subTensors)};
    } else {
      auto out = ib->Emit("SplitV", {input},
                          {{"size_splits", MakeValue(splitVector)},
                           {"split_dim", MakeValue<int64_t>(dim)},
                           {"num_split", MakeValue<int64_t>(splitVector.size())}});
      return {out};
    }
  }
}

DEF_PURE_SHAPE_CALC(narrow_fallback_calc)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    const auto &input_shape = inputs.at(kIndex0);
    int64_t rank = static_cast<int64_t>(input_shape.size());

    auto dim = inputs.at(kIndex1).back();
    dim = ops::DynamicDimWrap(dim, rank);

    auto start = inputs.at(kIndex2).back();
    start = start < 0 ? start + input_shape[dim] : start;

    auto length = inputs.at(kIndex3).back();

    std::vector<int64_t> begin(rank, 0);
    begin[dim] = start;

    std::vector<int64_t> size(input_shape);
    size[dim] = length;

    return {begin, size};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    const auto &input_shape = inputs[kIndex0];
    if (IsDynamicRank(input_shape)) {
      return {-1, -1};
    }
    int64_t rank = static_cast<int64_t>(input_shape.size());
    return {rank, rank};
  });

NodePtrList SplitWithSizeFallbackFunc(FallbackIRBuilder *ib, const NodePtr &input, const NodePtr &split_size_node,
                                      int64_t dim) {
  auto split_size = GetValue<std::vector<int64_t>>(split_size_node->BuildValue());
  if (split_size.size() == 1) {
    return {input};
  } else {
    NodePtrList outputs;
    auto dim_node = ib->Value(dim);
    int64_t start = 0;
    for (size_t i = 0; i < split_size.size(); ++i) {
      auto start_node = ib->Value(start);
      auto length_node = ib->Value(split_size[i]);
      auto shape_cal_result =
        ib->ShapeCalc(narrow_fallback_calc, {input, dim_node, start_node, length_node}, {kIndex1, kIndex2, kIndex3});
      outputs.push_back(ib->Slice(input, shape_cal_result[0], shape_cal_result[1]));
      start += split_size[i];
    }
    return outputs;
  }
}

REG_FALLBACK_BUILDER("SplitTensor").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto split_size = ib->GetInput(kIndex1);
  auto dim = ib->GetInput(kIndex2);
  if (!IsValueKnown(dim->BuildValue()) || !IsValueKnown(split_size->BuildValue())) {
    MS_EXCEPTION(ValueError) << "For `SplitWithTensor` , the `split_size` and `dim` must currently be a constant!";
  }
  const auto &input_shape = input->shape();
  auto dim_val = GetValue<int64_t>(dim->BuildValue());
  if (IsDynamicRank(input_shape)) {
    MS_EXCEPTION(ValueError)
      << "For `SplitTensor` op, the variable `input` is with dynamic rank, which is unsupported for now!";
  }
  if (dim_val < 0) {
    dim_val += SizeToLong(input_shape.size());
  }
  if (input_shape[dim_val] == abstract::Shape::kShapeDimAny) {
    MS_EXCEPTION(ValueError)
      << "For `Chunk` op, the target dim of `input` is with dynamic shape, which is unsupported for now!";
  }
  return SplitTensorFallbackFunc(ib, input, split_size, dim_val);
});

REG_FALLBACK_BUILDER("SplitWithSize").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto split_size = ib->GetInput(kIndex1);
  auto dim = ib->GetInput(kIndex2);
  if (!IsValueKnown(dim->BuildValue()) || !IsValueKnown(split_size->BuildValue())) {
    MS_EXCEPTION(ValueError) << "For `SplitWithSize` , the `split_size` and `dim` must currently be a constant!";
  }
  const auto &input_shape = input->shape();
  auto dim_val = GetValue<int64_t>(dim->BuildValue());
  if (dim_val < 0 && !IsDynamicRank(input_shape)) {
    dim_val += SizeToLong(input_shape.size());
  }
  return SplitWithSizeFallbackFunc(ib, input, split_size, dim_val);
});

REG_FALLBACK_BUILDER("Chunk").SetBody(BODYFUNC(ib) {
  auto in_tensor = ib->GetInput(kIndex0);
  auto chunks = ib->GetInput(kIndex1);
  auto dims = ib->GetInput(kIndex2);
  auto dim_value_ptr = dims->BuildValue();
  if (dim_value_ptr->isa<ValueAny>()) {
    MS_EXCEPTION(ValueError) << "For `Chunk` op, the `dims` only supports constant value for now!";
  }
  auto dim_value = GetValue<int64_t>(dim_value_ptr);
  auto chunks_value_ptr = chunks->BuildValue();
  if (chunks_value_ptr->isa<ValueAny>()) {
    MS_EXCEPTION(ValueError) << "For `Chunk` op, the variable `chunks` only supports constant value for now!";
  }
  auto chunks_value = GetValue<int64_t>(chunks_value_ptr);
  const auto &input_shape = in_tensor->shape();
  if (IsDynamicRank(input_shape)) {
    MS_EXCEPTION(ValueError)
      << "For `Chunk` op, the variable `input` is with dynamic rank, which is unsupported for now!";
  }
  if (dim_value < 0) {
    dim_value += SizeToLong(input_shape.size());
  }
  if (input_shape[dim_value] == abstract::Shape::kShapeDimAny) {
    MS_EXCEPTION(ValueError)
      << "For `Chunk` op, the target dim of `input` is with dynamic shape, which is unsupported for now!";
  }
  int64_t dim_size = input_shape[dim_value];
  int64_t split_size = (dim_size + chunks_value - 1) / chunks_value;
  if (split_size == 0 && dim_size == 0) {
    auto split_sizes = std::vector<ValuePtr>(chunks_value, std::make_shared<Int64Imm>(split_size));
    auto split_sizes_tuple = std::make_shared<ValueTuple>(split_sizes);
    return SplitWithSizeFallbackFunc(ib, in_tensor, ib->EmitValue(split_sizes_tuple), dim_value);
  } else {
    return SplitTensorFallbackFunc(ib, in_tensor, ib->Value(split_size), dim_value);
  }
});

REG_FALLBACK_BUILDER("InsertGemV2InBackward").SetBody(BODYFUNC(ib) { return {ib->GetInput(kIndex0)}; });

REG_FALLBACK_BUILDER("UnstackExtView").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto input_shape = input->shape();
  auto num = SizeToLong(input_shape.size());
  auto output_tuple = ib->Emit("Unstack", {input}, {{"num", MakeValue(num)}, {"axis", MakeValue(axis->BuildValue())}});
  return {output_tuple};
});

REG_FALLBACK_BUILDER("TransposeView").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto perm = ib->GetInput(kIndex1);
  return {ib->Transpose(input, perm)};
});

REG_FALLBACK_BUILDER("GatherNdExt").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto out = ib->GatherNd(input, indices);
  return {out};
});
}  // namespace expander
}  // namespace mindspore
