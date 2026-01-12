/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_OP_UTILS_H
#define MINDSPORE_CORE_OPS_OP_UTILS_H
#include <algorithm>
#include <climits>
#include <memory>
#include <utility>
#include <set>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include "ir/dtype/tensor_type.h"
#include "utils/value_utils.h"
#include "ops/op_def_utils.h"
#include "mindapi/base/shape_vector.h"
#include "primitive/math_ops.h"
#include "primitive/op_name.h"
#include "ops_utils/op_constants.h"
#include "mindapi/base/types.h"
#include "primitive/other_op_name.h"
#include "primitive/array_op_name.h"
#include "primitive/math_op_name.h"
#include "mindspore/core/include/ops/infer_info/infer_info.h"
#include "ops/infer_info/infer_info.h"

namespace mindspore::ops {
constexpr auto kBitSize = 64;
constexpr double SCALE_ZERO_THRESHOLD = 0.0;
constexpr double SCALE_ONE = 1.0;
const std::set<TypePtr> common_valid_types = {kInt8,   kInt16,  kInt32,   kInt64,   kUInt8,   kUInt16,
                                              kUInt32, kUInt64, kFloat16, kFloat32, kFloat64, kBFloat16};
const std::set<TypePtr> common_valid_types_with_bool = {
  kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16, kUInt32, kUInt64, kFloat16, kFloat32, kFloat64, kBool, kBFloat16};

const std::set<TypePtr> common_valid_types_with_complex = {kInt8,    kInt16,     kInt32,      kInt64,   kUInt8,
                                                           kUInt16,  kUInt32,    kUInt64,     kFloat16, kFloat32,
                                                           kFloat64, kComplex64, kComplex128, kBFloat16};

const std::set<TypePtr> common_valid_types_with_complex_and_bool = {
  kInt8,    kInt16,   kInt32,   kInt64,     kUInt8,      kUInt16, kUInt32,  kUInt64,
  kFloat16, kFloat32, kFloat64, kComplex64, kComplex128, kBool,   kBFloat16};

const std::set<TypePtr> common_integral_types = {kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16, kUInt32, kUInt64};
const std::set<TypePtr> common_float_types = {kFloat16, kFloat32, kFloat64, kBFloat16};
const std::set<TypePtr> all_types = {kBool,    kInt,     kInt8,    kInt16,     kInt32,      kInt64,
                                     kUInt,    kUInt8,   kUInt16,  kUInt32,    kUInt64,     kFloat,
                                     kFloat16, kFloat32, kFloat64, kComplex64, kComplex128, kBFloat16};

const std::set<TypeId> common_valid_type_ids = {kNumberTypeInt8,    kNumberTypeInt16,   kNumberTypeInt32,
                                                kNumberTypeInt64,   kNumberTypeUInt8,   kNumberTypeUInt16,
                                                kNumberTypeUInt32,  kNumberTypeUInt64,  kNumberTypeFloat16,
                                                kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeBFloat16};

const std::set<TypeId> common_valid_type_ids_with_bool = {
  kNumberTypeInt8,    kNumberTypeInt16,  kNumberTypeInt32,   kNumberTypeInt64,   kNumberTypeUInt8,
  kNumberTypeUInt16,  kNumberTypeUInt32, kNumberTypeUInt64,  kNumberTypeFloat16, kNumberTypeFloat32,
  kNumberTypeFloat64, kNumberTypeBool,   kNumberTypeBFloat16};

const std::set<TypeId> common_valid_type_ids_with_complex = {
  kNumberTypeInt8,    kNumberTypeInt16,     kNumberTypeInt32,      kNumberTypeInt64,   kNumberTypeUInt8,
  kNumberTypeUInt16,  kNumberTypeUInt32,    kNumberTypeUInt64,     kNumberTypeFloat16, kNumberTypeFloat32,
  kNumberTypeFloat64, kNumberTypeComplex64, kNumberTypeComplex128, kNumberTypeBFloat16};

const std::set<TypeId> common_mint_valid_type_ids = {kNumberTypeInt8,    kNumberTypeInt16,   kNumberTypeInt32,
                                                     kNumberTypeInt64,   kNumberTypeUInt8,   kNumberTypeFloat16,
                                                     kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeBFloat16};

const std::set<TypeId> common_mint_valid_type_ids_with_bool = {
  kNumberTypeInt8, kNumberTypeInt16,   kNumberTypeInt32,   kNumberTypeInt64,   kNumberTypeUInt8,
  kNumberTypeBool, kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeBFloat16};

const std::vector<TypeId> common_mint_valid_type_ids_with_complex_and_bool_vec = {
  kNumberTypeInt8,    kNumberTypeInt16,    kNumberTypeInt32,     kNumberTypeInt64,
  kNumberTypeUInt8,   kNumberTypeBool,     kNumberTypeFloat16,   kNumberTypeFloat32,
  kNumberTypeFloat64, kNumberTypeBFloat16, kNumberTypeComplex64, kNumberTypeComplex128};

const std::set<TypeId> common_valid_type_ids_with_complex_and_bool = {
  kNumberTypeInt8,    kNumberTypeInt16,     kNumberTypeInt32,      kNumberTypeInt64,   kNumberTypeUInt8,
  kNumberTypeUInt16,  kNumberTypeUInt32,    kNumberTypeUInt64,     kNumberTypeFloat16, kNumberTypeFloat32,
  kNumberTypeFloat64, kNumberTypeComplex64, kNumberTypeComplex128, kNumberTypeBool,    kNumberTypeBFloat16};

const std::set<TypeId> common_integral_type_ids = {kNumberTypeInt8,   kNumberTypeInt16, kNumberTypeInt32,
                                                   kNumberTypeInt64,  kNumberTypeUInt8, kNumberTypeUInt16,
                                                   kNumberTypeUInt32, kNumberTypeUInt64};

const std::set<TypeId> common_float_type_ids = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64,
                                                kNumberTypeBFloat16};

const std::set<TypeId> all_type_ids = {
  kNumberTypeBool,      kNumberTypeInt,        kNumberTypeInt8,    kNumberTypeInt16,   kNumberTypeInt32,
  kNumberTypeInt64,     kNumberTypeUInt,       kNumberTypeUInt8,   kNumberTypeUInt16,  kNumberTypeUInt32,
  kNumberTypeUInt64,    kNumberTypeFloat,      kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64,
  kNumberTypeComplex64, kNumberTypeComplex128, kNumberTypeBFloat16};

std::vector<int64_t> CalBroadCastShape(const std::vector<int64_t> &x_shape, const std::vector<int64_t> &y_shape,
                                       const std::string &op_name, const std::string &op_x_name = "input1",
                                       const std::string &op_y_name = "input2");
OPS_API std::vector<int64_t> CalBroadCastShapeV2(const std::vector<int64_t> &x_shape,
                                                 const std::vector<int64_t> &y_shape, const std::string &op_name,
                                                 const std::string &op_x_name = "input1",
                                                 const std::string &op_y_name = "input2");
abstract::ShapePtr BroadCastInferShape(const std::string &op_name,
                                       const std::vector<abstract::AbstractBasePtr> &input_args);
bool IsBroadcastable(const std::vector<int64_t> &x_shape, const std::vector<int64_t> &y_shape);
ShapeVector BroadCastInferShape(const std::string &op_name, const ValuePtrList &input_values);
BaseShapePtr EltwiseGradInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);
TypePtr EltwiseGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);
TypePtrList EltwiseGradSimpleInferType(const PrimitivePtr &primitive, const ValuePtrList &input_values);
ShapeArray EltwiseGradSimpleInferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values);
void ReduceFuncCheckAxisInferImpl(const PrimitivePtr &prim, std::vector<int64_t> *axis, const size_t dim);
bool CheckAndGetAxisValue(const std::vector<abstract::AbstractBasePtr> &input_args, std::vector<int64_t> *axis_value,
                          int64_t *axis_shape_v, const PrimitivePtr &primitive);
ShapeVector ReduceFuncCalShapeAxisDyn(const ShapeVector &x_shape, bool keep_dims = false);
ShapeVector ReduceFuncCalShapeInferImpl(const PrimitivePtr &primitive, const ShapeVector &x_shape,
                                        const std::vector<int64_t> &axis, bool keep_dims_value = false);
abstract::ShapePtr ReduceBaseInferShape(const PrimitivePtr &primitive,
                                        const std::vector<abstract::AbstractBasePtr> &input_args,
                                        const std::string &prim_name);
TypePtr ReduceBaseInferType(const PrimitivePtr &prim, const std::vector<abstract::AbstractBasePtr> &input_args,
                            const std::set<TypePtr> &check_list);
abstract::ShapePtr ReduceExtInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);
TypePtr ReduceExtInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args);

void BlockInvalid(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args, ShapeVector out_shape);
BaseShapePtr SetPadShape(const ShapeVector &x_shape, const ArrayValue<int64_t> &paddings);
BaseShapePtr PadInferShapeBase(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                               const size_t pad_dim);
size_t GetHashId(int a, int b);
TypeId ConvertTypeBetweenTensorAndScalar(const TypeId &tensor_type_id, const TypeId &scalar_type_id,
                                         const size_t hash_id);

bool ObscureShapeEqual(const ShapeVector &lhs, const ShapeVector &rhs);

// Get the shape value from abstract input arg
// Ops like DynamicBroadcastTo or Reshape can directly get the shape value
// from input which represents shape by invoking this function
// Do not support input with type of AbstractTuple of AbstractTensor
ShapeVector GetShapeValue(const PrimitivePtr &primitive, const AbstractBasePtr &input_arg);

inline ShapeVector ConvertBaseShapeToTensorShape(const BaseShapePtr &base) {
  auto shape_ptr = base->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  return shape_ptr->shape();
}

inline ShapeVector GetShapeFromTensor(const AbstractBasePtr &abs) {
  auto base_shape = abs->GetShape();
  return ConvertBaseShapeToTensorShape(base_shape);
}

void CheckSparseShape(ShapeVector sparse_shp, ShapeVector dense_shp);

void CheckSparseShape(const size_t shape_size, const size_t expected_dim, const std::string &arg_name);

void CheckSparseIndicesDtype(const TypePtr data_type, const std::string &arg_name);

void CheckSparseIndicesDtypeInt32(const TypePtr data_type, const std::string &arg_name);

inline void CheckInputShapeEmpty(const std::string &prim_name, const std::vector<AbstractBasePtr> &input_args) {
  for (size_t i = 0; i < input_args.size(); ++i) {
    MS_EXCEPTION_IF_NULL(input_args[i]->GetShape());
    if (input_args[i]->GetShape()->IsDimZero()) {
      MS_LOG(EXCEPTION) << "For '" << prim_name << "', input " << i << "'s shape should not be empty!";
    }
  }
}

ShapeVector ConvertToShapeVector(const abstract::AbstractTuplePtr &shape);

template <typename T>
std::shared_ptr<T> InferSparseAttr(const PrimitivePtr &primitive, const AbstractBasePtrList &args_abs_list);

template <typename T>
AbstractBasePtr TensorToSequenceInfer(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);

template <typename T>
AbstractBasePtr InferSequenceSetItem(const PrimitivePtr &primitive, const AbstractBasePtrList &args_abs_list);

TypePtr HighPriorityType(const TypePtr &x_type, const TypePtr &y_type, const std::string &op_name);

constexpr auto kCSRAvgRows = "csr_avg_rows";
constexpr auto kIsCSR = "is_csr";
constexpr auto kCSRDenseShape = "dense_shape";
constexpr auto kCSRAxis = "axis";
constexpr auto kHasDynamicValue = "has_dynamic_value";
constexpr int64_t kModeValid = 2;
constexpr int64_t kModeSame = 1;
constexpr int64_t kModePad = 0;
constexpr int64_t kModeCalculated = 0;

inline int64_t get_batch_rank(const PrimitivePtr &prim) {
  if (prim->HasAttr(kBatchRank)) {
    auto value_ptr = prim->GetAttr(kBatchRank);
    return GetValue<int64_t>(value_ptr);
  }
  return 0;
}

inline int64_t PadModeStringToInt(const std::string &pad) {
  std::string pad_mode = pad;
  (void)std::transform(pad_mode.begin(), pad_mode.end(), pad_mode.begin(), toupper);
  if (pad_mode == "VALID") {
    return kModeValid;
  } else if (pad_mode == "SAME") {
    return kModeSame;
  } else if (pad_mode == "PAD") {
    return kModePad;
  } else if (pad_mode == "CALCULATED") {
    return kModeCalculated;
  } else {
    MS_LOG(EXCEPTION) << "Got an invalid pad_mode string: " << pad_mode << ".";
  }
}

static inline TypeId PromoteType(TypeId a, TypeId b, const std::string &op_name) {
  const auto f32 = kNumberTypeFloat32;
  const auto f16 = kNumberTypeFloat16;
  const auto f64 = kNumberTypeFloat64;
  const auto bf16 = kNumberTypeBFloat16;
  const auto s8 = kNumberTypeInt8;
  const auto u8 = kNumberTypeUInt8;
  const auto s16 = kNumberTypeInt16;
  const auto u16 = kNumberTypeUInt16;
  const auto s32 = kNumberTypeInt32;
  const auto u32 = kNumberTypeUInt32;
  const auto s64 = kNumberTypeInt64;
  const auto u64 = kNumberTypeUInt64;
  const auto b1 = kNumberTypeBool;
  const auto c64 = kNumberTypeComplex64;
  const auto c128 = kNumberTypeComplex128;
  const auto ud = kTypeUnknown;

  static std::unordered_map<TypeId, size_t> typeid_idx = {{f32, 0},  {f16, 1},  {f64, 2}, {bf16, 3}, {s8, 4},
                                                          {u8, 5},   {s16, 6},  {u16, 7}, {s32, 8},  {u32, 9},
                                                          {s64, 10}, {u64, 11}, {b1, 12}, {c64, 13}, {c128, 14}};
  if (typeid_idx.find(a) == typeid_idx.end()) {
    MS_EXCEPTION(TypeError) << "For Op[" << op_name << "], the type " << TypeIdToString(a) << "is invalid";
  }
  if (typeid_idx.find(b) == typeid_idx.end()) {
    MS_EXCEPTION(TypeError) << "For Op[" << op_name << "], the type " << TypeIdToString(b) << "is invalid";
  }
  if (a == b) {
    return a;
  }

  static const std::vector<std::vector<TypeId>> promote_types_lookup = {
    /*         f32  f16  f64  bf16  s8  u8  s16  u16  s32  u32  s64  u64  b1 c64  c128 */
    /* f32 */ {f32, f32, f64, f32, f32, f32, f32, ud, f32, ud, f32, ud, f32, c64, c128},
    /* f16 */ {f32, f16, f64, f32, f16, f16, f16, ud, f16, ud, f16, ud, f16, c64, c128},
    /* f64 */ {f64, f64, f64, f64, f64, f64, f64, ud, f64, ud, f64, ud, f64, c128, c128},
    /* bf16*/ {f32, f32, f64, bf16, bf16, bf16, bf16, ud, bf16, ud, bf16, ud, bf16, c64, c128},
    /* s8  */ {f32, f16, f64, bf16, s8, s16, s16, ud, s32, ud, s64, ud, s8, c64, c128},
    /* u8  */ {f32, f16, f64, bf16, s16, u8, s16, ud, s32, ud, s64, ud, u8, c64, c128},
    /* s16 */ {f32, f16, f64, bf16, s16, s16, s16, ud, s32, ud, s64, ud, s16, c64, c128},
    /* u16 */ {ud, ud, ud, ud, ud, ud, ud, u16, ud, ud, ud, ud, ud, ud, ud},
    /* s32 */ {f32, f16, f64, bf16, s32, s32, s32, ud, s32, ud, s64, ud, s32, c64, c128},
    /* u32 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, u32, ud, ud, ud, ud, ud},
    /* s64 */ {f32, f16, f64, bf16, s64, s64, s64, ud, s64, ud, s64, ud, s64, c64, c128},
    /* u64 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, u64, ud, ud, ud},
    /* b1  */ {f32, f16, f64, bf16, s8, u8, s16, ud, s32, ud, s64, ud, b1, c64, c128},
    /* c64 */ {c64, c64, c128, c64, c64, c64, c64, ud, c64, ud, c64, ud, c64, c64, c128},
    /* c128*/ {c128, c128, c128, c128, c128, c128, c128, ud, c128, ud, c128, ud, c128, c128, c128},
  };

  auto return_type_id = promote_types_lookup[typeid_idx[a]][typeid_idx[b]];
  if (return_type_id == ud) {
    MS_EXCEPTION(TypeError) << "For Op[" << op_name << "], the promote output type is invalid";
  }
  return return_type_id;
}

// Promote type for Tensor-Scalar binary ops in infer stage.
// The rule matches current Muls infer: bool < integral < floating < complex.
static inline TypeId PromoteTensorScalarType(TypeId tensor_type_id, TypeId scalar_type_id) {
  constexpr int typeLevelBool = 0;
  constexpr int typeLevelInt = 1;
  constexpr int typeLevelFloat = 2;
  constexpr int typeLevelComplex = 3;

  auto is_integral = [](TypeId t) {
    return t == kNumberTypeInt8 || t == kNumberTypeInt16 || t == kNumberTypeInt32 || t == kNumberTypeInt64 ||
           t == kNumberTypeUInt8 || t == kNumberTypeUInt16 || t == kNumberTypeUInt32 || t == kNumberTypeUInt64;
  };
  auto is_floating = [](TypeId t) {
    return t == kNumberTypeFloat16 || t == kNumberTypeFloat32 || t == kNumberTypeFloat64 || t == kNumberTypeBFloat16;
  };
  auto type_to_level = [&](TypeId t) {
    if (t == kNumberTypeBool) {
      return typeLevelBool;
    }
    if (is_integral(t)) {
      return typeLevelInt;
    }
    if (is_floating(t)) {
      return typeLevelFloat;
    }
    return typeLevelComplex;
  };

  return (type_to_level(tensor_type_id) < type_to_level(scalar_type_id)) ? scalar_type_id : tensor_type_id;
}

static inline TypePtr PromoteType(TypePtr a, TypePtr b, const std::string &op_name) {
  const auto f32 = kNumberTypeFloat32;
  const auto f16 = kNumberTypeFloat16;
  const auto f64 = kNumberTypeFloat64;
  const auto bf16 = kNumberTypeBFloat16;
  const auto s8 = kNumberTypeInt8;
  const auto u8 = kNumberTypeUInt8;
  const auto s16 = kNumberTypeInt16;
  const auto u16 = kNumberTypeUInt16;
  const auto s32 = kNumberTypeInt32;
  const auto u32 = kNumberTypeUInt32;
  const auto s64 = kNumberTypeInt64;
  const auto u64 = kNumberTypeUInt64;
  const auto b1 = kNumberTypeBool;
  const auto c64 = kNumberTypeComplex64;
  const auto c128 = kNumberTypeComplex128;

  static std::unordered_map<TypeId, TypePtr> typeid_typeptr = {
    {f32, kFloat32}, {f16, kFloat16}, {f64, kFloat64}, {bf16, kBFloat16}, {s8, kInt8},
    {u8, kUInt8},    {s16, kInt16},   {u16, kUInt16},  {s32, kInt32},     {u32, kUInt32},
    {s64, kInt64},   {u64, kUInt64},  {b1, kBool},     {c64, kComplex64}, {c128, kComplex128}};

  TypeId a_type_id;
  if (a->isa<TensorType>()) {
    auto a_tensor_type = a->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(a_tensor_type);
    auto a_element = a_tensor_type->element();
    MS_EXCEPTION_IF_NULL(a_element);
    a_type_id = a_element->type_id();
  } else {
    a_type_id = a->type_id();
  }

  TypeId b_type_id;
  if (b->isa<TensorType>()) {
    auto b_tensor_type = b->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(b_tensor_type);
    auto b_element = b_tensor_type->element();
    MS_EXCEPTION_IF_NULL(b_element);
    b_type_id = b_element->type_id();
  } else {
    b_type_id = b->type_id();
  }

  auto return_type_id = PromoteType(a_type_id, b_type_id, op_name);

  return typeid_typeptr[return_type_id];
}

void CheckTensorScalarRank(const PrimitivePtr &primitive, const AbstractBasePtr input_arg, const std::string &arg_name);
bool IsFloatType(TypePtr type);
bool IsIntegralType(TypePtr type, bool include_bool);
OPS_API std::vector<int64_t> CalBroadCastShapeV3(const std::vector<int64_t> &x_shape,
                                                 const std::vector<int64_t> &y_shape);
OPS_API int ConvertReductionForAclnn(Reduction reduction);
OPS_API const char *ConvertReductionStrForAclnn(int64_t reduction);
OPS_API size_t CalOutputSize(const std::vector<int64_t> &output_shape, const size_t &type_size);
OPS_API ValueTuplePtr ConvertShapeVectorToValueTuple(const ShapeVector &shape_vector);
OPS_API double GetDoubleValueFromScalar(const FP32ImmPtr &scalar);
OPS_API ScalarPtr FetchRealScalar(const ScalarPtr &scalar);
OPS_API bool IsEnableHostNode(const AnfNodePtr &node);

static inline void CheckRank(const InferInfoPtr &infer_info, size_t supported_rank, const std::string &op_name,
                             const std::string &input_name) {
  auto actual_rank = infer_info->GetShape().size();
  if (!infer_info->IsDynamicRank() && actual_rank != supported_rank) {
    MS_LOG(EXCEPTION) << op_name << ": The rank of " << input_name << " must be " << supported_rank << ", but get "
                      << actual_rank;
  }
}

static inline bool IsShapeKnown(const InferInfoPtr &infer_info, size_t index) {
  return !infer_info->IsDynamicRank() && infer_info->GetShape()[index] != mindspore::abstract::Shape::kShapeDimAny;
}

static inline void CheckType(const std::set<TypeId> &valid_types, const TypeId &arg_type, const std::string &op_name,
                             const std::string &arg_name) {
  if (valid_types.count(arg_type) == 0) {
    MS_EXCEPTION(TypeError) << "For op [" << op_name << "], the dtype of input " << arg_name << " is invalid.";
  }
}

static inline bool UseOptimizedOpImpl() {
  static const bool use_optimize_op_impl = common::GetEnv("MS_OPTIMIZE_OP_IMPL") == "1";
  return use_optimize_op_impl;
}

#define RETURN_IF_OPTIONAL_HAS_VALUE(opt) \
  do {                                    \
    if (opt.has_value()) {                \
      return opt.value();                 \
    }                                     \
  } while (0)

template <typename T>
inline T ComputeScales(const double &scale, const size_t &input_size, const size_t &output_size) {
  if (scale > SCALE_ZERO_THRESHOLD) {
    return static_cast<T>(SCALE_ONE / scale);
  } else if (output_size > 0) {
    return (static_cast<T>(input_size) / output_size);
  }
  return 0;
}

inline size_t NearestNeighborSourceIndex(const float &scale, const size_t &dst_index, const size_t &input_size) {
  size_t src_index = std::min(static_cast<size_t>(floorf(SizeToFloat(dst_index) * scale)), input_size - 1);
  return src_index;
}

inline size_t NearestIndex(const size_t &output_index, const size_t &input_size, const size_t &output_size,
                           const double &scales) {
  constexpr size_t kNumberTwo = 2;
  if (output_size == input_size) {
    // scale_factor = 1
    return output_index;
  } else if (output_size == kNumberTwo * input_size) {
    // scale_factor = 2, shift input index
    return output_index >> 1;
  } else {
    auto scale = ComputeScales<float>(scales, input_size, output_size);
    return NearestNeighborSourceIndex(scale, output_index, input_size);
  }
}

template <typename T>
inline T AreaPixelComputeScale(int64_t input_size, int64_t output_size, bool align_corners, double scale) {
  if (align_corners) {
    if (output_size > 1) {
      return static_cast<T>(input_size - 1) / (output_size - 1);
    } else {
      return static_cast<T>(0);
    }
  } else {
    return ComputeScales<T>(scale, input_size, output_size);
  }
}

OPS_API void *GetOpPluginHandle();
OPS_API bool IsOpPluginKernel(const std::string &op_name);
OPS_API const std::unordered_set<std::string> &GetAllOpPluginKernelNames();
OPS_API bool IsOpPluginComputeDependOp(const std::string &op_name);
OPS_API const std::unordered_set<std::string> &GetAllOpPluginComputeDependOps();
}  // namespace mindspore::ops
#endif  // MINDSPORE_CORE_OPS_OP_UTILS_H
