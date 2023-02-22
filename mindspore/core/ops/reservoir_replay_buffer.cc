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

#include "ops/reservoir_replay_buffer.h"

#include <string>
#include <functional>
#include <memory>
#include <vector>

#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void ReservoirReplayBufferCreate::set_capacity(const int64_t &capacity) {
  (void)this->AddAttr(kCapacity, api::MakeValue(capacity));
}

void ReservoirReplayBufferCreate::set_shapes(const std::vector<std::vector<int64_t>> &shapes) {
  (void)this->AddAttr(kShapes, api::MakeValue(shapes));
}

void ReservoirReplayBufferCreate::set_types(const std::vector<TypePtr> &types) {
  auto res = std::dynamic_pointer_cast<PrimitiveC>(impl_);
  MS_EXCEPTION_IF_NULL(res);
  (void)res->AddAttr(kTypes, MakeValue(types));
}

void ReservoirReplayBufferCreate::set_schema(const std::vector<int64_t> &schema) {
  (void)this->AddAttr(kSchema, api::MakeValue(schema));
}

void ReservoirReplayBufferCreate::set_seed0(const int64_t &seed0) {
  (void)this->AddAttr(kSeed0, api::MakeValue(seed0));
}

void ReservoirReplayBufferCreate::set_seed1(const int64_t &seed1) {
  (void)this->AddAttr(kSeed1, api::MakeValue(seed1));
}

int64_t ReservoirReplayBufferCreate::get_capacity() const {
  auto value_ptr = GetAttr(kCapacity);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

std::vector<std::vector<int64_t>> ReservoirReplayBufferCreate::get_shapes() const {
  auto value_ptr = GetAttr(kShapes);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<std::vector<int64_t>>>(value_ptr);
}

std::vector<TypePtr> ReservoirReplayBufferCreate::get_types() const {
  auto res = std::dynamic_pointer_cast<PrimitiveC>(impl_);
  MS_EXCEPTION_IF_NULL(res);
  return GetValue<std::vector<TypePtr>>(res->GetAttr(kTypes));
}

std::vector<int64_t> ReservoirReplayBufferCreate::get_schema() const {
  auto value_ptr = GetAttr(kSchema);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

int64_t ReservoirReplayBufferCreate::get_seed0() const {
  auto value_ptr = GetAttr(kSeed0);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

int64_t ReservoirReplayBufferCreate::get_seed1() const {
  auto value_ptr = GetAttr(kSeed1);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void ReservoirReplayBufferCreate::Init(const int64_t &capacity, std::vector<std::vector<int64_t>> &shapes,
                                       const std::vector<TypePtr> &types, const int64_t &seed0, const int64_t &seed1) {
  auto op_name = this->name();
  if (shapes.size() != types.size()) {
    MS_LOG(EXCEPTION) << "For " << op_name
                      << " the rank of shapes and types should be same, but got the rank of shapes is " << shapes.size()
                      << ", and types is " << types.size();
  }

  std::vector<int64_t> schema;
  for (size_t i = 0; i < shapes.size(); i++) {
    size_t type_size = GetTypeByte(types[i]);
    size_t tensor_size = std::accumulate(shapes[i].begin(), shapes[i].end(), type_size, std::multiplies<int64_t>());
    schema.push_back(tensor_size);
  }

  this->set_capacity(capacity);
  this->set_shapes(shapes);
  this->set_types(types);
  this->set_schema(schema);
  this->set_seed0(seed0);
  this->set_seed1(seed1);
}

void ReservoirReplayBufferPush::set_handle(const int64_t &handle) {
  (void)this->AddAttr(kHandle, api::MakeValue(handle));
}

int64_t ReservoirReplayBufferPush::get_handle() const {
  auto value_ptr = GetAttr(kHandle);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void ReservoirReplayBufferPush::Init(const int64_t &handle) { this->set_handle(handle); }

void ReservoirReplayBufferSample::set_handle(const int64_t &handle) {
  (void)this->AddAttr(kHandle, api::MakeValue(handle));
}

void ReservoirReplayBufferSample::set_batch_size(const int64_t &batch_size) {
  (void)this->AddAttr(kBatchSize, api::MakeValue(batch_size));
}

void ReservoirReplayBufferSample::set_shapes(const std::vector<std::vector<int64_t>> &shapes) {
  (void)this->AddAttr(kShapes, api::MakeValue(shapes));
}

void ReservoirReplayBufferSample::set_types(const std::vector<TypePtr> &types) {
  auto res = std::dynamic_pointer_cast<PrimitiveC>(impl_);
  MS_EXCEPTION_IF_NULL(res);
  (void)res->AddAttr(kTypes, MakeValue(types));
}

void ReservoirReplayBufferSample::set_schema(const std::vector<int64_t> &schema) {
  (void)this->AddAttr(kSchema, api::MakeValue(schema));
}

int64_t ReservoirReplayBufferSample::get_handle() const {
  auto value_ptr = GetAttr(kHandle);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

int64_t ReservoirReplayBufferSample::get_batch_size() const {
  auto value_ptr = GetAttr(kBatchSize);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

std::vector<std::vector<int64_t>> ReservoirReplayBufferSample::get_shapes() const {
  auto value_ptr = GetAttr(kShapes);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<std::vector<int64_t>>>(value_ptr);
}

std::vector<TypePtr> ReservoirReplayBufferSample::get_types() const {
  auto res = std::dynamic_pointer_cast<PrimitiveC>(impl_);
  MS_EXCEPTION_IF_NULL(res);
  return GetValue<std::vector<TypePtr>>(res->GetAttr(kTypes));
}

std::vector<int64_t> ReservoirReplayBufferSample::get_schema() const {
  auto value_ptr = GetAttr(kSchema);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void ReservoirReplayBufferSample::Init(const int64_t &handle, const int64_t &batch_size,
                                       const std::vector<std::vector<int64_t>> &shapes,
                                       const std::vector<TypePtr> &types) {
  auto op_name = this->name();
  if (shapes.size() != types.size()) {
    MS_LOG(EXCEPTION) << "For " << op_name
                      << " the rank of shapes and types should be same, but got the rank of shapes is " << shapes.size()
                      << ", and rank of types is " << types.size();
  }

  std::vector<int64_t> schema;
  for (size_t i = 0; i < shapes.size(); i++) {
    size_t type_size = GetTypeByte(types[i]);
    size_t tensor_size = std::accumulate(shapes[i].begin(), shapes[i].end(), type_size, std::multiplies<int64_t>());
    schema.push_back(tensor_size);
  }
  this->set_schema(schema);
  this->set_handle(handle);
  this->set_batch_size(batch_size);
  this->set_shapes(shapes);
  this->set_types(types);
}

void ReservoirReplayBufferDestroy::set_handle(const int64_t &handle) {
  (void)this->AddAttr(kHandle, api::MakeValue(handle));
}

int64_t ReservoirReplayBufferDestroy::get_handle() const {
  auto value_ptr = GetAttr(kHandle);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void ReservoirReplayBufferDestroy::Init(const int64_t &handle) { this->set_handle(handle); }

MIND_API_OPERATOR_IMPL(ReservoirReplayBufferCreate, BaseOperator);
MIND_API_OPERATOR_IMPL(ReservoirReplayBufferPush, BaseOperator);
MIND_API_OPERATOR_IMPL(ReservoirReplayBufferSample, BaseOperator);
MIND_API_OPERATOR_IMPL(ReservoirReplayBufferDestroy, BaseOperator);

namespace {
BaseShapePtr CommonInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const ShapeVector &shape = {1};
  BaseShapePtr out_shape = std::make_shared<abstract::Shape>(shape);
  return out_shape;
}

AbstractBasePtr SampleInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &) {
  MS_EXCEPTION_IF_NULL(primitive);

  const std::string &prim_name = primitive->name();
  auto types = GetValue<std::vector<TypePtr>>(primitive->GetAttr("dtypes"));
  auto shapes = GetValue<std::vector<std::vector<int64_t>>>(primitive->GetAttr("shapes"));
  if (types.size() != shapes.size()) {
    MS_LOG(EXCEPTION) << "For Primitive[" << prim_name << "], the types and shapes rank should be same.";
  }

  auto batch_size = GetValue<int64_t>(primitive->GetAttr("batch_size"));
  AbstractBasePtrList output;
  for (size_t i = 0; i < shapes.size(); ++i) {
    auto shape = shapes[i];
    (void)shape.emplace(shape.begin(), batch_size);
    auto element = std::make_shared<abstract::AbstractScalar>(kAnyValue, types[i]);
    auto tensor = std::make_shared<abstract::AbstractTensor>(element, std::make_shared<abstract::Shape>(shape));
    (void)output.emplace_back(tensor);
  }

  return std::make_shared<abstract::AbstractTuple>(output);
}
}  // namespace

class MIND_API CreateInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return CommonInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);

    const std::string &prim_name = primitive->name();
    if (input_args.size() != 0) {
      MS_LOG(EXCEPTION) << "For Primitive[" << prim_name << "], the input should be empty.";
    }
    return kInt64;
  }
};

class MIND_API CommonInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return CommonInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    return kInt64;
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ReservoirReplayBufferCreate, prim::kPrimReservoirReplayBufferCreate, CreateInfer,
                                 false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ReservoirReplayBufferPush, prim::kPrimReservoirReplayBufferPush, CommonInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ReservoirReplayBufferDestroy, prim::kPrimReservoirReplayBufferDestroy, CommonInfer,
                                 false);

REGISTER_PRIMITIVE_EVAL_IMPL(ReservoirReplayBufferSample, prim::kPrimReservoirReplayBufferSample, SampleInfer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
