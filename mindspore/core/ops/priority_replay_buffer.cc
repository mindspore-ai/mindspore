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

#include "ops/priority_replay_buffer.h"
#include <string>
#include <algorithm>
#include <functional>
#include <memory>
#include <vector>
#include <numeric>

#include "ir/dtype/type.h"
#include "ir/value.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void PriorityReplayBufferCreate::set_capacity(const int64_t &capacity) {
  (void)this->AddAttr(kCapacity, api::MakeValue(capacity));
}

void PriorityReplayBufferCreate::set_alpha(const float &alpha) { (void)this->AddAttr(kAlpha, api::MakeValue(alpha)); }

void PriorityReplayBufferCreate::set_shapes(const std::vector<std::vector<int64_t>> &shapes) {
  (void)this->AddAttr(kShapes, api::MakeValue(shapes));
}

void PriorityReplayBufferCreate::set_types(const std::vector<TypePtr> &types) {
  auto res = std::dynamic_pointer_cast<PrimitiveC>(impl_);
  MS_EXCEPTION_IF_NULL(res);
  (void)res->AddAttr(kTypes, MakeValue(types));
}

void PriorityReplayBufferCreate::set_schema(const std::vector<int64_t> &schema) {
  (void)this->AddAttr(kSchema, api::MakeValue(schema));
}

void PriorityReplayBufferCreate::set_seed0(const int64_t &seed0) { (void)this->AddAttr(kSeed0, api::MakeValue(seed0)); }

void PriorityReplayBufferCreate::set_seed1(const int64_t &seed1) { (void)this->AddAttr(kSeed1, api::MakeValue(seed1)); }

int64_t PriorityReplayBufferCreate::get_capacity() const {
  auto value_ptr = GetAttr(kCapacity);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

float PriorityReplayBufferCreate::get_alpha() const {
  auto value_ptr = GetAttr(kAlpha);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<float>(value_ptr);
}

std::vector<std::vector<int64_t>> PriorityReplayBufferCreate::get_shapes() const {
  auto value_ptr = GetAttr(kShapes);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<std::vector<int64_t>>>(value_ptr);
}

std::vector<TypePtr> PriorityReplayBufferCreate::get_types() const {
  auto res = std::dynamic_pointer_cast<PrimitiveC>(impl_);
  MS_EXCEPTION_IF_NULL(res);
  return GetValue<std::vector<TypePtr>>(res->GetAttr(kTypes));
}

std::vector<int64_t> PriorityReplayBufferCreate::get_schema() const {
  auto value_ptr = GetAttr(kSchema);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

int64_t PriorityReplayBufferCreate::get_seed0() const {
  auto value_ptr = GetAttr(kSeed0);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

int64_t PriorityReplayBufferCreate::get_seed1() const {
  auto value_ptr = GetAttr(kSeed1);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void PriorityReplayBufferCreate::Init(const int64_t &capacity, const float &alpha,
                                      std::vector<std::vector<int64_t>> &shapes, const std::vector<TypePtr> &types,
                                      const int64_t &seed0, const int64_t &seed1) {
  auto op_name = this->name();
  if (shapes.size() != types.size()) {
    MS_LOG(EXCEPTION) << "For " << op_name
                      << " the rank of shapes and types should be the same, but got the rank of shapes is "
                      << shapes.size() << ", and types is " << types.size();
  }

  std::vector<int64_t> schema;
  for (size_t i = 0; i < shapes.size(); i++) {
    size_t type_size = GetTypeByte(types[i]);
    size_t tensor_size = std::accumulate(shapes[i].begin(), shapes[i].end(), type_size, std::multiplies<int64_t>());
    schema.push_back(tensor_size);
  }

  this->set_capacity(capacity);
  this->set_alpha(alpha);
  this->set_shapes(shapes);
  this->set_types(types);
  this->set_schema(schema);
  this->set_seed0(seed0);
  this->set_seed1(seed1);
}

void PriorityReplayBufferPush::set_handle(const int64_t &handle) {
  (void)this->AddAttr(kHandle, api::MakeValue(handle));
}

int64_t PriorityReplayBufferPush::get_handle() const {
  auto value_ptr = GetAttr(kHandle);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void PriorityReplayBufferPush::Init(const int64_t &handle) { this->set_handle(handle); }

void PriorityReplayBufferSample::set_handle(const int64_t &handle) {
  (void)this->AddAttr(kHandle, api::MakeValue(handle));
}

void PriorityReplayBufferSample::set_batch_size(const int64_t &batch_size) {
  (void)this->AddAttr(kBatchSize, api::MakeValue(batch_size));
}

void PriorityReplayBufferSample::set_shapes(const std::vector<std::vector<int64_t>> &shapes) {
  (void)this->AddAttr(kShapes, api::MakeValue(shapes));
}

void PriorityReplayBufferSample::set_types(const std::vector<TypePtr> &types) {
  auto res = std::dynamic_pointer_cast<PrimitiveC>(impl_);
  MS_EXCEPTION_IF_NULL(res);
  (void)res->AddAttr(kTypes, MakeValue(types));
}

void PriorityReplayBufferSample::set_schema(const std::vector<int64_t> &schema) {
  (void)this->AddAttr(kSchema, api::MakeValue(schema));
}

int64_t PriorityReplayBufferSample::get_handle() const {
  auto value_ptr = GetAttr(kHandle);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

int64_t PriorityReplayBufferSample::get_batch_size() const {
  auto value_ptr = GetAttr(kBatchSize);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

std::vector<std::vector<int64_t>> PriorityReplayBufferSample::get_shapes() const {
  auto value_ptr = GetAttr(kShapes);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<std::vector<int64_t>>>(value_ptr);
}

std::vector<TypePtr> PriorityReplayBufferSample::get_types() const {
  auto res = std::dynamic_pointer_cast<PrimitiveC>(impl_);
  MS_EXCEPTION_IF_NULL(res);
  return GetValue<std::vector<TypePtr>>(res->GetAttr(kTypes));
}

std::vector<int64_t> PriorityReplayBufferSample::get_schema() const {
  auto value_ptr = GetAttr(kSchema);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void PriorityReplayBufferSample::Init(const int64_t &handle, const int64_t batch_size,
                                      const std::vector<std::vector<int64_t>> &shapes,
                                      const std::vector<TypePtr> &types) {
  auto op_name = this->name();
  if (shapes.size() != types.size()) {
    MS_LOG(EXCEPTION) << "For " << op_name
                      << " the rank of shapes and types should be the same, but got the rank of shapes is "
                      << shapes.size() << ", and types is " << types.size();
  }

  std::vector<int64_t> schema;
  for (size_t i = 0; i < shapes.size(); i++) {
    size_t type_size = GetTypeByte(types[i]);
    size_t tensor_size = std::accumulate(shapes[i].begin(), shapes[i].end(), type_size, std::multiplies<int64_t>());
    schema.push_back(tensor_size);
  }

  this->set_handle(handle);
  this->set_batch_size(batch_size);
  this->set_shapes(shapes);
  this->set_types(types);
  this->set_schema(schema);
}

void PriorityReplayBufferUpdate::set_handle(const int64_t &handle) {
  (void)this->AddAttr(kHandle, api::MakeValue(handle));
}

int64_t PriorityReplayBufferUpdate::get_handle() const {
  auto value_ptr = GetAttr(kHandle);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void PriorityReplayBufferUpdate::Init(const int64_t &handle) { this->set_handle(handle); }

void PriorityReplayBufferDestroy::set_handle(const int64_t &handle) {
  (void)this->AddAttr(kHandle, api::MakeValue(handle));
}

int64_t PriorityReplayBufferDestroy::get_handle() const {
  auto value_ptr = GetAttr(kHandle);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void PriorityReplayBufferDestroy::Init(const int64_t &handle) { this->set_handle(handle); }

MIND_API_OPERATOR_IMPL(PriorityReplayBufferCreate, BaseOperator);
MIND_API_OPERATOR_IMPL(PriorityReplayBufferPush, BaseOperator);
MIND_API_OPERATOR_IMPL(PriorityReplayBufferSample, BaseOperator);
MIND_API_OPERATOR_IMPL(PriorityReplayBufferUpdate, BaseOperator);
MIND_API_OPERATOR_IMPL(PriorityReplayBufferDestroy, BaseOperator);

REGISTER_PRIMITIVE_C(kNamePriorityReplayBufferCreate, PriorityReplayBufferCreate);
REGISTER_PRIMITIVE_C(kNamePriorityReplayBufferPush, PriorityReplayBufferPush);
REGISTER_PRIMITIVE_C(kNamePriorityReplayBufferSample, PriorityReplayBufferSample);
REGISTER_PRIMITIVE_C(kNamePriorityReplayBufferUpdate, PriorityReplayBufferUpdate);
REGISTER_PRIMITIVE_C(kNamePriorityReplayBufferDestroy, PriorityReplayBufferDestroy);
}  // namespace ops
}  // namespace mindspore
