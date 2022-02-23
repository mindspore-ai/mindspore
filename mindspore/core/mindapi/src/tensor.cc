/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include <memory>
#include "mindapi/ir/tensor.h"
#include "mindapi/src/helper.h"
#include "ir/tensor.h"

namespace mindspore::api {
using TensorImpl = mindspore::tensor::Tensor;

MIND_API_BASE_IMPL(Tensor, TensorImpl, Value);

Tensor::Tensor(TypeId data_type, const ShapeVector &shape) : Value(std::make_shared<TensorImpl>(data_type, shape)) {}

Tensor::Tensor(TypeId data_type, const ShapeVector &shape, void *data, size_t data_len)
    : Value(std::make_shared<TensorImpl>(data_type, shape, data, data_len)) {}

const ShapeVector &Tensor::shape() const { return ToRef<TensorImpl>(impl_).shape(); }

void Tensor::set_shape(const ShapeVector &shape) { (void)ToRef<TensorImpl>(impl_).set_shape(shape); }

TypeId Tensor::data_type() const { return ToRef<TensorImpl>(impl_).data_type(); }

void Tensor::set_data_type(const TypeId data_type) { (void)ToRef<TensorImpl>(impl_).set_data_type(data_type); }

const void *Tensor::data() const { return ToRef<TensorImpl>(impl_).data_c(); }

void *Tensor::data() { return ToRef<TensorImpl>(impl_).data_c(); }

size_t Tensor::DataSize() const { return ToRef<TensorImpl>(impl_).DataSize(); }

size_t Tensor::Size() const { return ToRef<TensorImpl>(impl_).Size(); }
}  // namespace mindspore::api
