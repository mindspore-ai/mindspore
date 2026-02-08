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

#include "include/pynative/utils/pyboost/custom/tensor.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "ir/tensor.h"
#include "ir/format_utils.h"
#include "ir/tensor_new.h"
#include "include/utils/tensor_utils.h"
#include "include/pynative/utils/pynative_utils.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"

namespace ms {
namespace {
inline mindspore::Int64ImmPtr MakeI64Value(int64_t v) { return std::make_shared<mindspore::Int64Imm>(v); }
inline mindspore::ValueTuplePtr MakeI64Tuple(const std::vector<int64_t> &v) {
  return mindspore::MakeValue(v)->cast<mindspore::ValueTuplePtr>();
}
inline std::vector<Tensor> ToTensorList(const std::vector<mindspore::tensor::TensorPtr> &tensors) {
  std::vector<Tensor> outs;
  outs.reserve(tensors.size());
  (void)std::transform(tensors.begin(), tensors.end(), std::back_inserter(outs), [](auto &t) { return Tensor(t); });
  return outs;
}

mindspore::tensor::TensorPtr StubNodeToTensor(const mindspore::ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  if (mindspore::utils::isa<mindspore::stub::StubNode>(v)) {
    auto stub = mindspore::utils::cast<mindspore::stub::StubNodePtr>(v);
    return stub->WaitValue()->cast<mindspore::tensor::TensorPtr>();
  }
  if (v->isa<mindspore::tensor::Tensor>()) {
    return v->cast<mindspore::tensor::TensorPtr>();
  }
  MS_LOG(EXCEPTION) << "It should be stub tensor, but got " << v->ToString();
}
}  // namespace

Tensor::RealTensorHolder::RealTensorHolder(const mindspore::ValuePtr &value)
    : value_(value), tensor_(value->cast<mindspore::tensor::TensorPtr>()) {}

Tensor::Tensor(TypeId type_id, const ShapeVector &shape)
    : Tensor(mindspore::tensor::from_spec(type_id, shape, mindspore::device::DeviceType::kNone)) {}

Tensor::Tensor(const mindspore::ValuePtr &value) {
  if (value != nullptr) {
    _tensor_holder_ = std::make_shared<RealTensorHolder>(value);
    if (_tensor_holder_->tensor_ != nullptr) {
      _tensor_holder_->tensor_->set_need_pipeline_sync(true);
    }
  }
}

void *Tensor::GetDataPtr() const {
  auto t = tensor();
  MS_EXCEPTION_IF_NULL(t);
  auto device_address = t->device_address();
  MS_EXCEPTION_IF_NULL(device_address);
  auto offset = t->DataItemSize() * t->storage_offset();
  return static_cast<void *>(static_cast<int8_t *>(t->device_address()->GetMutablePtr()) + offset);
}

TypeId Tensor::data_type() const {
  MS_EXCEPTION_IF_NULL(tensor());
  return tensor()->data_type();
}

std::string Tensor::format() const {
  auto t = tensor();
  MS_EXCEPTION_IF_NULL(t);
  if (t->device_address() == nullptr) {
    return "DefaultFormat";
  }
  return mindspore::kernel::GetFormatFromEnumToStr(t->format());
}

void Tensor::set_format(const std::string &format) const {
  auto t = tensor();
  MS_EXCEPTION_IF_NULL(t);
  MS_EXCEPTION_IF_CHECK_FAIL(t->device_address(), "The device address of the Tensor is nullptr");
  auto device_sync = t->device_address();
  auto device_address = std::dynamic_pointer_cast<mindspore::device::DeviceAddress>(device_sync);
  MS_EXCEPTION_IF_NULL(device_address);

  t->set_format(mindspore::kernel::GetFormatFromStrToEnum(format));
}

const ShapeVector &Tensor::shape() const {
  MS_EXCEPTION_IF_NULL(tensor());
  return tensor()->shape();
}

size_t Tensor::numel() const {
  auto &s = shape();
  return static_cast<size_t>(std::accumulate(s.begin(), s.end(), 1LL, std::multiplies<int64_t>()));
}

std::vector<int64_t> Tensor::stride() const {
  MS_EXCEPTION_IF_NULL(tensor());
  return tensor()->stride();
}

int64_t Tensor::storage_offset() const {
  MS_EXCEPTION_IF_NULL(tensor());
  return tensor()->storage_offset();
}

bool Tensor::is_contiguous() const {
  MS_EXCEPTION_IF_NULL(tensor());
  return tensor()->is_contiguous();
}

void Tensor::SetNeedContiguous(bool flag) const {
  MS_EXCEPTION_IF_CHECK_FAIL(is_defined(), "The Tensor is not defined");
  _tensor_holder_->need_contiguous_ = flag;
}

bool Tensor::need_contiguous() const {
  MS_EXCEPTION_IF_CHECK_FAIL(is_defined(), "The Tensor is not defined");
  return _tensor_holder_->need_contiguous_;
}

const mindspore::ValuePtr &Tensor::stub_node() const {
  MS_EXCEPTION_IF_CHECK_FAIL(is_defined(), "The Tensor is not defined");
  return _tensor_holder_->value_;
}

const mindspore::tensor::TensorPtr &Tensor::tensor() const {
  MS_EXCEPTION_IF_CHECK_FAIL(is_defined(), "The Tensor is not defined");
  return _tensor_holder_->tensor_;
}

void Tensor::ConvertStubNodeToTensor() const {
  if (stub_node() != nullptr) {
    _tensor_holder_->tensor_ = StubNodeToTensor(stub_node());
    // release the stub node object.
    _tensor_holder_->value_ = nullptr;
  }
}

void Tensor::AssignTensor(const Tensor &src) const { tensor()->AssignValue(*(src.tensor())); }

Tensor Tensor::cast(TypeId dtype) const {
  return Tensor(mindspore::kernel::pyboost::cast(tensor(), MakeI64Value(static_cast<int64_t>(dtype))));
}

std::vector<Tensor> Tensor::chunk(int64_t chunks, int64_t dim) const {
  return ToTensorList(mindspore::kernel::pyboost::chunk(tensor(), chunks, dim));
}

Tensor Tensor::contiguous() const { return Tensor(mindspore::kernel::pyboost::contiguous(tensor())); }

Tensor Tensor::flatten(int64_t start_dim, int64_t end_dim) const {
  return Tensor(mindspore::kernel::pyboost::flatten_ext(tensor(), start_dim, end_dim));
}

Tensor Tensor::index_select(int64_t dim, const Tensor &index) const {
  return Tensor(mindspore::kernel::pyboost::index_select(tensor(), MakeI64Value(dim), index.tensor()));
}

Tensor Tensor::reshape(const std::vector<int64_t> &shape) const {
  return Tensor(mindspore::kernel::pyboost::reshape(tensor(), shape));
}

Tensor Tensor::repeat(const std::vector<int64_t> &repeats) const {
  return Tensor(mindspore::kernel::pyboost::repeat(tensor(), MakeI64Tuple(repeats)));
}

Tensor Tensor::repeat_interleave(const Tensor &repeats, const std::optional<int64_t> &dim,
                                 const std::optional<int64_t> &output_size) const {
  std::optional<mindspore::Int64ImmPtr> dim_opt = std::nullopt;
  std::optional<mindspore::Int64ImmPtr> output_size_opt = std::nullopt;
  if (dim.has_value()) {
    dim_opt = MakeI64Value(dim.value());
  }
  if (output_size.has_value()) {
    output_size_opt = MakeI64Value(output_size.value());
  }
  return Tensor(
    mindspore::kernel::pyboost::repeat_interleave_tensor(tensor(), repeats.tensor(), dim_opt, output_size_opt));
}

Tensor Tensor::repeat_interleave(int64_t repeats, const std::optional<int64_t> &dim,
                                 const std::optional<int64_t> &output_size) const {
  std::optional<mindspore::Int64ImmPtr> dim_opt = std::nullopt;
  std::optional<mindspore::Int64ImmPtr> output_size_opt = std::nullopt;
  if (dim.has_value()) {
    dim_opt = MakeI64Value(dim.value());
  }
  if (output_size.has_value()) {
    output_size_opt = MakeI64Value(output_size.value());
  }
  return Tensor(
    mindspore::kernel::pyboost::repeat_interleave_int(tensor(), MakeI64Value(repeats), dim_opt, output_size_opt));
}
}  // namespace ms

namespace pybind11 {
namespace detail {
bool type_caster<ms::Tensor>::load(handle src, bool) {
  if (mindspore::tensor::IsTensorPy(src)) {
    auto v = mindspore::tensor::ConvertToValue(src);
    if (v->isa<mindspore::tensor::Tensor>()) {
      v->cast<mindspore::tensor::TensorPtr>()->set_need_pipeline_sync(true);
    }
    value = ms::Tensor(v);
    return true;
  }
  // the value is initialized as an undefined Tensor for None input.
  return src.is_none();
}

handle type_caster<ms::Tensor>::cast(const ms::Tensor &src, return_value_policy, handle) {
  MS_EXCEPTION_IF_NULL(src.tensor());
  return handle(mindspore::tensor::Wrap(src.tensor()));
}
}  // namespace detail
}  // namespace pybind11
