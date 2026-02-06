/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/dvm/pyboost_impl/lazy_fusion_op.h"
#include <memory>
#include <vector>
#include <string>
#include <tuple>
#include <utility>
#include <algorithm>
#include "ir/tensor_new.h"
#include "infer/ops_func_impl/tile.h"
#include "kernel/ascend/dvm/pyboost_impl/lazy_fusion_kernel.h"
#include "kernel/ascend/dvm/pyboost_impl/lazy_fusion_flags.h"
#include "include/runtime/pipeline/pipeline.h"
#include "view/view_strides_calculator.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_e.h"
#include "primitive/auto_generate/gen_ops_primitive_f.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_n.h"
#include "primitive/auto_generate/gen_ops_primitive_p.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "include/pynative/utils/pyboost/auto_generate/copy.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

#define CALL_START(op_name, k)                                                        \
  do {                                                                                \
    MS_LOG(INFO) << op_name << " call start, kernel id is " << (k->id()) << " " << k; \
  } while (0)

#define CALL_END(op_name, k)                                                        \
  do {                                                                              \
    MS_LOG(INFO) << op_name << " call end, kernel id is " << (k->id()) << " " << k; \
  } while (0)

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
constexpr unsigned int OpLevel_0 = 0;
constexpr unsigned int OpLevel_1 = 1;

bool EnableFuse(const std::string &op, unsigned int op_level) {
  const auto &disable_ops = LazyFusionFlags::GetInstance().disable_ops;
  const auto &enable_ops = LazyFusionFlags::GetInstance().enable_ops;
  const auto &enable_ops_only = LazyFusionFlags::GetInstance().enable_ops_only;
  if (!enable_ops_only.empty()) {
    return std::find(enable_ops_only.begin(), enable_ops_only.end(), op) != enable_ops_only.end();
  }
  if (op_level > OpLevel_0 && std::find(enable_ops.begin(), enable_ops.end(), op) == enable_ops.end()) {
    // for ops that not fused by default
    return false;
  }
  return std::find(disable_ops.begin(), disable_ops.end(), op) == disable_ops.end();
}

bool IsFloatType(const TypeId &type) {
  switch (type) {
    case kNumberTypeFloat16:
    case kNumberTypeFloat32:
    case kNumberTypeBFloat16:
      return true;
    default:
      return false;
  }
}

bool NeedSync() {
  static auto need_sync = runtime::OpExecutor::NeedSync();
  return (need_sync && !runtime::OpExecutor::GetInstance().async_for_graph());
}

bool IsFloatIntType(const TypeId &type) { return IsFloatType(type) || type == kNumberTypeInt32; }

bool IsSupportType(const TypeId &type) { return IsFloatIntType(type) || type == kNumberTypeBool; }

bool InputCheck(const TensorPtr &x, const std::function<bool(const TypeId &type)> &type_check = IsFloatType) {
  return !NeedSync() && x->is_contiguous() && type_check(x->data_type());
}

bool CheckShapeBroadcast(const ShapeVector &shape1, const ShapeVector &shape2) {
  auto check_func = [](const auto &shape_s, const auto &shape_l) {
    auto diff = shape_l.size() - shape_s.size();
    for (size_t i = 0; i < shape_s.size(); ++i) {
      auto sh1 = shape_s[i];
      auto sh2 = shape_l[i + diff];
      if (sh1 != sh2 && sh1 != 1 && sh2 != 1) {
        return false;
      }
    }
    return true;
  };
  return shape1.size() < shape2.size() ? check_func(shape1, shape2) : check_func(shape2, shape1);
}

bool BinaryInputCheck(const TensorPtr &input_tensor, const TensorPtr &other_tensor,
                      const std::function<bool(const TypeId &type)> &type_check = IsFloatType) {
  return !NeedSync() && input_tensor->is_contiguous() && other_tensor->is_contiguous() &&
         type_check(input_tensor->data_type()) && other_tensor->data_type() == input_tensor->data_type() &&
         CheckShapeBroadcast(input_tensor->shape(), other_tensor->shape());
}

inline bool BinaryExtCheck(const TensorPtr &input_tensor, const TensorPtr &other_tensor, bool inplace) {
  if (inplace) {
    // inplace op support different data type
    return InputCheck(input_tensor) && InputCheck(other_tensor) &&
           CheckShapeBroadcast(input_tensor->shape(), other_tensor->shape());
  }
  // non inplace op should have same data type
  return BinaryInputCheck(input_tensor, other_tensor);
}

bool IsScalar(const TensorPtr &x) {
  return x->shape().empty() && (x->device_address()->GetDeviceType() == device::DeviceType::kCPU) &&
         (x->DataSize() == 1);
}

template <typename T>
std::pair<bool, T> GetScalarValue(const ScalarPtr &s) {
  MS_EXCEPTION_IF_NULL(s);
  if (s->isa<Int64Imm>()) {
    return std::make_pair(true, static_cast<T>(GetValue<int64_t>(s)));
  } else if (s->isa<FP32Imm>()) {
    return std::make_pair(true, static_cast<T>(GetValue<float>(s)));
  } else if (s->isa<Int32Imm>()) {
    return std::make_pair(true, static_cast<T>(GetValue<int32_t>(s)));
  }
  return std::make_pair(false, T(0));
}

ShapeVector GetReduceDim(const std::optional<ValueTuplePtr> &dim, size_t rank) {
  ShapeVector dim_value;
  dim_value.reserve(rank);
  if (dim.has_value()) {
    auto dim_data = dim.value()->value();
    (void)std::transform(dim_data.begin(), dim_data.end(), std::back_inserter(dim_value),
                         [](const ValuePtr &v) { return v->cast<Int64ImmPtr>()->value(); });
  } else {
    for (int64_t i = 0; i < static_cast<int64_t>(rank); ++i) {
      dim_value.push_back(i);
    }
  }
  return dim_value;
}

template <typename F, typename... Args>
void DvmCall(const std::string &op_name, OpRunner *op, const F &func, const Args &...inputs) {
  size_t stream = op->stream_id();
  const DeviceContext *context = op->device_context();
  PyBoostUtils::PrepareOpInputs(context, stream, inputs...);
  // Prevent the indirect invocation of FlushLazyFusion() within the PrepareOpInputs() function
  // from releasing the current LazyFusionAscend pointer.
  auto k = g_lazy_fusion_manager.Get(context, stream);
  CALL_START(op_name, k);
  auto tensor = func(k);
  tensor->set_need_pipeline_sync(true);
  auto &outputs = const_cast<std::vector<tensor::TensorPtr> &>(op->outputs());
  outputs.emplace_back(std::move(tensor));
  op->CreateOutputSimpleInfo();
  CALL_END(op_name, k);
}

template <typename T>
T TensorToScalar(const tensor::TensorPtr &tensor) {
  switch (tensor->data_type()) {
    case kNumberTypeBool:
      return static_cast<T>(static_cast<bool *>(tensor->data_c())[0]);
    case kNumberTypeFloat16:
      return static_cast<T>(static_cast<float16 *>(tensor->data_c())[0]);
    case kNumberTypeFloat32:
      return static_cast<T>(static_cast<float *>(tensor->data_c())[0]);
    case kNumberTypeInt32:
      return static_cast<T>(static_cast<int32_t *>(tensor->data_c())[0]);
    case kNumberTypeBFloat16:
      return static_cast<T>(static_cast<bfloat16 *>(tensor->data_c())[0]);
    default:
      return static_cast<T>(0);
  }
  return static_cast<T>(0);
}

void BinaryDvmCall(const std::string &op_name, OpRunner *op, dvm::BinaryOpType op_type, const TensorPtr &input_tensor,
                   const TensorPtr &other_tensor, const TypeId dst_type) {
  size_t stream = op->stream_id();
  const DeviceContext *context = op->device_context();
  LazyFusionKernelAscend *k;
  auto type_id = input_tensor->data_type();
  dvm::NDObject *obj = nullptr;
  if (IsScalar(input_tensor)) {
    PyBoostUtils::PrepareOpInputs(context, stream, other_tensor);
    k = g_lazy_fusion_manager.Get(context, stream);
    CALL_START(op_name, k);
    if (type_id == kNumberTypeInt32) {
      auto scalar_value = TensorToScalar<int32_t>(input_tensor);
      MS_LOG(INFO) << op_name << " input_tensor is scalar, value: " << scalar_value << ", kernel id is " << k->id()
                   << " " << k;
      obj = k->Binary(op_type, scalar_value, k->Input(other_tensor));
    } else {
      auto scalar_value = TensorToScalar<float>(input_tensor);
      MS_LOG(INFO) << op_name << " input_tensor is scalar, value: " << scalar_value << ", kernel id is " << k->id()
                   << " " << k;
      obj = k->Binary(op_type, scalar_value, k->Input(other_tensor));
    }
  } else if (IsScalar(other_tensor)) {
    PyBoostUtils::PrepareOpInputs(context, stream, input_tensor);
    k = g_lazy_fusion_manager.Get(context, stream);
    CALL_START(op_name, k);
    if (type_id == kNumberTypeInt32) {
      auto scalar_value = TensorToScalar<int32_t>(other_tensor);
      MS_LOG(INFO) << op_name << " other_tensor is scalar, value: " << scalar_value << ", kernel id is " << k->id()
                   << " " << k;
      obj = k->Binary(op_type, k->Input(input_tensor), scalar_value);
    } else {
      auto scalar_value = TensorToScalar<float>(other_tensor);
      MS_LOG(INFO) << op_name << " other_tensor is scalar, value: " << scalar_value << ", kernel id is " << k->id()
                   << " " << k;
      obj = k->Binary(op_type, k->Input(input_tensor), scalar_value);
    }
  } else {
    PyBoostUtils::PrepareOpInputs(context, stream, input_tensor, other_tensor);
    k = g_lazy_fusion_manager.Get(context, stream);
    CALL_START(op_name, k);
    obj = k->Binary(op_type, k->Input(input_tensor), k->Input(other_tensor));
  }
  auto tensor = k->Output(obj, dst_type, k->GetShape(obj));

  tensor->set_need_pipeline_sync(true);
  auto &outputs = const_cast<std::vector<tensor::TensorPtr> &>(op->outputs());
  outputs.emplace_back(std::move(tensor));
  op->CreateOutputSimpleInfo();
  CALL_END(op_name, k);
}

bool AddExtDvmCall(const std::string &op_name, OpRunner *op, const TensorPtr &input_tensor,
                   const TensorPtr &other_tensor, const ScalarPtr &alpha, bool inplace) {
  if (!BinaryExtCheck(input_tensor, other_tensor, inplace)) {
    return false;
  }
  auto input_type = input_tensor->data_type();
  auto other_type = other_tensor->data_type();
  bool all_bf16 = false;
  float scalar = 1.0f;
  if (alpha->isa<Int64Imm>()) {
    scalar = static_cast<float>(GetValue<int64_t>(alpha));
  } else if (alpha->isa<FP32Imm>()) {
    scalar = GetValue<float>(alpha);
    if (input_type == kNumberTypeBFloat16 && other_type == kNumberTypeBFloat16) {
      scalar = static_cast<float>(static_cast<bfloat16>(scalar));
      all_bf16 = true;
    }
  } else {
    return false;
  }
  DvmCall(
    op_name, op,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      auto input_obj = k->Input(input_tensor);
      auto other_obj = k->Input(other_tensor);
      auto input_dtype = k->GetDType(input_obj);
      auto other_dtype = k->GetDType(other_obj);
      auto compute_type = other_dtype == dvm::DType::kFloat32 ? other_dtype : input_dtype;
      input_obj = k->Cast(input_obj, compute_type);
      other_obj = k->Cast(other_obj, compute_type);
      if (scalar != 1.0f) {
        other_obj = k->Binary(dvm::BinaryOpType::kMul, other_obj, scalar);
        if (all_bf16) {
          other_obj = k->Cast(k->Cast(other_obj, dvm::DType::kBFloat16), dvm::DType::kFloat32);
        }
      }
      auto out_obj = k->Binary(dvm::BinaryOpType::kAdd, input_obj, other_obj);
      if (inplace) {
        // update
        k->Output(input_tensor, out_obj, true);
        return input_tensor;
      }
      return k->Output(out_obj, input_type, k->GetShape(out_obj));
    },
    input_tensor, other_tensor);
  return true;
}

bool SubExtDvmCall(const std::string &op_name, OpRunner *op, const TensorPtr &input_tensor,
                   const TensorPtr &other_tensor, const ScalarPtr &alpha, bool inplace) {
  if (!BinaryExtCheck(input_tensor, other_tensor, inplace)) {
    return false;
  }
  dvm::DType compute_type = dvm::DType::kTypeEnd;
  float scalar = 1.0f;
  if (alpha->isa<Int64Imm>()) {
    scalar = static_cast<float>(GetValue<int64_t>(alpha));
  } else if (alpha->isa<FP32Imm>()) {
    compute_type = dvm::DType::kFloat32;
    scalar = GetValue<float>(alpha);
  } else {
    return false;
  }
  DvmCall(
    op_name, op,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      auto input_obj = k->Input(input_tensor);
      auto other_obj = k->Input(other_tensor);
      auto input_dtype = k->GetDType(input_obj);
      auto other_dtype = k->GetDType(other_obj);
      if (compute_type == dvm::DType::kTypeEnd) {
        compute_type = other_dtype == dvm::DType::kFloat32 ? other_dtype : input_dtype;
      }
      input_obj = k->Cast(input_obj, compute_type);
      other_obj = k->Cast(other_obj, compute_type);
      if (scalar != 1.0f) {
        other_obj = k->Binary(dvm::BinaryOpType::kMul, other_obj, scalar);
      }
      auto out_obj = k->Binary(dvm::BinaryOpType::kSub, input_obj, other_obj);
      if (inplace) {
        // update
        k->Output(input_tensor, out_obj, true);
        return input_tensor;
      }
      return k->Output(out_obj, input_tensor->data_type(), k->GetShape(out_obj));
    },
    input_tensor, other_tensor);
  return true;
}

bool SameTensor(const TensorPtr &tensor1, const TensorPtr &tensor2) {
  MS_EXCEPTION_IF_NULL(tensor1);
  MS_EXCEPTION_IF_NULL(tensor2);
  auto addr1 = tensor1->device_address();
  auto addr2 = tensor2->device_address();
  auto ptr1 = addr1 == nullptr ? nullptr : addr1->GetMutablePtr();
  auto ptr2 = addr2 == nullptr ? nullptr : addr2->GetMutablePtr();
  if (ptr1 == nullptr || ptr2 == nullptr || ptr2 != ptr1) {
    return false;
  }
  if (tensor1->data_type() != tensor2->data_type()) {
    return false;
  }
  if (tensor1->shape() != tensor2->shape()) {
    return false;
  }
  if (!tensor1->is_contiguous() || !tensor2->is_contiguous()) {
    return false;
  }
  auto s1 = tensor1->storage_info();
  auto s2 = tensor2->storage_info();
  if (s1 != nullptr && s2 != nullptr) {
    return s1->storage_offset == s2->storage_offset && s1->shape == s2->shape && s1->strides == s2->strides;
  }
  return true;
}

tensor::TensorPtr ToContiguous(const TensorPtr &tensor, size_t stream_id) {
  if (tensor->is_contiguous()) {
    return tensor;
  }
  auto copy_op = CREATE_PYBOOST_OP(Copy, device::DeviceType::kAscend);
  copy_op->set_stream_id(stream_id);
  return copy_op->Call(tensor);
}

struct MatMulAdapter {
  TensorPtr x_tensor;
  TensorPtr y_tensor;
  bool trans_a;
  bool trans_b;
  ShapeVector x_shape;
  ShapeVector y_shape;

  MatMulAdapter(const TensorPtr &x, const TensorPtr &y, bool ta, bool tb)
      : x_tensor(x), y_tensor(y), trans_a(ta), trans_b(tb) {
    x_shape = x_tensor->shape();
    y_shape = y_tensor->shape();
  }

  bool Check(OpRunner *op) {
    auto data_type = x_tensor->data_type();
    if (NeedSync() || y_tensor->data_type() != data_type ||
        (data_type != kNumberTypeFloat16 && data_type != kNumberTypeBFloat16)) {
      return false;
    }
    if (x_shape.size() < kDim2 || x_shape.size() > kDim4 || y_shape.size() < kDim2 || y_shape.size() > kDim4) {
      return false;
    }
    bool check_x = (x_tensor->is_contiguous() || CheckTensorTranspose(x_tensor, &trans_a, &x_shape));
    bool check_y = (y_tensor->is_contiguous() || CheckTensorTranspose(y_tensor, &trans_b, &y_shape));
    if (!CheckMatMulShape()) {
      return false;
    }
    if (!check_x) {
      x_tensor = ToContiguous(x_tensor, op->stream_id());
    }
    if (!check_y) {
      y_tensor = ToContiguous(y_tensor, op->stream_id());
    }
    return true;
  }

 private:
  bool CheckMatMulShape() {
    // The value of max dimension size is due to two constraints:
    // 1. current implementation does not support stride between two rows greater than UINT16_MAX
    // 2. even if the row stride in the original input shape does not exceed UINT16_MAX, after address
    // alignment, it can potentially exceed UINT16_MAX.
    // The current value of kMaxDimSize guarantees that after address alignment, row stride is within
    // a reasonable range.
    constexpr int64_t kMaxDimSize = UINT16_MAX - UINT8_MAX;
    return x_shape.back() <= kMaxDimSize && y_shape.back() <= kMaxDimSize;
  }

  bool CheckTensorTranspose(const TensorPtr &tensor, bool *transpose, ShapeVector *shape) {
    auto storage_info = tensor->storage_info();
    MS_EXCEPTION_IF_NULL(storage_info);
    const auto &cur_shape = storage_info->shape;
    const auto &cur_strides = storage_info->strides;
    int64_t dim1 = SizeToLong(cur_shape.size()) - 1;
    int64_t dim2 = dim1 - 1;
    if (cur_strides[dim2] == 1 && cur_strides[dim1] == cur_shape[dim2]) {
      int64_t tmpNxD = cur_shape[dim1] * cur_shape[dim2];
      for (int64_t batchDim = dim1 - 2; batchDim >= 0; batchDim--) {
        if (cur_strides[batchDim] != tmpNxD) {
          return false;
        }
        tmpNxD *= cur_shape[batchDim];
      }
      (*transpose) ^= true;
      std::swap((*shape)[dim1], (*shape)[dim2]);
      return true;
    }
    return false;
  }
};

template <typename... Args>
void DumpOp(const std::string &op_name, OpRunner *op, const Args &...inputs) {
  if (LazyFusionFlags::GetInstance().dump_as_text) {
    auto k = g_lazy_fusion_manager.Get(op->device_context(), op->stream_id());
    k->DumpOp(op_name, inputs...);
  }
}
}  // namespace

tensor::TensorPtr ConcatAscendDvm::Call(const ValueTuplePtr &tensors_tensor_list, const Int64ImmPtr &axis) {
  // Concat elimination, limit: 1. axis is 0 2. all inputs are view op
  const auto &lst = tensors_tensor_list->value();
  if (lst.empty()) {
    return ConcatAscend::Call(tensors_tensor_list, axis);
  }
  std::vector<TensorPtr> tensors_tensor_list_vector(lst.size());
  auto axis_imm = GetValue<int64_t>(axis);
  ShapeVector output_shape;
  TypeId output_type{kTypeUnknown};
  for (size_t i = 0; i < lst.size(); ++i) {
    auto input_i = GetValue<TensorPtr>(lst[i]);
    // check if input is view
    if (input_i->is_contiguous()) {
      return ConcatAscend::Call(tensors_tensor_list, axis);
    }
    if (i == 0) {
      output_shape = input_i->shape();
      // check if axis is 0
      if (output_shape.empty() || (axis_imm != 0 && axis_imm != -static_cast<int64_t>(output_shape.size()))) {
        return ConcatAscend::Call(tensors_tensor_list, axis);
      }
      output_type = input_i->data_type();
    } else {
      const auto &shape_i = input_i->shape();
      if (input_i->data_type() != output_type || input_i->shape().size() != output_shape.size()) {
        return ConcatAscend::Call(tensors_tensor_list, axis);
      }
      output_shape[0] += shape_i[0];
    }
    tensors_tensor_list_vector[i] = input_i;
  }
  MS_LOG(INFO) << op_name() << " call start";
  // create output tensor
  auto output_tensor = tensor::from_spec(output_type, output_shape, device::DeviceType::kNone);
  output_tensor->set_need_pipeline_sync(true);
  outputs_.push_back(output_tensor);
  PyBoostUtils::PrepareOpInputs(device_context_, stream_id_, tensors_tensor_list_vector);
  PyBoostUtils::PrepareOpOutputs(device_context_, stream_id_, outputs_);
  ProfileTrackerTask();
  // Async
  auto op = get_op();
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, tensors_tensor_list_vector]() {
    MS_LOG(INFO) << "Run device task " << op_name() << " start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, tensors_tensor_list_vector);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);
    auto device_address = std::static_pointer_cast<device::DeviceAddress>(outputs[0]->device_address());
    auto device_ptr = reinterpret_cast<uint8_t *>(const_cast<void *>(device_address->GetPtr()));
    auto device_size = device_address->GetSize();
    size_t offset = 0;
    for (size_t i = 0; i < tensors_tensor_list_vector.size(); ++i) {
      auto output_tensor_i =
        tensor::from_spec(outputs[0]->data_type(), tensors_tensor_list_vector[i]->shape(), device::DeviceType::kNone);
      auto sz = LongToSize(output_tensor_i->DataNBytes());
      device_address->set_ptr(device_ptr + offset);
      device_address->SetSize(sz);
      output_tensor_i->set_device_address(device_address);
      output_tensor_i->set_format(outputs[0]->format());
      output_tensor_i->set_storage_info(outputs[0]->storage_info());
      LAUNCH_ACLNN(aclnnInplaceCopy, device_context, op->stream_id(), output_tensor_i, tensors_tensor_list_vector[i]);
      offset += sz;
    }
    // Recover device_address
    device_address->set_ptr(device_ptr);
    device_address->SetSize(device_size);
    MS_LOG(INFO) << "Run device task " << op_name() << " end";
  }));
  CreateOutputSimpleInfo();
  ProfileTrackerInput(tensors_tensor_list, axis);
  ProfileTrackerOutput(outputs_[0]);
  MS_LOG(INFO) << op_name() << " call end";
  return outputs_[0];
}

tensor::TensorPtr CastAscendDvm::Call(const TensorPtr &input_tensor, const Int64ImmPtr &dtype) {
  auto dst_type = static_cast<TypeId>(GetValue<int64_t>(dtype));
  if (!InputCheck(input_tensor, IsSupportType) || !IsSupportType(dst_type)) {
    return CastAscend::Call(input_tensor, dtype);
  }
  DvmCall(
    op_name_, this,
    [&input_tensor, dst_type](LazyFusionKernelAscend *k) -> TensorPtr {
      auto src_obj = k->Input(input_tensor, false);
      auto dst_dtype = k->TransType(dst_type);
      auto obj = k->Cast(src_obj, dst_dtype);
      return k->Output(obj, dst_type, input_tensor->shape());
    },
    input_tensor);
  DumpOp(op_name_, this, input_tensor, dst_type);
  return outputs_.front();
}

tensor::TensorPtr AbsAscendDvm::Call(const TensorPtr &input_tensor) {
  if (!InputCheck(input_tensor, IsFloatIntType)) {
    return AbsAscend::Call(input_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      auto obj = k->Unary(dvm::UnaryOpType::kAbs, k->Input(input_tensor));
      return k->Output(obj, input_tensor->data_type(), input_tensor->shape());
    },
    input_tensor);
  DumpOp(op_name_, this, input_tensor);
  return outputs_.front();
}

tensor::TensorPtr NegAscendDvm::Call(const TensorPtr &input_tensor) {
  if (!InputCheck(input_tensor, IsFloatIntType)) {
    return NegAscend::Call(input_tensor);
  }
  DvmCall(
    op_name_, this,
    [&input_tensor](LazyFusionKernelAscend *k) -> TensorPtr {
      auto obj = input_tensor->data_type() == kNumberTypeInt32
                   ? k->Binary(dvm::BinaryOpType::kMul, k->Input(input_tensor), -1)
                   : k->Binary(dvm::BinaryOpType::kMul, k->Input(input_tensor), -1.0f);
      return k->Output(obj, input_tensor->data_type(), input_tensor->shape());
    },
    input_tensor);
  DumpOp(op_name_, this, input_tensor);
  return outputs_.front();
}

tensor::TensorPtr ExpAscendDvm::Call(const TensorPtr &input_tensor) {
  if (!InputCheck(input_tensor)) {
    return ExpAscend::Call(input_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      auto obj = k->Unary(dvm::UnaryOpType::kExp, k->Input(input_tensor));
      return k->Output(obj, input_tensor->data_type(), input_tensor->shape());
    },
    input_tensor);
  DumpOp(op_name_, this, input_tensor);
  return outputs_.front();
}

tensor::TensorPtr SqrtAscendDvm::Call(const TensorPtr &x_tensor) {
  if (!InputCheck(x_tensor)) {
    return SqrtAscend::Call(x_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      auto obj = k->Unary(dvm::UnaryOpType::kSqrt, k->Input(x_tensor));
      return k->Output(obj, x_tensor->data_type(), x_tensor->shape());
    },
    x_tensor);
  DumpOp(op_name_, this, x_tensor);
  return outputs_.front();
}

tensor::TensorPtr ReciprocalAscendDvm::Call(const TensorPtr &x_tensor) {
  if (!InputCheck(x_tensor)) {
    return ReciprocalAscend::Call(x_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      auto obj = k->Unary(dvm::UnaryOpType::kReciprocal, k->Input(x_tensor));
      return k->Output(obj, x_tensor->data_type(), x_tensor->shape());
    },
    x_tensor);
  DumpOp(op_name_, this, x_tensor);
  return outputs_.front();
}

tensor::TensorPtr IsFiniteAscendDvm::Call(const TensorPtr &x_tensor) {
  if (!InputCheck(x_tensor)) {
    return IsFiniteAscend::Call(x_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      auto obj = k->Unary(dvm::UnaryOpType::kIsFinite, k->Input(x_tensor));
      return k->Output(obj, kNumberTypeBool, x_tensor->shape());
    },
    x_tensor);
  DumpOp(op_name_, this, x_tensor);
  return outputs_.front();
}

tensor::TensorPtr RoundAscendDvm::Call(const TensorPtr &x_tensor, const Int64ImmPtr &decimals) {
  if (!InputCheck(x_tensor) || decimals->value() != 0) {
    return RoundAscend::Call(x_tensor, decimals);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      auto obj = k->Unary(dvm::UnaryOpType::kRound, k->Input(x_tensor));
      return k->Output(obj, x_tensor->data_type(), x_tensor->shape());
    },
    x_tensor);
  DumpOp(op_name_, this, x_tensor, decimals);
  return outputs_.front();
}

tensor::TensorPtr CeilAscendDvm::Call(const TensorPtr &x_tensor) {
  if (!InputCheck(x_tensor)) {
    return CeilAscend::Call(x_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      auto obj = k->Unary(dvm::UnaryOpType::kCeil, k->Input(x_tensor));
      return k->Output(obj, x_tensor->data_type(), x_tensor->shape());
    },
    x_tensor);
  DumpOp(op_name_, this, x_tensor);
  return outputs_.front();
}

tensor::TensorPtr FloorAscendDvm::Call(const TensorPtr &x_tensor) {
  if (!InputCheck(x_tensor)) {
    return FloorAscend::Call(x_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      auto obj = k->Unary(dvm::UnaryOpType::kFloor, k->Input(x_tensor));
      return k->Output(obj, x_tensor->data_type(), x_tensor->shape());
    },
    x_tensor);
  DumpOp(op_name_, this, x_tensor);
  return outputs_.front();
}

tensor::TensorPtr TruncAscendDvm::Call(const TensorPtr &x_tensor) {
  if (!InputCheck(x_tensor)) {
    return TruncAscend::Call(x_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      auto obj = k->Unary(dvm::UnaryOpType::kTrunc, k->Input(x_tensor));
      return k->Output(obj, x_tensor->data_type(), x_tensor->shape());
    },
    x_tensor);
  DumpOp(op_name_, this, x_tensor);
  return outputs_.front();
}

tensor::TensorPtr EqualAscendDvm::Call(const TensorPtr &input_tensor, const TensorPtr &other_tensor) {
  if (!BinaryInputCheck(input_tensor, other_tensor)) {
    return EqualAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kEqual, input_tensor, other_tensor, kNumberTypeBool);
  DumpOp(op_name_, this, input_tensor, other_tensor);
  return outputs_.front();
}

tensor::TensorPtr NotEqualAscendDvm::Call(const TensorPtr &input_tensor, const TensorPtr &other_tensor) {
  if (!BinaryInputCheck(input_tensor, other_tensor)) {
    return NotEqualAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kNotEqual, input_tensor, other_tensor, kNumberTypeBool);
  DumpOp(op_name_, this, input_tensor, other_tensor);
  return outputs_.front();
}

tensor::TensorPtr GreaterAscendDvm::Call(const TensorPtr &input_tensor, const TensorPtr &other_tensor) {
  if (!BinaryInputCheck(input_tensor, other_tensor)) {
    return GreaterAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kGreater, input_tensor, other_tensor, kNumberTypeBool);
  DumpOp(op_name_, this, input_tensor, other_tensor);
  return outputs_.front();
}

tensor::TensorPtr GreaterEqualAscendDvm::Call(const TensorPtr &input_tensor, const TensorPtr &other_tensor) {
  if (!BinaryInputCheck(input_tensor, other_tensor)) {
    return GreaterEqualAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kGreaterEqual, input_tensor, other_tensor, kNumberTypeBool);
  DumpOp(op_name_, this, input_tensor, other_tensor);
  return outputs_.front();
}

tensor::TensorPtr LessAscendDvm::Call(const TensorPtr &input_tensor, const TensorPtr &other_tensor) {
  if (!BinaryInputCheck(input_tensor, other_tensor)) {
    return LessAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kLess, input_tensor, other_tensor, kNumberTypeBool);
  DumpOp(op_name_, this, input_tensor, other_tensor);
  return outputs_.front();
}

tensor::TensorPtr LessEqualAscendDvm::Call(const TensorPtr &input_tensor, const TensorPtr &other_tensor) {
  if (!BinaryInputCheck(input_tensor, other_tensor)) {
    return LessEqualAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kLessEqual, input_tensor, other_tensor, kNumberTypeBool);
  DumpOp(op_name_, this, input_tensor, other_tensor);
  return outputs_.front();
}

tensor::TensorPtr AddAscendDvm::Call(const TensorPtr &input_tensor, const TensorPtr &other_tensor) {
  if (!BinaryInputCheck(input_tensor, other_tensor, IsFloatIntType)) {
    return AddAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kAdd, input_tensor, other_tensor, input_tensor->data_type());
  DumpOp(op_name_, this, input_tensor, other_tensor);
  return outputs_.front();
}

tensor::TensorPtr MulAscendDvm::Call(const TensorPtr &input_tensor, const TensorPtr &other_tensor) {
  if (!BinaryInputCheck(input_tensor, other_tensor, IsFloatIntType)) {
    return MulAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kMul, input_tensor, other_tensor, input_tensor->data_type());
  DumpOp(op_name_, this, input_tensor, other_tensor);
  return outputs_.front();
}

tensor::TensorPtr SubAscendDvm::Call(const TensorPtr &input_tensor, const TensorPtr &other_tensor) {
  if (!BinaryInputCheck(input_tensor, other_tensor, IsFloatIntType)) {
    return SubAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kSub, input_tensor, other_tensor, input_tensor->data_type());
  DumpOp(op_name_, this, input_tensor, other_tensor);
  return outputs_.front();
}

tensor::TensorPtr DivAscendDvm::Call(const TensorPtr &input_tensor, const TensorPtr &other_tensor) {
  if (!BinaryInputCheck(input_tensor, other_tensor)) {
    return DivAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kDiv, input_tensor, other_tensor, input_tensor->data_type());
  DumpOp(op_name_, this, input_tensor, other_tensor);
  return outputs_.front();
}

tensor::TensorPtr PowAscendDvm::Call(const TensorPtr &input_tensor, const TensorPtr &other_tensor) {
  if (!BinaryInputCheck(input_tensor, other_tensor)) {
    return PowAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kPow, input_tensor, other_tensor, input_tensor->data_type());
  DumpOp(op_name_, this, input_tensor, other_tensor);
  return outputs_.front();
}

tensor::TensorPtr MaximumAscendDvm::Call(const TensorPtr &input_tensor, const TensorPtr &other_tensor) {
  if (!BinaryInputCheck(input_tensor, other_tensor, IsFloatIntType)) {
    return MaximumAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kMaximum, input_tensor, other_tensor, input_tensor->data_type());
  DumpOp(op_name_, this, input_tensor, other_tensor);
  return outputs_.front();
}

tensor::TensorPtr MinimumAscendDvm::Call(const TensorPtr &input_tensor, const TensorPtr &other_tensor) {
  if (!BinaryInputCheck(input_tensor, other_tensor, IsFloatIntType)) {
    return MinimumAscend::Call(input_tensor, other_tensor);
  }
  BinaryDvmCall(op_name_, this, dvm::BinaryOpType::kMinimum, input_tensor, other_tensor, input_tensor->data_type());
  DumpOp(op_name_, this, input_tensor, other_tensor);
  return outputs_.front();
}

tensor::TensorPtr MulsAscendDvm::Call(const TensorPtr &input_tensor, const ScalarPtr &other) {
  auto [succ, scalar] = GetScalarValue<float>(other);
  if (!InputCheck(input_tensor) || !succ) {
    return MulsAscend::Call(input_tensor, other);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      auto obj = k->Binary(dvm::BinaryOpType::kMul, k->Input(input_tensor), scalar);
      return k->Output(obj, input_tensor->data_type(), input_tensor->shape());
    },
    input_tensor);
  DumpOp(op_name_, this, input_tensor, other);
  return outputs_.front();
}

tensor::TensorPtr LogicalNotAscendDvm::Call(const TensorPtr &x_tensor) {
  if (!InputCheck(x_tensor, IsSupportType)) {
    return LogicalNotAscend::Call(x_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      auto input_obj = k->Cast(k->Input(x_tensor), dvm::DType::kBool);
      auto obj = k->Unary(dvm::UnaryOpType::kLogicalNot, input_obj);
      return k->Output(obj, kNumberTypeBool, x_tensor->shape());
    },
    x_tensor);
  DumpOp(op_name_, this, x_tensor);
  return outputs_.front();
}

tensor::TensorPtr LogicalAndAscendDvm::Call(const TensorPtr &input_tensor, const TensorPtr &other_tensor) {
  if (!BinaryInputCheck(input_tensor, other_tensor, IsSupportType)) {
    return LogicalAndAscend::Call(input_tensor, other_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      auto input_obj = k->Cast(k->Input(input_tensor), dvm::DType::kBool);
      auto other_obj = k->Cast(k->Input(other_tensor), dvm::DType::kBool);
      auto obj = k->Binary(dvm::BinaryOpType::kLogicalAnd, input_obj, other_obj);
      return k->Output(obj, kNumberTypeBool, k->GetShape(obj));
    },
    input_tensor, other_tensor);
  DumpOp(op_name_, this, input_tensor, other_tensor);
  return outputs_.front();
}

tensor::TensorPtr LogicalOrAscendDvm::Call(const TensorPtr &input_tensor, const TensorPtr &other_tensor) {
  if (!BinaryInputCheck(input_tensor, other_tensor, IsSupportType)) {
    return LogicalOrAscend::Call(input_tensor, other_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      auto input_obj = k->Cast(k->Input(input_tensor), dvm::DType::kBool);
      auto other_obj = k->Cast(k->Input(other_tensor), dvm::DType::kBool);
      auto obj = k->Binary(dvm::BinaryOpType::kLogicalOr, input_obj, other_obj);
      return k->Output(obj, kNumberTypeBool, k->GetShape(obj));
    },
    input_tensor, other_tensor);
  DumpOp(op_name_, this, input_tensor, other_tensor);
  return outputs_.front();
}

tensor::TensorPtr SigmoidAscendDvm::Call(const TensorPtr &input_tensor) {
  if (!InputCheck(input_tensor)) {
    return SigmoidAscend::Call(input_tensor);
  }
  DvmCall(
    op_name_, this,
    [&input_tensor](LazyFusionKernelAscend *k) -> TensorPtr {
      auto input_obj = k->Input(input_tensor);
      input_obj = k->Cast(input_obj, dvm::DType::kFloat32);
      auto neg_x = k->Binary(dvm::BinaryOpType::kMul, input_obj, -1.0f);
      auto exp_neg_x = k->Unary(dvm::UnaryOpType::kExp, neg_x);
      auto add_exp = k->Binary(dvm::BinaryOpType::kAdd, exp_neg_x, 1.0f);
      auto obj = k->Unary(dvm::UnaryOpType::kReciprocal, add_exp);
      // obj cast inside Output
      return k->Output(obj, input_tensor->data_type(), input_tensor->shape());
    },
    input_tensor);
  DumpOp(op_name_, this, input_tensor);
  return outputs_.front();
}

tensor::TensorPtr SigmoidGradAscendDvm::Call(const TensorPtr &y_tensor, const TensorPtr &dy_tensor) {
  if (!BinaryInputCheck(y_tensor, dy_tensor)) {
    return SigmoidGradAscend::Call(y_tensor, dy_tensor);
  }
  DvmCall(
    op_name_, this,
    [&y_tensor, &dy_tensor](LazyFusionKernelAscend *k) -> TensorPtr {
      auto y_obj = k->Input(y_tensor);
      auto dy_obj = k->Input(dy_tensor);
      y_obj = k->Cast(y_obj, dvm::DType::kFloat32);
      dy_obj = k->Cast(dy_obj, dvm::DType::kFloat32);
      auto one_sub_y = k->Binary(dvm::BinaryOpType::kSub, 1.0f, y_obj);
      auto y_mul_dy = k->Binary(dvm::BinaryOpType::kMul, y_obj, dy_obj);
      auto obj = k->Binary(dvm::BinaryOpType::kMul, one_sub_y, y_mul_dy);
      return k->Output(obj, y_tensor->data_type(), y_tensor->shape());
    },
    y_tensor, dy_tensor);
  DumpOp(op_name_, this, y_tensor, dy_tensor);
  return outputs_.front();
}

tensor::TensorPtr SiLUAscendDvm::Call(const TensorPtr &input_tensor) {
  if (!InputCheck(input_tensor)) {
    return SiLUAscend::Call(input_tensor);
  }
  DvmCall(
    op_name_, this,
    [&input_tensor](LazyFusionKernelAscend *k) -> TensorPtr {
      auto input_obj = k->Input(input_tensor);
      input_obj = k->Cast(input_obj, dvm::DType::kFloat32);
      auto neg_x = k->Binary(dvm::BinaryOpType::kMul, input_obj, -1.0f);
      auto exp_neg_x = k->Unary(dvm::UnaryOpType::kExp, neg_x);
      auto add_exp = k->Binary(dvm::BinaryOpType::kAdd, exp_neg_x, 1.0f);
      auto obj = k->Binary(dvm::BinaryOpType::kDiv, input_obj, add_exp);
      return k->Output(obj, input_tensor->data_type(), input_tensor->shape());
    },
    input_tensor);
  DumpOp(op_name_, this, input_tensor);
  return outputs_.front();
}

tensor::TensorPtr SiLUGradAscendDvm::Call(const TensorPtr &dout_tensor, const TensorPtr &x_tensor) {
  if (!BinaryInputCheck(dout_tensor, x_tensor)) {
    return SiLUGradAscend::Call(dout_tensor, x_tensor);
  }
  DvmCall(
    op_name_, this,
    [&dout_tensor, &x_tensor](LazyFusionKernelAscend *k) -> TensorPtr {
      auto dout_obj = k->Input(dout_tensor);
      auto x_obj = k->Input(x_tensor);
      x_obj = k->Cast(x_obj, dvm::DType::kFloat32);
      dout_obj = k->Cast(dout_obj, dvm::DType::kFloat32);
      auto neg_x = k->Binary(dvm::BinaryOpType::kMul, x_obj, -1.0f);
      auto exp_neg_x = k->Unary(dvm::UnaryOpType::kExp, neg_x);
      auto add_exp = k->Binary(dvm::BinaryOpType::kAdd, exp_neg_x, 1.0f);
      auto sigmod = k->Unary(dvm::UnaryOpType::kReciprocal, add_exp);
      auto out = k->Binary(dvm::BinaryOpType::kDiv, x_obj, add_exp);
      auto sigmod_out0 = k->Binary(dvm::BinaryOpType::kAdd, sigmod, out);
      auto sigmod_out1 = k->Binary(dvm::BinaryOpType::kMul, sigmod, out);
      auto sub_res = k->Binary(dvm::BinaryOpType::kSub, sigmod_out0, sigmod_out1);
      auto obj = k->Binary(dvm::BinaryOpType::kMul, sub_res, dout_obj);
      return k->Output(obj, x_tensor->data_type(), x_tensor->shape());
    },
    dout_tensor, x_tensor);
  DumpOp(op_name_, this, dout_tensor, x_tensor);
  return outputs_.front();
}

tensor::TensorPtr GeLUAscendDvm::Call(const TensorPtr &input_tensor) {
  if (!InputCheck(input_tensor)) {
    return GeLUAscend::Call(input_tensor);
  }
  DvmCall(
    op_name_, this,
    [&input_tensor](LazyFusionKernelAscend *k) -> TensorPtr {
      // Constants used in the GeLU approximation
      constexpr float csv_value = 0.044715f;
      constexpr float csv_value_sqrt_eight_div_pi = -1.5957691216057308f;
      auto x_obj = k->Input(input_tensor);
      x_obj = k->Cast(x_obj, dvm::DType::kFloat32);
      // Compute x^2
      auto x_squared = k->Binary(dvm::BinaryOpType::kMul, x_obj, x_obj);
      // Compute x^3
      auto x_cubed = k->Binary(dvm::BinaryOpType::kMul, x_squared, x_obj);
      // mul_1 = x_cubed * csv_value
      auto mul_1 = k->Binary(dvm::BinaryOpType::kMul, x_cubed, csv_value);
      // tanh_res = x_obj + mul_1
      auto tanh_res = k->Binary(dvm::BinaryOpType::kAdd, x_obj, mul_1);
      // y = tanh_res * csv_value_sqrt_eight_div_pi
      auto y = k->Binary(dvm::BinaryOpType::kMul, tanh_res, csv_value_sqrt_eight_div_pi);
      // exp_0 = exp(y)
      auto exp_0 = k->Unary(dvm::UnaryOpType::kExp, y);
      // add_0 = exp_0 + 1
      auto add_0 = k->Binary(dvm::BinaryOpType::kAdd, exp_0, 1.0f);
      // result = x_obj / add_0
      auto result = k->Binary(dvm::BinaryOpType::kDiv, x_obj, add_0);
      return k->Output(result, input_tensor->data_type(), input_tensor->shape());
    },
    input_tensor);
  DumpOp(op_name_, this, input_tensor);
  return outputs_.front();
}

tensor::TensorPtr GeLUGradAscendDvm::Call(const TensorPtr &dy_tensor, const TensorPtr &x_tensor,
                                          const TensorPtr &y_tensor) {
  if (!InputCheck(dy_tensor) || !InputCheck(x_tensor) || !InputCheck(y_tensor) ||
      x_tensor->data_type() != dy_tensor->data_type() || y_tensor->data_type() != dy_tensor->data_type()) {
    return GeLUGradAscend::Call(dy_tensor, x_tensor, y_tensor);
  }
  DvmCall(
    op_name_, this,
    [&dy_tensor, &x_tensor](LazyFusionKernelAscend *k) -> TensorPtr {
      // Constants used in the GeLU gradient computation
      constexpr float cs_value = 0.044715f;
      constexpr float cs_sqrt_two_div_pi = 0.7978845608028564f;  // sqrt(2 / π)
      constexpr float cs_value_tri = 0.134145f;                  // cs_value * 3
      auto dy_obj = k->Input(dy_tensor);
      auto x_obj = k->Input(x_tensor);
      x_obj = k->Cast(x_obj, dvm::DType::kFloat32);
      dy_obj = k->Cast(dy_obj, dvm::DType::kFloat32);
      // Compute x^2
      auto x_squared = k->Binary(dvm::BinaryOpType::kMul, x_obj, x_obj);
      // Compute x^3
      auto x_cubed = k->Binary(dvm::BinaryOpType::kMul, x_squared, x_obj);
      // Calculate mul_right
      auto mul_double_mul_tri = k->Binary(dvm::BinaryOpType::kMul, cs_value_tri, x_squared);
      auto mul_add_one = k->Binary(dvm::BinaryOpType::kAdd, 1.0f, mul_double_mul_tri);
      auto mul_right = k->Binary(dvm::BinaryOpType::kMul, cs_sqrt_two_div_pi, mul_add_one);
      // Calculate tanh_para
      auto mul_triple_mul_csvalue = k->Binary(dvm::BinaryOpType::kMul, cs_value, x_cubed);
      auto mul_add_x = k->Binary(dvm::BinaryOpType::kAdd, x_obj, mul_triple_mul_csvalue);
      auto tanh_para = k->Binary(dvm::BinaryOpType::kMul, cs_sqrt_two_div_pi, mul_add_x);
      // Compute tanh_res = tanh(tanh_para)
      auto Tanh = [&k](const auto &input) {
        // Implement tanh(x) = 1 - 2 / (e^{2x} + 1)
        auto two_input = k->Binary(dvm::BinaryOpType::kMul, input, 2.0f);
        auto exp_two_input = k->Unary(dvm::UnaryOpType::kExp, two_input);
        auto denom = k->Binary(dvm::BinaryOpType::kAdd, exp_two_input, 1.0f);
        auto two_div_denom = k->Binary(dvm::BinaryOpType::kDiv, 2.0f, denom);
        return k->Binary(dvm::BinaryOpType::kSub, 1.0f, two_div_denom);
      };
      auto tanh_res = Tanh(tanh_para);
      // Compute 0.5 * (1.0 + tanh_res)
      auto tanh_res_add_one = k->Binary(dvm::BinaryOpType::kAdd, 1.0f, tanh_res);
      auto half_mul_tanh_res_add_one = k->Binary(dvm::BinaryOpType::kMul, 0.5f, tanh_res_add_one);
      // Compute 1.0 - tanh_res^2
      auto tanh_res_squared = k->Binary(dvm::BinaryOpType::kMul, tanh_res, tanh_res);
      auto one_sub_tanh_res_squared = k->Binary(dvm::BinaryOpType::kSub, 1.0f, tanh_res_squared);
      // Compute 0.5 * x_obj * (1 - tanh_res^2)
      auto half_mul_x = k->Binary(dvm::BinaryOpType::kMul, 0.5f, x_obj);
      auto mul_tmp = k->Binary(dvm::BinaryOpType::kMul, half_mul_x, one_sub_tanh_res_squared);
      // Compute mul_final = mul_tmp * mul_right
      auto mul_final = k->Binary(dvm::BinaryOpType::kMul, mul_tmp, mul_right);
      // Compute result_tmp = half_mul_tanh_res_add_one + mul_final
      auto result_tmp = k->Binary(dvm::BinaryOpType::kAdd, half_mul_tanh_res_add_one, mul_final);
      // Compute result = dy_obj * result_tmp
      auto result = k->Binary(dvm::BinaryOpType::kMul, dy_obj, result_tmp);
      return k->Output(result, x_tensor->data_type(), x_tensor->shape());
    },
    dy_tensor, x_tensor);
  DumpOp(op_name_, this, dy_tensor, x_tensor, y_tensor);
  return outputs_.front();
}

tensor::TensorPtr ReLUAscendDvm::Call(const TensorPtr &input_tensor) {
  if (!InputCheck(input_tensor)) {
    return ReLUAscend::Call(input_tensor);
  }
  DvmCall(
    op_name_, this,
    [&input_tensor](LazyFusionKernelAscend *k) -> TensorPtr {
      auto obj = k->Binary(dvm::BinaryOpType::kMaximum, k->Input(input_tensor), 0.0f);
      return k->Output(obj, input_tensor->data_type(), input_tensor->shape());
    },
    input_tensor);
  DumpOp(op_name_, this, input_tensor);
  return outputs_.front();
}

tensor::TensorPtr SumExtAscendDvm::Call(const TensorPtr &input_tensor, const std::optional<ValueTuplePtr> &dim,
                                        const BoolImmPtr &keepdim, const std::optional<Int64ImmPtr> &dtype) {
  auto input_type = input_tensor->data_type();
  auto dst_type = dtype.has_value() ? static_cast<TypeId>(GetValue<int64_t>(dtype.value())) : input_type;
  // the Cast after ReduceSum will has performance problem
  if (input_type != kNumberTypeFloat32 || dst_type != input_type || !InputCheck(input_tensor)) {
    return SumExtAscend::Call(input_tensor, dim, keepdim, dtype);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      auto dim_value = GetReduceDim(dim, input_tensor->shape().size());
      auto reduce_obj =
        k->Reduce(dvm::ReduceOpType::kSum, k->Input(input_tensor), k->GetShapeRef(dim_value), keepdim->value());
      return k->Output(reduce_obj, input_type, k->GetShape(reduce_obj));
    },
    input_tensor);
  DumpOp(op_name_, this, input_tensor, dim, keepdim, dtype);
  FlushLazyFusion();  // reduce post fusion not supported
  return outputs_.front();
}

tensor::TensorPtr AddExtAscendDvm::Call(const TensorPtr &input_tensor, const TensorPtr &other_tensor,
                                        const ScalarPtr &alpha) {
  if (!AddExtDvmCall(op_name_, this, input_tensor, other_tensor, alpha, false)) {
    return AddExtAscend::Call(input_tensor, other_tensor, alpha);
  }
  DumpOp(op_name_, this, input_tensor, other_tensor, alpha);
  return outputs_.front();
}

tensor::TensorPtr SubExtAscendDvm::Call(const TensorPtr &input_tensor, const TensorPtr &other_tensor,
                                        const ScalarPtr &alpha) {
  if (!SubExtDvmCall(op_name_, this, input_tensor, other_tensor, alpha, false)) {
    return SubExtAscend::Call(input_tensor, other_tensor, alpha);
  }
  DumpOp(op_name_, this, input_tensor, other_tensor, alpha);
  return outputs_.front();
}

tensor::TensorPtr TileAscendDvm::Call(const TensorPtr &input_tensor, const ValueTuplePtr &dims) {
  if (!InputCheck(input_tensor, IsFloatIntType)) {
    return TileAscend::Call(input_tensor, dims);
  }
  auto input_shape = input_tensor->shape();
  ShapeVector output_shape = ConvertValueTupleToVector<int64_t>(dims);
  ops::AdaptShapeAndMultipies(&input_shape, &output_shape);
  for (size_t i = 0; i < output_shape.size(); ++i) {
    if (output_shape[i] != 1 && input_shape[i] != 1) {
      return TileAscend::Call(input_tensor, dims);
    }
    output_shape[i] *= input_shape[i];
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      auto out_obj = k->Broadcast(k->Input(input_tensor), k->GetShapeRef(output_shape));
      return k->Output(out_obj, input_tensor->data_type(), output_shape);
    },
    input_tensor);
  DumpOp(op_name_, this, input_tensor, dims);
  return outputs_.front();
}

tensor::TensorPtr LinalgVectorNormAscendDvm::Call(const TensorPtr &x_tensor, const FP32ImmPtr &ord,
                                                  const std::optional<ValueTuplePtr> &dim, const BoolImmPtr &keepdim,
                                                  const std::optional<Int64ImmPtr> &dtype) {
  auto output_type = dtype.has_value() ? static_cast<TypeId>(GetValue<int64_t>(dtype.value())) : x_tensor->data_type();
  auto ord_value = ord->value();
  if (!InputCheck(x_tensor) || output_type != kNumberTypeFloat32 ||
      (ord_value != 1.0f && ord_value != 2.0f && ord_value != 3.0f)) {
    return LinalgVectorNormAscend::Call(x_tensor, ord, dim, keepdim, dtype);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      auto dim_value = GetReduceDim(dim, x_tensor->shape().size());
      auto input_obj = k->Cast(k->Input(x_tensor), dvm::DType::kFloat32);
      dvm::NDObject *out_obj = nullptr;
      // sum(|x|^ord)^(1/ord)
      if (ord_value == 1.0f) {
        auto x_abs = k->Unary(dvm::UnaryOpType::kAbs, input_obj);
        out_obj = k->Reduce(dvm::ReduceOpType::kSum, x_abs, k->GetShapeRef(dim_value), keepdim->value());
      } else if (ord_value == 2.0f) {
        auto x_square = k->Binary(dvm::BinaryOpType::kMul, input_obj, input_obj);
        auto x_sum = k->Reduce(dvm::ReduceOpType::kSum, x_square, k->GetShapeRef(dim_value), keepdim->value());
        out_obj = k->Unary(dvm::UnaryOpType::kSqrt, x_sum);
      } else {
        auto x_abs = k->Unary(dvm::UnaryOpType::kAbs, input_obj);
        auto x_pow = k->Binary(dvm::BinaryOpType::kPow, x_abs, ord_value);
        auto x_sum = k->Reduce(dvm::ReduceOpType::kSum, x_pow, k->GetShapeRef(dim_value), keepdim->value());
        out_obj = k->Binary(dvm::BinaryOpType::kPow, x_sum, 1.0f / ord_value);
      }
      return k->Output(out_obj, output_type, k->GetShape(out_obj));
    },
    x_tensor);
  DumpOp(op_name_, this, x_tensor, ord, dim, keepdim, dtype);
  FlushLazyFusion();
  return outputs_.front();
}

std::tuple<tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr> AdamWAscendDvm::Call(
  const TensorPtr &var_tensor, const TensorPtr &m_tensor, const TensorPtr &v_tensor, const TensorPtr &max_v_tensor,
  const TensorPtr &gradient_tensor, const TensorPtr &step_tensor, const FP32ImmPtr &lr, const FP32ImmPtr &beta1,
  const FP32ImmPtr &beta2, const FP32ImmPtr &decay, const FP32ImmPtr &eps, const BoolImmPtr &amsgrad,
  const BoolImmPtr &maximize) {
  auto var_type = var_tensor->data_type();
  bool all_contiguous = var_tensor->is_contiguous() && m_tensor->is_contiguous() && v_tensor->is_contiguous() &&
                        max_v_tensor->is_contiguous() && gradient_tensor->is_contiguous() &&
                        step_tensor->is_contiguous();
  const auto lr_imm = GetValue<float>(lr);
  const auto beta1_imm = GetValue<float>(beta1);
  const auto beta2_imm = GetValue<float>(beta2);
  const auto decay_imm = GetValue<float>(decay);
  const auto epsilon_imm = GetValue<float>(eps);
  const auto amsgrad_imm = GetValue<bool>(amsgrad);
  const auto maximize_imm = GetValue<bool>(maximize);
  // input check
  if (!all_contiguous || var_type != kNumberTypeFloat32 || m_tensor->data_type() != var_type ||
      v_tensor->data_type() != var_type || (maximize_imm && max_v_tensor->data_type() != var_type) ||
      gradient_tensor->data_type() != var_type) {
    return AdamWAscend::Call(var_tensor, m_tensor, v_tensor, max_v_tensor, gradient_tensor, step_tensor, lr, beta1,
                             beta2, decay, eps, amsgrad, maximize);
  }
  if (amsgrad_imm) {
    PyBoostUtils::PrepareOpInputs(device_context_, stream_id_, var_tensor, m_tensor, v_tensor, max_v_tensor,
                                  gradient_tensor, step_tensor);
  } else {
    PyBoostUtils::PrepareOpInputs(device_context_, stream_id_, var_tensor, m_tensor, v_tensor, gradient_tensor,
                                  step_tensor);
  }
  auto k = g_lazy_fusion_manager.Get(device_context_, stream_id_);
  CALL_START(op_name_, k);
  auto var_obj = k->Input(var_tensor);
  auto m_obj = k->Input(m_tensor);
  auto v_obj = k->Input(v_tensor);
  auto gradient_obj = k->Input(gradient_tensor);
  auto step_obj = k->Input(step_tensor);
  auto grad = maximize_imm ? k->Binary(dvm::BinaryOpType::kMul, gradient_obj, -1.0f) : gradient_obj;
  // m_t <-- beta1 * m + (1 - beta1) * grad
  auto m_t = k->Binary(dvm::BinaryOpType::kAdd, k->Binary(dvm::BinaryOpType::kMul, m_obj, beta1_imm),
                       k->Binary(dvm::BinaryOpType::kMul, grad, 1.0f - beta1_imm));
  // v_t <-- beta2 * v + (1 - beta2) * grad^2
  auto v_t =
    k->Binary(dvm::BinaryOpType::kAdd, k->Binary(dvm::BinaryOpType::kMul, v_obj, beta2_imm),
              k->Binary(dvm::BinaryOpType::kMul, k->Binary(dvm::BinaryOpType::kMul, grad, grad), 1.0f - beta2_imm));
  // var_t <-- var - lr * deday * var
  auto var_t = decay_imm == 0.0f ? var_obj : k->Binary(dvm::BinaryOpType::kMul, var_obj, 1.0f - lr_imm * decay_imm);
  // real step is step + 1
  auto step_value = step_tensor->data_type() == var_type ? step_obj : k->Cast(step_obj, k->TransType(var_type));
  step_value = k->Binary(dvm::BinaryOpType::kAdd, step_value, 1.0f);
  // var_t <-- var_t - (m_t * lr / (1 - beta1^t)) / (sqrt(max(max_v, v_t) / (1 - beta2^t)) + eps)
  auto scale1 =
    k->Binary(dvm::BinaryOpType::kDiv, lr_imm,
              k->Binary(dvm::BinaryOpType::kSub, 1.0f, k->Binary(dvm::BinaryOpType::kPow, beta1_imm, step_value)));
  auto m_t_scale = k->Binary(dvm::BinaryOpType::kMul, m_t, scale1);
  auto v_t_used = amsgrad_imm ? k->Binary(dvm::BinaryOpType::kMaximum, k->Input(max_v_tensor), v_t) : v_t;
  auto scale2 =
    k->Binary(dvm::BinaryOpType::kDiv, 1.0f,
              k->Binary(dvm::BinaryOpType::kSub, 1.0f, k->Binary(dvm::BinaryOpType::kPow, beta2_imm, step_value)));
  auto v_t_sqrt = k->Unary(dvm::UnaryOpType::kSqrt, k->Binary(dvm::BinaryOpType::kMul, v_t_used, scale2));
  var_t =
    k->Binary(dvm::BinaryOpType::kSub, var_t,
              k->Binary(dvm::BinaryOpType::kDiv, m_t_scale, k->Binary(dvm::BinaryOpType::kAdd, v_t_sqrt, epsilon_imm)));
  // update
  outputs_.push_back(var_tensor);
  k->Output(outputs_[kIndex0], var_t, true);
  outputs_.push_back(m_tensor);
  k->Output(outputs_[kIndex1], m_t, true);
  outputs_.push_back(v_tensor);
  k->Output(outputs_[kIndex2], v_t, true);
  for (const auto &output : outputs_) {
    output->set_need_pipeline_sync(true);
  }
  CreateOutputSimpleInfo();
  CALL_END(op_name_, k);
  DumpOp(op_name_, this, var_tensor, m_tensor, v_tensor, max_v_tensor, gradient_tensor, step_tensor, lr, beta1, beta2,
         decay, eps, amsgrad, maximize);
  FlushLazyFusion();
  return std::make_tuple(outputs_[kIndex0], outputs_[kIndex1], outputs_[kIndex2]);
}

tensor::TensorPtr InplaceCopyAscendDvm::Call(const TensorPtr &variable_tensor, const TensorPtr &value_tensor,
                                             const BoolImmPtr &non_blocking) {
  if (!InputCheck(variable_tensor, IsFloatIntType) || !InputCheck(value_tensor, IsFloatIntType) ||
      non_blocking->value()) {
    return InplaceCopyAscend::Call(variable_tensor, value_tensor, non_blocking);
  }
  if (SameTensor(variable_tensor, value_tensor)) {
    MS_LOG(INFO) << op_name() << " call skip";
    PyBoostUtils::PrepareOpInputs(device_context_, stream_id_, variable_tensor, value_tensor);
    outputs_.push_back(variable_tensor);
    outputs_[0]->set_need_pipeline_sync(true);
    CreateOutputSimpleInfo();
    return outputs_[0];
  }
  PyBoostUtils::PrepareOpInputs(device_context_, stream_id_, variable_tensor, value_tensor);
  auto k = g_lazy_fusion_manager.Get(device_context_, stream_id_);
  CALL_START(op_name_, k);
  // copy value_tensor to variable_tensor
  auto value_obj = k->Input(value_tensor, false);
  if (value_tensor->data_type() != variable_tensor->data_type()) {
    value_obj = k->Cast(value_obj, k->TransType(variable_tensor->data_type()));
  }
  if (value_tensor->shape() != variable_tensor->shape()) {
    value_obj = k->Broadcast(value_obj, k->GetShapeRef(variable_tensor->shape()));
  }
  outputs_.push_back(variable_tensor);
  k->Output(outputs_[0], value_obj, true);
  outputs_[0]->set_need_pipeline_sync(true);
  CreateOutputSimpleInfo();
  CALL_END(op_name_, k);
  DumpOp(op_name_, this, variable_tensor, value_tensor, non_blocking);
  FlushLazyFusion();
  return outputs_[0];
}

tensor::TensorPtr InplaceDivAscendDvm::Call(const TensorPtr &input_tensor, const TensorPtr &other_tensor) {
  if (!InputCheck(input_tensor) || !InputCheck(other_tensor)) {
    return InplaceDivAscend::Call(input_tensor, other_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      auto input_obj = k->Input(input_tensor);
      auto other_obj = k->Input(other_tensor);
      auto input_dtype = k->GetDType(input_obj);
      auto other_dtype = k->GetDType(other_obj);
      // inplace op supports different data types, should convert to same data type here
      if (other_dtype != input_dtype) {
        if (other_dtype == dvm::DType::kFloat32) {
          input_obj = k->Cast(input_obj, other_dtype);
        } else {
          other_obj = k->Cast(other_obj, input_dtype);
        }
      }
      auto out_obj = k->Binary(dvm::BinaryOpType::kDiv, input_obj, other_obj);
      // update
      k->Output(input_tensor, out_obj, true);
      return input_tensor;
    },
    input_tensor, other_tensor);
  DumpOp(op_name_, this, input_tensor, other_tensor);
  FlushLazyFusion();
  return outputs_[0];
}

tensor::TensorPtr InplaceExpAscendDvm::Call(const TensorPtr &input_tensor) {
  if (!InputCheck(input_tensor)) {
    return InplaceExpAscend::Call(input_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      auto out_obj = k->Unary(dvm::UnaryOpType::kExp, k->Input(input_tensor));
      // update
      k->Output(input_tensor, out_obj, true);
      return input_tensor;
    },
    input_tensor);
  DumpOp(op_name_, this, input_tensor);
  FlushLazyFusion();
  return outputs_[0];
}

tensor::TensorPtr InplaceAddExtAscendDvm::Call(const TensorPtr &input_tensor, const TensorPtr &other_tensor,
                                               const ScalarPtr &alpha) {
  if (!AddExtDvmCall(op_name_, this, input_tensor, other_tensor, alpha, true)) {
    return InplaceAddExtAscend::Call(input_tensor, other_tensor, alpha);
  }
  DumpOp(op_name_, this, input_tensor, other_tensor, alpha);
  FlushLazyFusion();
  return outputs_[0];
}

tensor::TensorPtr InplaceSubExtAscendDvm::Call(const TensorPtr &input_tensor, const TensorPtr &other_tensor,
                                               const ScalarPtr &alpha) {
  if (!SubExtDvmCall(op_name_, this, input_tensor, other_tensor, alpha, true)) {
    return InplaceSubExtAscend::Call(input_tensor, other_tensor, alpha);
  }
  DumpOp(op_name_, this, input_tensor, other_tensor, alpha);
  FlushLazyFusion();
  return outputs_[0];
}

tensor::TensorPtr InplaceReLUAscendDvm::Call(const TensorPtr &input_tensor) {
  if (!InputCheck(input_tensor)) {
    return InplaceReLUAscend::Call(input_tensor);
  }
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      auto out_obj = k->Binary(dvm::BinaryOpType::kMaximum, k->Input(input_tensor), 0.0f);
      // update
      k->Output(input_tensor, out_obj, true);
      return input_tensor;
    },
    input_tensor);
  DumpOp(op_name_, this, input_tensor);
  FlushLazyFusion();
  return outputs_[0];
}

tensor::TensorPtr MaskedFillAscendDvm::Call(const mindspore::tensor::TensorPtr &input_x_tensor,
                                            const mindspore::tensor::TensorPtr &mask_tensor,
                                            const mindspore::tensor::TensorPtr &value_tensor) {
  if (!InputCheck(input_x_tensor) || !InputCheck(mask_tensor) || !InputCheck(value_tensor)) {
    return MaskedFillAscend::Call(input_x_tensor, mask_tensor, value_tensor);
  }

  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      auto out_obj = k->Select(k->Input(value_tensor), k->Input(input_x_tensor), k->Input(mask_tensor));
      // update
      k->Output(input_x_tensor, out_obj);
      return input_x_tensor;
    },
    input_x_tensor, mask_tensor, value_tensor);
  DumpOp(op_name_, this, input_x_tensor, mask_tensor, value_tensor);
  FlushLazyFusion();
  return outputs_[0];
}

tensor::TensorPtr MatMulCall(const std::string &op_name, OpRunner *op, const TensorPtr &x_tensor,
                             const TensorPtr &y_tensor, bool trans_a, bool trans_b) {
  MatMulAdapter info(x_tensor, y_tensor, trans_a, trans_b);
  if (!info.Check(op)) {
    return nullptr;
  }
  FlushLazyFusion();
  DvmCall(
    op_name, op,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      auto input_obj = k->Input(info.x_tensor, false, info.x_shape);
      auto weight_obj = k->Input(info.y_tensor, false, info.y_shape);
      auto out_obj = k->MatMul(input_obj, weight_obj, info.trans_a, info.trans_b, nullptr);
      return k->Output(out_obj, x_tensor->data_type(), k->GetShape(out_obj));
    },
    info.x_tensor, info.y_tensor);
  return op->output(kIndex0);
}

tensor::TensorPtr DenseAscendDvm::Call(const TensorPtr &input_tensor, const TensorPtr &weight_tensor,
                                       const std::optional<TensorPtr> &bias_tensor) {
  TensorPtr bias = nullptr;
  if (bias_tensor.has_value()) {
    bias = bias_tensor.value();
    if (bias->shape().size() != kDim1 || !bias->is_contiguous()) {
      return DenseAscend::Call(input_tensor, weight_tensor, bias_tensor);
    }
  }
  MatMulAdapter info(input_tensor, weight_tensor, false, true);
  if (!info.Check(this)) {
    return DenseAscend::Call(input_tensor, weight_tensor, bias_tensor);
  }
  FlushLazyFusion();  // forward fusion not allowed
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      auto input_obj = k->Input(info.x_tensor, false, info.x_shape);
      auto weight_obj = k->Input(info.y_tensor, false, info.y_shape);
      auto bias_obj = bias == nullptr ? nullptr : k->Input(bias, false);
      auto out_obj = k->MatMul(input_obj, weight_obj, info.trans_a, info.trans_b, bias_obj);
      return k->Output(out_obj, input_tensor->data_type(), k->GetShape(out_obj));
    },
    info.x_tensor, info.y_tensor, bias_tensor);
  DumpOp(op_name_, this, input_tensor, weight_tensor, bias_tensor);
  return outputs_.front();
}

mindspore::tensor::TensorPtr BaddbmmAscendDvm::Call(const mindspore::tensor::TensorPtr &input_tensor,
                                                    const mindspore::tensor::TensorPtr &batch1_tensor,
                                                    const mindspore::tensor::TensorPtr &batch2_tensor,
                                                    const mindspore::ScalarPtr &beta,
                                                    const mindspore::ScalarPtr &alpha) {
  auto [beta_succ, beta_scalar] = GetScalarValue<float>(beta);
  auto [alpha_succ, alpha_scalar] = GetScalarValue<float>(alpha);
  if ((alpha_scalar != 0 && !InputCheck(input_tensor)) || !beta_succ || !alpha_succ) {
    return BaddbmmAscend::Call(input_tensor, batch1_tensor, batch2_tensor, beta, alpha);
  }
  MatMulAdapter info(batch1_tensor, batch2_tensor, false, false);
  if (!info.Check(this)) {
    return BaddbmmAscend::Call(input_tensor, batch1_tensor, batch2_tensor, beta, alpha);
  }
  FlushLazyFusion();  // forward fusion not allowed
  auto build_kernel = [&](LazyFusionKernelAscend *k) {
    auto input_obj = k->Input(info.x_tensor, false, info.x_shape);
    auto weight_obj = k->Input(info.y_tensor, false, info.y_shape);
    auto out_obj = k->MatMul(input_obj, weight_obj, info.trans_a, info.trans_b, nullptr);
    if (k->GetDType(out_obj) == dvm::kBFloat16) {
      out_obj = k->Cast(out_obj, dvm::kFloat32);
    }
    if (beta_scalar == 0) {
      out_obj = k->Binary(dvm::BinaryOpType::kMul, out_obj, alpha_scalar);
    } else {
      auto scaled_matmul = k->Binary(dvm::BinaryOpType::kMul, out_obj, alpha_scalar);
      auto scaled_input = k->Binary(dvm::BinaryOpType::kMul, k->Input(input_tensor), beta_scalar);
      out_obj = k->Binary(dvm::BinaryOpType::kAdd, scaled_matmul, scaled_input);
    }
    return k->Output(out_obj, info.x_tensor->data_type(), k->GetShape(out_obj));
  };

  if (beta_scalar == 0) {
    DvmCall(op_name_, this, build_kernel, info.x_tensor, info.y_tensor);
  } else {
    DvmCall(op_name_, this, build_kernel, info.x_tensor, info.y_tensor, input_tensor);
  }
  return outputs_.front();
}

tensor::TensorPtr MatMulAscendDvm::Call(const TensorPtr &input_tensor, const TensorPtr &mat2_tensor,
                                        const BoolImmPtr &transpose_a, const BoolImmPtr &transpose_b) {
  bool trans_a = GetValue<bool>(transpose_a);
  bool trans_b = GetValue<bool>(transpose_b);
  tensor::TensorPtr ret = MatMulCall(op_name_, this, input_tensor, mat2_tensor, trans_a, trans_b);
  if (ret == nullptr) {
    return MatMulAscend::Call(input_tensor, mat2_tensor, transpose_a, transpose_b);
  }
  DumpOp(op_name_, this, input_tensor, mat2_tensor, transpose_a, transpose_b);
  return ret;
}

tensor::TensorPtr BatchMatMulAscendDvm::Call(const TensorPtr &x_tensor, const TensorPtr &y_tensor,
                                             const BoolImmPtr &transpose_a, const BoolImmPtr &transpose_b) {
  bool trans_a = GetValue<bool>(transpose_a);
  bool trans_b = GetValue<bool>(transpose_b);
  tensor::TensorPtr ret = MatMulCall(op_name_, this, x_tensor, y_tensor, trans_a, trans_b);
  if (ret == nullptr) {
    return BatchMatMulAscend::Call(x_tensor, y_tensor, transpose_a, transpose_b);
  }
  DumpOp(op_name_, this, x_tensor, y_tensor, transpose_a, transpose_b);
  return ret;
}

tensor::TensorPtr MatMulExtAscendDvm::Call(const mindspore::tensor::TensorPtr &input_tensor,
                                           const mindspore::tensor::TensorPtr &other_tensor) {
  tensor::TensorPtr ret = MatMulCall(op_name_, this, input_tensor, other_tensor, false, false);
  if (ret == nullptr) {
    return MatMulExtAscend::Call(input_tensor, other_tensor);
  }
  DumpOp(op_name_, this, input_tensor, other_tensor);
  return ret;
}

tensor::TensorPtr BatchMatMulExtAscendDvm::Call(const mindspore::tensor::TensorPtr &input_tensor,
                                                const mindspore::tensor::TensorPtr &other_tensor) {
  tensor::TensorPtr ret = MatMulCall(op_name_, this, input_tensor, other_tensor, false, false);
  if (ret == nullptr) {
    return BatchMatMulExtAscend::Call(input_tensor, other_tensor);
  }
  DumpOp(op_name_, this, input_tensor, other_tensor);
  return ret;
}

std::tuple<TensorPtr, TensorPtr> BatchNormStatsAscendDvm::Call(const TensorPtr &input_tensor,
                                                               const mindspore::FP32ImmPtr &eps) {
  // input check
  if (NeedSync() || input_tensor->data_type() != kNumberTypeFloat32) {
    return BatchNormStatsAscend::Call(input_tensor, eps);
  }
  const auto &input_shape = input_tensor->shape();
  ShapeVector axis;
  axis.reserve(input_shape.size());
  int64_t count = 1;
  for (int64_t i = 0; i < static_cast<int64_t>(input_shape.size()); ++i) {
    if (i != 1) {  // reduce all axis except C channel(axis 1)
      axis.push_back(i);
      count *= input_shape[i];
    }
  }
  if (count <= 0) {
    return BatchNormStatsAscend::Call(input_tensor, eps);
  }
  auto x = ToContiguous(input_tensor, stream_id_);
  PyBoostUtils::PrepareOpInputs(device_context_, stream_id_, x);
  auto k = g_lazy_fusion_manager.Get(device_context_, stream_id_);
  CALL_START(op_name_, k);
  auto axis_ref = k->GetShapeRef(axis);
  auto input_obj = k->Input(x);
  float coef = 1.0f / static_cast<float>(count);

  // calc mean
  auto input_mean =
    k->Reduce(dvm::ReduceOpType::kSum, k->Binary(dvm::BinaryOpType::kMul, input_obj, coef), axis_ref, true);

  // calc invstd
  auto input_sub_mean = k->Binary(dvm::BinaryOpType::kSub, input_obj, input_mean);
  auto input_var = k->Binary(dvm::BinaryOpType::kMul, input_sub_mean, input_sub_mean);
  input_var = k->Reduce(dvm::ReduceOpType::kSum, k->Binary(dvm::BinaryOpType::kMul, input_var, coef), axis_ref, false);
  auto eps_imm = GetValue<float>(eps);
  auto invstd = k->Unary(dvm::UnaryOpType::kReciprocal,
                         k->Unary(dvm::UnaryOpType::kSqrt, k->Binary(dvm::BinaryOpType::kAdd, input_var, eps_imm)));

  auto out1 = k->Output(input_mean, x->data_type(), k->GetShape(invstd));
  auto out2 = k->Output(invstd, x->data_type(), k->GetShape(invstd));
  outputs_.push_back(out1);
  outputs_.push_back(out2);
  for (const auto &output : outputs_) {
    output->set_need_pipeline_sync(true);
  }
  CreateOutputSimpleInfo();
  CALL_END(op_name_, k);
  DumpOp(op_name_, this, input_tensor, eps);
  FlushLazyFusion();  // output shape reshape to 1D, must flush here
  return std::make_tuple(outputs_[kIndex0], outputs_[kIndex1]);
}

std::tuple<TensorPtr, TensorPtr> BatchNormGatherStatsWithCountsAscendDvm::Call(
  const TensorPtr &input_tensor, const TensorPtr &mean_tensor, const TensorPtr &invstd_tensor,
  const std::optional<TensorPtr> &running_mean_tensor_opt, const std::optional<TensorPtr> &running_var_tensor_opt,
  const mindspore::FP32ImmPtr &momentum, const mindspore::FP32ImmPtr &eps,
  const std::optional<TensorPtr> &counts_tensor_opt) {
  TensorPtr counts_tensor = counts_tensor_opt.has_value() ? counts_tensor_opt.value() : nullptr;
  TensorPtr running_mean_tensor = running_mean_tensor_opt.has_value() ? running_mean_tensor_opt.value() : nullptr;
  TensorPtr running_var_tensor = running_var_tensor_opt.has_value() ? running_var_tensor_opt.value() : nullptr;
  // input check
  if (NeedSync() || input_tensor->data_type() != kNumberTypeFloat32 || counts_tensor == nullptr) {
    return BatchNormGatherStatsWithCountsAscend::Call(input_tensor, mean_tensor, invstd_tensor, running_mean_tensor_opt,
                                                      running_var_tensor_opt, momentum, eps, counts_tensor_opt);
  }
  // running_mean and running_var must be contiguous, otherwise the update will has precision error
  if ((running_mean_tensor != nullptr && !running_mean_tensor->is_contiguous()) ||
      (running_var_tensor != nullptr && !running_var_tensor->is_contiguous())) {
    return BatchNormGatherStatsWithCountsAscend::Call(input_tensor, mean_tensor, invstd_tensor, running_mean_tensor_opt,
                                                      running_var_tensor_opt, momentum, eps, counts_tensor_opt);
  }
  auto x = ToContiguous(input_tensor, stream_id_);
  auto mean_all = ToContiguous(mean_tensor, stream_id_);
  auto invstd_all = ToContiguous(invstd_tensor, stream_id_);
  counts_tensor = ToContiguous(counts_tensor, stream_id_);
  FlushLazyFusion();  // forward fusion not allowed, because inputs need reshape
  auto momentum_imm = GetValue<float>(momentum);
  auto momentum_imm_reverse = 1.0f - momentum_imm;
  auto eps_imm = GetValue<float>(eps);
  PyBoostUtils::PrepareOpInputs(device_context_, stream_id_, x, mean_all, invstd_all, running_mean_tensor,
                                running_var_tensor, counts_tensor);
  auto k = g_lazy_fusion_manager.Get(device_context_, stream_id_);
  CALL_START(op_name_, k);

  // count shape 1D -> 2D
  ShapeVector count_shape(mean_all->shape());
  count_shape[count_shape.size() - 1] = 1;
  ShapeVector counts_axis;
  counts_axis.reserve(count_shape.size());
  for (int64_t i = 0; i < static_cast<int64_t>(count_shape.size()); ++i) {
    counts_axis.push_back(i);
  }
  auto count_axis_ref = k->GetShapeRef(counts_axis);

  ShapeVector mean_axis;
  mean_axis.reserve(mean_all->shape().size());
  for (int64_t i = 0; i < static_cast<int64_t>(mean_all->shape().size()) - 1; ++i) {  // last axis is C channel
    mean_axis.push_back(i);
  }
  auto mean_axis_ref = k->GetShapeRef(mean_axis);

  auto x_data_type = x->data_type();
  auto x_dtype = k->TransType(x_data_type);
  auto count_all_obj = k->Cast(k->Input(counts_tensor, true, count_shape), x_dtype);
  auto global_counts = k->Reduce(dvm::ReduceOpType::kSum, count_all_obj, count_axis_ref, false);

  // calc global mean
  auto mean_all_obj = k->Cast(k->Input(mean_all), x_dtype);
  auto global_mean = k->Reduce(
    dvm::ReduceOpType::kSum,
    k->Binary(dvm::BinaryOpType::kDiv, k->Binary(dvm::BinaryOpType::kMul, mean_all_obj, count_all_obj), global_counts),
    mean_axis_ref, false);
  auto global_mean_tensor = k->Output(global_mean, x_data_type, k->GetShape(global_mean));

  // calc global invstd
  auto mean_sub_all = k->Binary(dvm::BinaryOpType::kSub, mean_all_obj, global_mean);
  auto std_all = k->Unary(dvm::UnaryOpType::kReciprocal, k->Cast(k->Input(invstd_all), x_dtype));
  auto var_all = k->Binary(dvm::BinaryOpType::kAdd, k->Binary(dvm::BinaryOpType::kMul, std_all, std_all), -eps_imm);
  var_all = k->Binary(dvm::BinaryOpType::kAdd, var_all, k->Binary(dvm::BinaryOpType::kMul, mean_sub_all, mean_sub_all));
  var_all = k->Binary(dvm::BinaryOpType::kMul, var_all, count_all_obj);
  auto global_var_sum = k->Reduce(dvm::ReduceOpType::kSum, var_all, mean_axis_ref, false);
  auto global_var = k->Binary(dvm::BinaryOpType::kDiv, global_var_sum, global_counts);
  auto global_invstd =
    k->Unary(dvm::UnaryOpType::kReciprocal,
             k->Unary(dvm::UnaryOpType::kSqrt, k->Binary(dvm::BinaryOpType::kAdd, global_var, eps_imm)));
  auto global_invstd_tensor = k->Output(global_invstd, x_data_type, k->GetShape(global_invstd));

  // update running_mean
  if (running_mean_tensor != nullptr) {
    auto running_mean_new = k->Binary(
      dvm::BinaryOpType::kAdd,
      k->Binary(dvm::BinaryOpType::kMul, k->Cast(k->Input(running_mean_tensor), x_dtype), momentum_imm_reverse),
      k->Binary(dvm::BinaryOpType::kMul, global_mean, momentum_imm));
    k->Output(running_mean_tensor, running_mean_new, true);
  }

  // update running_var
  if (running_var_tensor != nullptr) {
    auto unbiased_global_var =
      k->Binary(dvm::BinaryOpType::kDiv, global_var_sum, k->Binary(dvm::BinaryOpType::kAdd, global_counts, -1.0f));
    auto running_var_new = k->Binary(
      dvm::BinaryOpType::kAdd,
      k->Binary(dvm::BinaryOpType::kMul, k->Cast(k->Input(running_var_tensor), x_dtype), momentum_imm_reverse),
      k->Binary(dvm::BinaryOpType::kMul, unbiased_global_var, momentum_imm));
    k->Output(running_var_tensor, running_var_new, true);
  }
  outputs_.push_back(global_mean_tensor);
  outputs_.push_back(global_invstd_tensor);
  for (const auto &output : outputs_) {
    output->set_need_pipeline_sync(true);
  }
  CreateOutputSimpleInfo();
  CALL_END(op_name_, k);
  DumpOp(op_name_, this, input_tensor, mean_tensor, invstd_tensor, running_mean_tensor_opt, running_var_tensor_opt,
         momentum, eps, counts_tensor_opt);
  FlushLazyFusion();
  return std::make_tuple(outputs_[kIndex0], outputs_[kIndex1]);
}

TensorPtr BatchNormElemtAscendDvm::Call(const TensorPtr &input_tensor,
                                        const std::optional<TensorPtr> &weight_tensor_opt,
                                        const std::optional<TensorPtr> &bias_tensor_opt,
                                        const std::optional<TensorPtr> &mean_tensor_opt,
                                        const std::optional<TensorPtr> &invstd_tensor_opt,
                                        const mindspore::FP32ImmPtr &eps) {
  TensorPtr weight_tensor = weight_tensor_opt.has_value() ? weight_tensor_opt.value() : nullptr;
  TensorPtr bias_tensor = bias_tensor_opt.has_value() ? bias_tensor_opt.value() : nullptr;
  TensorPtr mean_tensor = mean_tensor_opt.has_value() ? mean_tensor_opt.value() : nullptr;
  TensorPtr invstd_tensor = invstd_tensor_opt.has_value() ? invstd_tensor_opt.value() : nullptr;
  // input check
  if (NeedSync() || input_tensor->data_type() != kNumberTypeFloat32) {
    return BatchNormElemtAscend::Call(input_tensor, weight_tensor_opt, bias_tensor_opt, mean_tensor_opt,
                                      invstd_tensor_opt, eps);
  }
  auto x = ToContiguous(input_tensor, stream_id_);
  if (weight_tensor != nullptr) {
    weight_tensor = ToContiguous(weight_tensor, stream_id_);
  }
  if (bias_tensor != nullptr) {
    bias_tensor = ToContiguous(bias_tensor, stream_id_);
  }
  if (mean_tensor != nullptr) {
    mean_tensor = ToContiguous(mean_tensor, stream_id_);
  }
  if (invstd_tensor != nullptr) {
    invstd_tensor = ToContiguous(invstd_tensor, stream_id_);
  }
  FlushLazyFusion();  // forward fusion not allowed, because inputs need reshape
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      // (x - mean) * invstd * weight + bias
      auto input_obj = k->Input(x);
      auto input_dtype = k->GetDType(input_obj);
      ShapeVector new_shape(x->shape().size(), 1);
      new_shape[1] = x->shape()[1];
      if (mean_tensor != nullptr) {
        input_obj =
          k->Binary(dvm::BinaryOpType::kSub, input_obj, k->Cast(k->Input(mean_tensor, true, new_shape), input_dtype));
      }
      if (invstd_tensor != nullptr) {
        input_obj =
          k->Binary(dvm::BinaryOpType::kMul, input_obj, k->Cast(k->Input(invstd_tensor, true, new_shape), input_dtype));
      }
      if (weight_tensor != nullptr) {
        input_obj =
          k->Binary(dvm::BinaryOpType::kMul, input_obj, k->Cast(k->Input(weight_tensor, true, new_shape), input_dtype));
      }
      if (bias_tensor != nullptr) {
        input_obj =
          k->Binary(dvm::BinaryOpType::kAdd, input_obj, k->Cast(k->Input(bias_tensor, true, new_shape), input_dtype));
      }
      return k->Output(input_obj, x->data_type(), x->shape());
    },
    x, weight_tensor, bias_tensor, mean_tensor, invstd_tensor);
  DumpOp(op_name_, this, input_tensor, weight_tensor_opt, bias_tensor_opt, mean_tensor_opt, invstd_tensor_opt, eps);
  return outputs_.front();
}

TensorPtr BatchNormElemtGradAscendDvm::Call(const TensorPtr &dout_tensor, const TensorPtr &input_tensor,
                                            const TensorPtr &mean_tensor, const TensorPtr &invstd_tensor,
                                            const TensorPtr &weight_tensor, const TensorPtr &sumd_dy_tensor,
                                            const TensorPtr &sum_dy_xmu_tensor, const TensorPtr &count_tensor) {
  // input check
  if (NeedSync() || input_tensor->data_type() != kNumberTypeFloat32) {
    return BatchNormElemtGradAscend::Call(dout_tensor, input_tensor, mean_tensor, invstd_tensor, weight_tensor,
                                          sumd_dy_tensor, sum_dy_xmu_tensor, count_tensor);
  }
  auto dout_tensor_c = ToContiguous(dout_tensor, stream_id_);
  auto input_tensor_c = ToContiguous(input_tensor, stream_id_);
  auto mean_tensor_c = ToContiguous(mean_tensor, stream_id_);
  auto invstd_tensor_c = ToContiguous(invstd_tensor, stream_id_);
  auto weight_tensor_c = ToContiguous(weight_tensor, stream_id_);
  auto sumd_dy_tensor_c = ToContiguous(sumd_dy_tensor, stream_id_);
  auto sum_dy_xmu_tensor_c = ToContiguous(sum_dy_xmu_tensor, stream_id_);
  auto count_tensor_c = ToContiguous(count_tensor, stream_id_);
  FlushLazyFusion();  // forward fusion not allowed, because inputs need reshape
  DvmCall(
    op_name_, this,
    [&](LazyFusionKernelAscend *k) -> TensorPtr {
      auto x_obj = k->Input(input_tensor_c);
      auto x_dtype = k->GetDType(x_obj);
      ShapeVector new_shape(input_tensor_c->shape().size(), 1);
      new_shape[1] = input_tensor_c->shape()[1];
      ShapeVector counts_axis;
      counts_axis.reserve(count_tensor_c->shape().size());
      for (int64_t i = 0; i < static_cast<int64_t>(count_tensor_c->shape().size()); ++i) {
        counts_axis.push_back(i);
      }
      auto count_axis_ref = k->GetShapeRef(counts_axis);
      auto global_counts =
        k->Reduce(dvm::ReduceOpType::kSum, k->Cast(k->Input(count_tensor_c), x_dtype), count_axis_ref, false);
      auto invstd_obj = k->Cast(k->Input(invstd_tensor_c, true, new_shape), x_dtype);
      auto invstd_dy_xmu =
        k->Binary(dvm::BinaryOpType::kMul, k->Binary(dvm::BinaryOpType::kMul, invstd_obj, invstd_obj),
                  k->Binary(dvm::BinaryOpType::kDiv, k->Cast(k->Input(sum_dy_xmu_tensor_c, true, new_shape), x_dtype),
                            global_counts));
      auto x_sub_mean =
        k->Binary(dvm::BinaryOpType::kSub, x_obj, k->Cast(k->Input(mean_tensor_c, true, new_shape), x_dtype));
      auto x_invstd = k->Binary(dvm::BinaryOpType::kMul, x_sub_mean, invstd_dy_xmu);
      auto t1 = k->Binary(dvm::BinaryOpType::kSub, k->Cast(k->Input(dout_tensor_c), x_dtype),
                          k->Binary(dvm::BinaryOpType::kDiv,
                                    k->Cast(k->Input(sumd_dy_tensor_c, true, new_shape), x_dtype), global_counts));
      auto t2 = k->Binary(dvm::BinaryOpType::kSub, t1, x_invstd);
      auto obj = k->Binary(
        dvm::BinaryOpType::kMul, t2,
        k->Binary(dvm::BinaryOpType::kMul, invstd_obj, k->Cast(k->Input(weight_tensor_c, true, new_shape), x_dtype)));
      return k->Output(obj, input_tensor_c->data_type(), input_tensor_c->shape());
    },
    dout_tensor_c, input_tensor_c, mean_tensor_c, invstd_tensor_c, weight_tensor_c, sumd_dy_tensor_c,
    sum_dy_xmu_tensor_c, count_tensor_c);
  DumpOp(op_name_, this, dout_tensor, input_tensor, mean_tensor, invstd_tensor, weight_tensor, sumd_dy_tensor,
         sum_dy_xmu_tensor, count_tensor);
  return outputs_.front();
}

#define MS_REPLACE_DVM_OP(op_level, clazz)                                                                         \
  if (EnableFuse(#clazz, op_level)) {                                                                              \
    MS_LOG(INFO) << "Register dvm op [" << #clazz << "]";                                                          \
    OpFactory<clazz>::Get().op_creator()[device::DeviceType::kAscend] = []() {                                     \
      return std::make_shared<clazz##AscendDvm>(prim::kPrim##clazz,                                                \
                                                runtime::OpRunner::GetDeviceContext(device::DeviceType::kAscend)); \
    };                                                                                                             \
  }

void RegisterLazyFusionOp() {
  MS_REPLACE_DVM_OP(OpLevel_0, Concat);
  MS_REPLACE_DVM_OP(OpLevel_0, Cast);
  MS_REPLACE_DVM_OP(OpLevel_0, Abs);
  MS_REPLACE_DVM_OP(OpLevel_0, Neg);
  MS_REPLACE_DVM_OP(OpLevel_0, Exp);
  MS_REPLACE_DVM_OP(OpLevel_0, Sqrt);
  MS_REPLACE_DVM_OP(OpLevel_0, Reciprocal);
  MS_REPLACE_DVM_OP(OpLevel_0, IsFinite);
  MS_REPLACE_DVM_OP(OpLevel_0, Round);
  MS_REPLACE_DVM_OP(OpLevel_0, Ceil);
  MS_REPLACE_DVM_OP(OpLevel_0, Floor);
  MS_REPLACE_DVM_OP(OpLevel_0, Trunc);
  MS_REPLACE_DVM_OP(OpLevel_0, Equal);
  MS_REPLACE_DVM_OP(OpLevel_0, NotEqual);
  MS_REPLACE_DVM_OP(OpLevel_0, Greater);
  MS_REPLACE_DVM_OP(OpLevel_0, GreaterEqual);
  MS_REPLACE_DVM_OP(OpLevel_0, Less);
  MS_REPLACE_DVM_OP(OpLevel_0, LessEqual);
  MS_REPLACE_DVM_OP(OpLevel_0, Add);
  MS_REPLACE_DVM_OP(OpLevel_0, Mul);
  MS_REPLACE_DVM_OP(OpLevel_0, Muls);
  MS_REPLACE_DVM_OP(OpLevel_0, Sub);
  MS_REPLACE_DVM_OP(OpLevel_0, Div);
  MS_REPLACE_DVM_OP(OpLevel_0, Pow);
  MS_REPLACE_DVM_OP(OpLevel_0, Maximum);
  MS_REPLACE_DVM_OP(OpLevel_0, Minimum);
  MS_REPLACE_DVM_OP(OpLevel_0, LogicalNot);
  MS_REPLACE_DVM_OP(OpLevel_0, LogicalAnd);
  MS_REPLACE_DVM_OP(OpLevel_0, LogicalOr);
  MS_REPLACE_DVM_OP(OpLevel_0, Sigmoid);
  MS_REPLACE_DVM_OP(OpLevel_0, SigmoidGrad);
  MS_REPLACE_DVM_OP(OpLevel_0, SiLU);
  MS_REPLACE_DVM_OP(OpLevel_0, SiLUGrad);
  MS_REPLACE_DVM_OP(OpLevel_0, GeLU);
  MS_REPLACE_DVM_OP(OpLevel_0, GeLUGrad);
  MS_REPLACE_DVM_OP(OpLevel_0, ReLU);
  MS_REPLACE_DVM_OP(OpLevel_0, SumExt);
  MS_REPLACE_DVM_OP(OpLevel_0, AddExt);
  MS_REPLACE_DVM_OP(OpLevel_0, SubExt);
  MS_REPLACE_DVM_OP(OpLevel_0, Tile);
  MS_REPLACE_DVM_OP(OpLevel_0, LinalgVectorNorm);
  MS_REPLACE_DVM_OP(OpLevel_0, InplaceDiv);
  MS_REPLACE_DVM_OP(OpLevel_0, InplaceExp);
  MS_REPLACE_DVM_OP(OpLevel_0, InplaceAddExt);
  MS_REPLACE_DVM_OP(OpLevel_0, InplaceSubExt);
  MS_REPLACE_DVM_OP(OpLevel_0, InplaceReLU);
  MS_REPLACE_DVM_OP(OpLevel_1, Dense);
  MS_REPLACE_DVM_OP(OpLevel_1, MatMul);
  MS_REPLACE_DVM_OP(OpLevel_1, BatchMatMul);
  MS_REPLACE_DVM_OP(OpLevel_1, BatchMatMulExt);
  MS_REPLACE_DVM_OP(OpLevel_1, MatMulExt);
  MS_REPLACE_DVM_OP(OpLevel_0, BatchNormStats);
  MS_REPLACE_DVM_OP(OpLevel_0, BatchNormGatherStatsWithCounts);
  MS_REPLACE_DVM_OP(OpLevel_0, BatchNormElemt);
  MS_REPLACE_DVM_OP(OpLevel_0, BatchNormElemtGrad);
}

void LazyFusionAscendInit() {
  if (LazyFusionFlags::GetInstance().opt_level < OptLevel_1 || runtime::RuntimeConf::GetInstance()->launch_blocking()) {
    MS_LOG(INFO) << "Skip init lazy fusion.";
    return;
  }
  MS_LOG(INFO) << "Init lazy fusion.";
  RegisterLazyFusionOp();
  runtime::Pipeline::Get().UpdateBackendStage(
    std::make_unique<LazyFusionQueue>("backend_queue", runtime::kThreadWaitLevel::kLevelBackend));
}

MS_REGISTER_LAZY_FUSION_INIT(kAscendDevice, LazyFusionAscendInit);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
