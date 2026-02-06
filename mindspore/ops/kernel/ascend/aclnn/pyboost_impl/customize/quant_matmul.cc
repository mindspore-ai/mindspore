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

#include "kernel/ascend/aclnn/pyboost_impl/customize/quant_matmul.h"
#include <string>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
int64_t CheckAndGetGroups(const std::vector<int64_t> &group_sizes_list) {
  int64_t groups = 0;
  if (group_sizes_list.empty()) {
    return groups;
  }
  constexpr size_t kGroupSizeLen = 3;
  if (group_sizes_list.size() != kGroupSizeLen) {
    MS_EXCEPTION(ValueError) << "For QuantMatmul, group_sizes should contain three elements, but got "
                             << group_sizes_list;
  }
  auto group_m = group_sizes_list[kIndex0];
  auto group_n = group_sizes_list[kIndex1];
  auto group_k = group_sizes_list[kIndex2];
  constexpr int64_t group_max = 65535LL;
  if (group_m > group_max || group_n > group_max || group_k > group_max) {
    MS_EXCEPTION(ValueError) << "For QuantMatmul, all elements of group_sizes can't be larger than 65535, but got "
                             << group_sizes_list;
  }
  if (group_m < 0 || group_n < 0 || group_k < 0) {
    MS_EXCEPTION(ValueError) << "For QuantMatmul, all elements of group_sizes can't be less than 0, but got "
                             << group_sizes_list;
  }
  constexpr auto kGroupmBitOffset = 32;
  constexpr auto kGroupnBitOffset = 16;

  groups = static_cast<int64_t>((static_cast<uint64_t>(group_m) << kGroupmBitOffset) +
                                (static_cast<uint64_t>(group_n) << kGroupnBitOffset) + static_cast<uint64_t>(group_k));
  return groups;
}

TensorPtr QuantMatmulFastEmpty(const ShapeVector &shape, const TypeId &type, const size_t &stream_id) {
  device::DeviceType device_name = device::DeviceType::kAscend;

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto default_device_name = DeviceManagerConf::GetInstance()->device_type();
  if (default_device_name != device_name) {
    device_name = default_device_name;
    MS_LOG(DEBUG) << "Using default device_name: " << default_device_name;
  }

  auto device_ctx = runtime::OpRunner::GetDeviceContext(device_name);
  MS_EXCEPTION_IF_NULL(device_ctx);

  std::vector<tensor::TensorPtr> outputs;
  kernel::pyboost::PyBoostUtils::CreateOutputTensor(type, shape, &outputs);
  kernel::pyboost::PyBoostUtils::PrepareOpOutputs(device_ctx, stream_id, outputs);
  auto fn = [device_ctx, outputs]() { kernel::pyboost::PyBoostUtils::MallocOpOutputs(device_ctx, outputs); };

  if (!runtime::OpExecutor::NeedSync()) {
    runtime::OpExecutor::GetInstance().PushSimpleOpRunTask(std::make_shared<runtime::PassthroughNoWaitDeviceTask>(fn));
  } else {
    fn();
  }

  return outputs.at(kIndex0);
}
}  // namespace

void QuantMatmulAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &x1, const TensorPtr &x2,
                                const TensorPtr &scale, const std::optional<TensorPtr> &offset,
                                const std::optional<TensorPtr> &pertoken_scale, const std::optional<TensorPtr> &bias,
                                const std::optional<Int64ImmPtr> &output_dtype,
                                const std::optional<Int64ImmPtr> &x1_dtype, const std::optional<Int64ImmPtr> &x2_dtype,
                                const std::optional<Int64ImmPtr> &pertoken_scale_dtype,
                                const std::optional<Int64ImmPtr> &scale_dtype,
                                const std::optional<ValueTuplePtr> &group_sizes) {
  OpRunner::InferOpOutput(op, x1, x2, scale, offset, pertoken_scale, bias, output_dtype, x1_dtype, x2_dtype,
                          pertoken_scale_dtype, scale_dtype, group_sizes);
  TensorPtr x1_val = x1;
  if (x1_dtype.has_value()) {
    x1_val = PyBoostUtils::CastTensor(x1_val, static_cast<TypeId>(x1_dtype.value()->value()),
                                      op->device_context()->device_context_key_.device_type_);
  }
  TensorPtr x2_val = x2;
  if (x2_dtype.has_value()) {
    x2_val = PyBoostUtils::CastTensor(x2_val, static_cast<TypeId>(x2_dtype.value()->value()),
                                      op->device_context()->device_context_key_.device_type_);
  }
  auto output_dtype_id = op->output_value_simple_info()->dtype_vector_[kIndex0]->type_id();
  auto y_scale = QuantMatmulFastEmpty({0}, output_dtype_id, op->stream_id());
  auto x1_offset = QuantMatmulFastEmpty({0}, output_dtype_id, op->stream_id());
  auto y_offset = QuantMatmulFastEmpty({0}, output_dtype_id, op->stream_id());
  auto transpose1 = false;
  auto transpose2 = false;
  auto group_sizes_list = ConvertValueTupleToVector<int64_t>(group_sizes);
  auto group_size_val = CheckAndGetGroups(group_sizes_list);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x1_val, x2_val, scale, offset, pertoken_scale,
                                bias);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  MS_LOG(DEBUG) << "QuantMatmul x1_val.shape = " << ShapeVectorToStr(x1_val->shape())
                << ", dtype = " << TypeIdToString(x1_val->Dtype()->type_id())
                << "; x2_val.shape = " << ShapeVectorToStr(x2_val->shape())
                << ", dtype = " << TypeIdToString(x2_val->Dtype()->type_id())
                << "; scale.shape = " << ShapeVectorToStr(scale->shape())
                << ", dtype = " << TypeIdToString(scale->Dtype()->type_id());

  if (scale->data_type() == kNumberTypeFloat32 && !pertoken_scale.has_value() && output_dtype_id != kNumberTypeInt32) {
    // quant_param path
    auto scale_shape = scale->shape();
    auto quant_param_shape = scale_shape;
    if (quant_param_shape.size() == 1 && offset.has_value()) {
      auto offset_shape = offset.value()->shape();
      quant_param_shape = scale_shape.at(kIndex0) > offset_shape.at(kIndex0) ? scale_shape : offset_shape;
    }
    auto quant_param = QuantMatmulFastEmpty(quant_param_shape, kNumberTypeInt64, op->stream_id());
    PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
      [op, x1_val, x2_val, pertoken_scale, scale, y_scale, x1_offset, offset, y_offset, bias, transpose1, transpose2,
       group_size_val, quant_param]() {
        MS_LOG(DEBUG) << "Run device task QuantMatmul start with quant_param";
        auto device_context = op->device_context();
        const auto &outputs = op->outputs();
        // Malloc for input tensors
        PyBoostUtils::MallocOpInputs(device_context, x1_val, x2_val, pertoken_scale, scale, offset);
        // Malloc for output tensors
        PyBoostUtils::MallocOpOutputs(device_context, outputs);
        LAUNCH_ACLNN(aclnnTransQuantParamV2, device_context, op->stream_id(), scale, offset, quant_param);

        LAUNCH_ACLNN(aclnnQuantMatmulV5, device_context, op->stream_id(), x1_val, x2_val, pertoken_scale, quant_param,
                     y_scale, x1_offset, offset, y_offset, bias, transpose1, transpose2, group_size_val,
                     outputs[kIndex0]);
        MS_LOG(DEBUG) << "Run device task QuantMatmul end with quant_param";
      }));
  } else {
    PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, x1_val, x2_val, pertoken_scale, scale,
                                                                            y_scale, x1_offset, offset, y_offset, bias,
                                                                            transpose1, transpose2, group_size_val]() {
      MS_LOG(DEBUG) << "Run device task QuantMatmul start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, x1_val, x2_val, pertoken_scale, scale, offset);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      LAUNCH_ACLNN(aclnnQuantMatmulV5, device_context, op->stream_id(), x1_val, x2_val, pertoken_scale, scale, y_scale,
                   x1_offset, offset, y_offset, bias, transpose1, transpose2, group_size_val, outputs[kIndex0]);
      MS_LOG(DEBUG) << "Run device task QuantMatmul end";
    }));
  }
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
