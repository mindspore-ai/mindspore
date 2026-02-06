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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_index_fill_scalar.h"
#include <memory>
#include <string>
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

tensor::TensorPtr InplaceIndexFillScalarAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input,
                                                        const Int64ImmPtr &dim, const TensorPtr &index,
                                                        const ScalarPtr &value) {
  auto index_shape = index->shape();
  if (MS_UNLIKELY(index_shape.size() > 1)) {
    MS_LOG(EXCEPTION) << "For [" << op->primitive()->name() << "], the rank of input 'index'"
                      << " must be in [0, 1], but got " << index_shape.size() << ".";
  }
  auto dim_imm = GetValue<int64_t>(dim);
  std::vector<int64_t> index_vector;
  auto index_cpu = index->cpu();
  TypeId index_type_id = static_cast<TypeId>(index_cpu->data_type_c());
  size_t elem_num = index_cpu->DataSize();
  if (index_type_id == TypeId::kNumberTypeInt64) {
    int64_t *elem_ptr = static_cast<int64_t *>(index_cpu->data_c());
    for (size_t i = 0; i < elem_num; i++) {
      index_vector.push_back(elem_ptr[i]);
    }
  } else if (index_type_id == TypeId::kNumberTypeInt32) {
    int32_t *elem_ptr = static_cast<int32_t *>(index_cpu->data_c());
    for (size_t i = 0; i < elem_num; i++) {
      index_vector.push_back(elem_ptr[i]);
    }
  } else {
    MS_EXCEPTION(TypeError) << "For [" << op->primitive()->name() << "], the input 'index'"
                            << " for conversion to int array must be of type Int32 or Int64,"
                            << " but got " << TypeIdToString(index_type_id);
  }

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input);
  op->set_outputs({input});

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input, dim_imm, index_vector, value]() {
    MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
    auto device_context = op->device_context();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input);
    LAUNCH_ACLNN(aclnnInplaceIndexFillTensor, device_context, op->stream_id(), input, dim_imm, index_vector, value);
    MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
