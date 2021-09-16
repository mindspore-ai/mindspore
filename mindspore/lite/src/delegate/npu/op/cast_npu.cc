/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "src/delegate/npu/op/cast_npu.h"
#include "src/delegate/npu/npu_converter_utils.h"

namespace mindspore {
int CastNPUOp::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                         const std::vector<mindspore::MSTensor> &out_tensors) {
  CHECK_LESS_RETURN(in_tensors.size(), C2NUM);
  auto in_tensor = in_tensors[1];
  CHECK_NULL_RETURN(in_tensor);
  CHECK_NULL_RETURN(in_tensor.Data().get());

  if (in_tensors.size() >= C2NUM && in_tensor.ElementNum() == 1) {
    dst_type_ = reinterpret_cast<const int *>(in_tensor.Data().get())[0];
  } else {
    MS_LOG(WARNING) << "NPU dst dtype is attribute.";
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int CastNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                    const std::vector<mindspore::MSTensor> &out_tensors) {
  CHECK_LESS_RETURN(in_tensors.size(), 1);
  CHECK_NULL_RETURN(in_tensors[0]);
  CHECK_NULL_RETURN(cast_);

  cast_ = new (std::nothrow) hiai::op::CastT(name_);
  if (cast_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }
  cast_->set_attr_dst_dtype(ConverterToNPUDataType(static_cast<DataType>(dst_type_)));
  cast_->set_attr_src_dtype(ConverterToNPUDataType(static_cast<DataType>(in_tensors[0].DataType())));
  return RET_OK;
}

int CastNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                            const std::vector<mindspore::MSTensor> &out_tensors,
                            const std::vector<ge::Operator *> &npu_inputs) {
  CHECK_NULL_RETURN(cast_);
  cast_->set_input_x(*npu_inputs[0]);
  return RET_OK;
}

ge::Operator *CastNPUOp::GetNPUOp() { return this->cast_; }

CastNPUOp::~CastNPUOp() {
  if (cast_ != nullptr) {
    delete cast_;
    cast_ = nullptr;
  }
}
}  // namespace mindspore
