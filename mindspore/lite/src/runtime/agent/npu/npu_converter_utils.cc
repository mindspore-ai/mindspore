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

#include "src/runtime/agent/npu/npu_converter_utils.h"
namespace mindspore::lite {
ge::Shape ConverterToNPUShape(const std::vector<int> &src_shape) {
  vector<int64_t> shapes;
  shapes.reserve(src_shape.size());
  for (int i = 0; i < src_shape.size(); i++) {
    shapes.push_back(src_shape[i]);
  }
  return ge::Shape({shapes});
}

ge::Format ConverterToNPUFormat(schema::Format format) {
  ge::Format ge_format;
  switch (format) {
    case schema::Format_NCHW:
      ge_format = ge::FORMAT_NCHW;
      break;
    case schema::Format_NHWC:
    case schema::Format_KHWC:
      ge_format = ge::FORMAT_NHWC;
      break;
    default:
      MS_LOG(ERROR) << "Unsupported format:" << format;
      // use unused format to indicate errors.
      ge_format = ge::FORMAT_ND;
      break;
  }
  return ge_format;
}

ge::DataType ConverterToNPUDataType(TypeId type_id) {
  ge::DataType data_type;
  switch (type_id) {
    case kNumberTypeFloat:
    case kNumberTypeFloat32:
      data_type = ge::DT_FLOAT;
      break;
    case kNumberTypeFloat16:
      data_type = ge::DT_FLOAT16;
      break;
    case kNumberTypeInt8:
      data_type = ge::DT_INT8;
      break;
    case kNumberTypeUInt8:
      data_type = ge::DT_UINT8;
      break;
    case kNumberTypeInt16:
      data_type = ge::DT_INT16;
      break;
    case kNumberTypeInt32:
      data_type = ge::DT_INT32;
      break;
    case kNumberTypeUInt32:
      data_type = ge::DT_UINT32;
      break;
    default:
      data_type = ge::DT_UNDEFINED;
      break;
  }
  return data_type;
}

hiai::op::Data *ConverterToNPUData(Tensor *src, const std::string &name) {
  auto data = new (std::nothrow) hiai::op::Data(name);
  if (data == nullptr) {
    MS_LOG(ERROR) << "new data failed.";
    return data;
  }
  ge::TensorDesc tensor_desc(ConverterToNPUShape(src->shape()), ConverterToNPUFormat(src->format()),
                             ConverterToNPUDataType(src->data_type()));
  data->update_input_desc_x(tensor_desc);
  return data;
}

std::shared_ptr<ge::Tensor> ConverterToNPUTensor(Tensor *src) {
  std::shared_ptr<ge::Tensor> ge_tensor = std::shared_ptr<ge::Tensor>(new (std::nothrow) ge::Tensor());
  if (ge_tensor == nullptr) {
    MS_LOG(ERROR) << "new ge_tensor failed.";
    return ge_tensor;
  }
  ge::TensorDesc tensor_desc(ConverterToNPUShape(src->shape()), ConverterToNPUFormat(src->format()),
                             ConverterToNPUDataType(src->data_type()));

  ge_tensor->SetTensorDesc(tensor_desc);

  if (src->data_c() != nullptr) {
    ge_tensor->SetData(reinterpret_cast<const uint8_t *>(src->data_c()), src->Size());
  }
  return ge_tensor;
}

// mode  : Either 0 (product), 1 (sum), 2 (max), 3 (mean). Defaults to 1 (sum).
int ConverterToNPUEltwiseMode(schema::EltwiseMode mode) {
  int mode_num = 1;
  switch (mode) {
    case schema::EltwiseMode_PROD:
      mode_num = 0;
      break;
    case schema::EltwiseMode_SUM:
      mode_num = 1;
      break;
    case schema::EltwiseMode_MAXIMUM:
      mode_num = 2;
      break;
    default:
      MS_LOG(ERROR) << "Unsupported Eltwise mode.";
  }
  return mode_num;
}
}  // namespace mindspore::lite
