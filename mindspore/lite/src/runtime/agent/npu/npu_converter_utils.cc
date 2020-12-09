/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
/*
 *    mode  : Activation mode, with options as follows:
 *            0 : Sigmoid
 *            1 : ReLU
 *            2 : Tanh
 *            3 : Clipped ReLU
 *            4 : ELU
 *            5 : PReLU
 *            6 : Abs
 *            7 : Relu1
 *            8 : Softsign
 *            9 : Softplus
 *            10 : Hardsigmoid
 *            11 : Threshold ReLU
 *            12 : Selu
 *            13 : Linear
 *            14 : Relu6
 *            15 : GeLU.
 */
int ConverterToNPUActMode(schema::ActivationType type) {
  switch (type) {
    case schema::ActivationType_NO_ACTIVATION:
      return -1;
    case schema::ActivationType_SIGMOID:
      return 0;
    case schema::ActivationType_RELU:
      return 1;
    case schema::ActivationType_TANH:
      return 2;
    case schema::ActivationType_ELU:
      return 4;
    case schema::ActivationType_LEAKY_RELU:
      return 5;
    case schema::ActivationType_ABS:
      return 6;
    case schema::ActivationType_RELU1:
      return 7;
    case schema::ActivationType_SOFTSIGN:
      return 8;
    case schema::ActivationType_SOFTPLUS:
      return 9;
    case schema::ActivationType_HSIGMOID:
      return 10;
    case schema::ActivationType_THRESHOLDRELU:
      return 11;
    case schema::ActivationType_SELU:
      return 12;
    case schema::ActivationType_LINEAR:
      return 13;
    case schema::ActivationType_RELU6:
      return 14;
    default:
      MS_LOG(ERROR) << "Unsupport activation type to NPU." << type;
      return -1;
  }
}
}  // namespace mindspore::lite
