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

#include "src/litert/delegate/npu/npu_converter_utils.h"
#include "src/litert/delegate/npu/op/npu_op.h"
namespace mindspore::lite {
#define C4NUM 4
#define C8NUM 8
#ifdef ENABLE_ARM
void Float32ToFloat16(const float *__restrict input, float16_t *__restrict output, int number) {
  int i = 0;
#ifdef ENABLE_ARM64
  int count = (number & ~(C8NUM - 1));
  for (; i < count; i += C8NUM) {
    float32x4_t in1 = vld1q_f32(input + i);
    float16x4_t out1 = vcvt_f16_f32(in1);
    float32x4_t in2 = vld1q_f32(input + i + C4NUM);
    float16x4_t out2 = vcvt_f16_f32(in2);
    float16x8_t out = vcombine_f16(out1, out2);
    vst1q_f16(output + i, out);
  }
#endif
  for (; i < number; ++i) {
    output[i] = static_cast<float16_t>(input[i]);
  }
}

void Float16ToFloat32(const float16_t *__restrict input, float *__restrict output, int number) {
  int i = 0;
#ifdef ENABLE_ARM64
  int count = number & ~(C8NUM - 1);
  for (; i < count; i += C8NUM) {
    float16x8_t in = vld1q_f16(input + i);
    float16x4_t in1 = vget_low_f16(in);
    float16x4_t in2 = vget_high_f16(in);
    float32x4_t out1 = vcvt_f32_f16(in1);
    vst1q_f32(output + i, out1);
    float32x4_t out2 = vcvt_f32_f16(in2);
    vst1q_f32(output + i + C4NUM, out2);
  }
#endif
  for (; i < number; ++i) {
    output[i] = static_cast<float>(input[i]);
  }
}
#endif

ge::Shape ConverterToNPUShape(const std::vector<int64_t> &src_shape, bool is_expand_4d) {
  std::vector<int64_t> shapes;
  shapes.reserve(src_shape.size());
  for (int i = 0; i < src_shape.size(); i++) {
    shapes.push_back(src_shape[i]);
  }
  if (is_expand_4d) {
    if (shapes.size() == 1) {
      return ge::Shape({1, shapes[0], 1, 1});
    } else {
      for (int i = src_shape.size(); i < NPU_SHAPE_SIZE; i++) {
        shapes.push_back(1);
      }
    }
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

ge::DataType ConverterToNPUDataType(DataType type_id) {
  ge::DataType data_type;
  switch (type_id) {
    case DataType::kNumberTypeFloat32:
    case DataType::kNumberTypeFloat16:
      data_type = ge::DT_FLOAT;
      break;
    case DataType::kNumberTypeInt8:
      data_type = ge::DT_INT8;
      break;
    case DataType::kNumberTypeUInt8:
      data_type = ge::DT_UINT8;
      break;
    case DataType::kNumberTypeInt16:
      data_type = ge::DT_INT16;
      break;
    case DataType::kNumberTypeInt32:
      data_type = ge::DT_INT32;
      break;
    case DataType::kNumberTypeUInt32:
      data_type = ge::DT_UINT32;
      break;
    case DataType::kNumberTypeBool:
      data_type = ge::DT_BOOL;
      break;
    default:
      data_type = ge::DT_UNDEFINED;
      break;
  }
  return data_type;
}

hiai::op::Data *ConverterToNPUData(const mindspore::MSTensor &src, const std::string &name) {
  auto data = new (std::nothrow) hiai::op::Data(name);
  if (data == nullptr) {
    MS_LOG(ERROR) << "new data failed.";
    return data;
  }
  ge::TensorDesc tensor_desc(ConverterToNPUShape(src.Shape()), ge::FORMAT_NCHW, ConverterToNPUDataType(src.DataType()));
  data->update_input_desc_x(tensor_desc);
  return data;
}

std::shared_ptr<ge::Tensor> ConverterToNPUTensor(mindspore::MSTensor src, bool is_expand_4d) {
  std::shared_ptr<ge::Tensor> ge_tensor = std::make_shared<ge::Tensor>();
  if (ge_tensor == nullptr) {
    MS_LOG(ERROR) << "new ge_tensor failed.";
    return nullptr;
  }
  ge::TensorDesc tensor_desc(ConverterToNPUShape(src.Shape(), is_expand_4d), ge::FORMAT_NCHW,
                             ConverterToNPUDataType(src.DataType()));

  ge_tensor->SetTensorDesc(tensor_desc);

  if (src.Data() != nullptr) {
    if (src.DataType() == DataType::kNumberTypeFloat16) {
#ifdef ENABLE_ARM
      auto fp32_data = malloc(src.ElementNum() * sizeof(float));
      if (fp32_data == nullptr) {
        MS_LOG(ERROR) << "malloc failed for fp32 data";
        return nullptr;
      }
      Float16ToFloat32(reinterpret_cast<float16_t *>(src.MutableData()), reinterpret_cast<float *>(fp32_data),
                       src.ElementNum());
      ge_tensor->SetData(reinterpret_cast<const uint8_t *>(fp32_data), src.ElementNum() * sizeof(float));
      free(fp32_data);
      fp32_data = nullptr;
#else
      MS_LOG(ERROR) << "This platform does not support fp16.";
      return nullptr;
#endif
    } else {
      ge_tensor->SetData(reinterpret_cast<const uint8_t *>(src.MutableData()), src.DataSize());
    }
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

int TransFormAxis(int axis) {
  switch (axis) {
    case NHWC_N:
      return NCHW_N;
    case NHWC_H:
      return NCHW_H;
    case NHWC_W:
      return NCHW_W;
    case NHWC_C:
      return NCHW_C;
    default:
      return NCHW_INVALID;
  }
}

int ConverterToNPUActivationMode(schema::ActivationType type) {
  switch (type) {
    case schema::ActivationType_SIGMOID:
      return SIGMOID;
    case schema::ActivationType_RELU:
      return RELU;
    case schema::ActivationType_TANH:
      return TANH;
    case schema::ActivationType_LEAKY_RELU:
      return P_RELU;
    case schema::ActivationType_HSIGMOID:
      return HARD_SIGMOID;
    case schema::ActivationType_RELU6:
      return RELU6;
    case schema::ActivationType_ELU:
      return ELU;
    case schema::ActivationType_GELU:
      return GELU;
    default:
      return ACTIVATION_INVALID;
  }
}
}  // namespace mindspore::lite
