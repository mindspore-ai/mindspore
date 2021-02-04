/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "coder/utils/print_utils.h"

namespace mindspore::lite::micro {

std::string GetPrintFormat(const lite::Tensor *tensor) {
  switch (tensor->data_type()) {
    case kNumberTypeFloat: {
      return "%f";
    }
    case kNumberTypeInt8: {
      return "%c";
    }
    case kNumberTypeInt32: {
      return "%d";
    }
    case kNumberTypeUInt8: {
      return "%d";
    }
    case kNumberTypeInt16: {
      return "%d";
    }
    case kNumberTypeUInt32: {
      return "%ld";
    }
    case kNumberTypeInt64: {
      return "%l64d";
    }
    case kNumberTypeUInt16: {
      return "%f";
    }
    case kNumberTypeFloat16: {
      MS_LOG(WARNING) << "unsupported data type: kNumberTypeFloat16";
      return "float ";
    }
    default:
      MS_LOG(WARNING) << "unsupported data type: " << tensor->data_type();
      return "%d";
  }
}

template <typename T>
void PrintTensorData(const lite::Tensor *tensor, std::ofstream &of, const std::string &left = "\t") {
  const int NUM = 20;
  T *data = reinterpret_cast<T *>(tensor->data_c());
  of << "{\n" << left;
  int len = tensor->ElementsNum();
  if (typeid(T) == typeid(float)) {
    of.precision(kWeightPrecision);
    for (int i = 0; i < len - 1; ++i) {
      of << data[i] << ",";
      if (i % NUM == NUM - 1) {
        of << std::endl << left;
      }
    }
    if (len > 0) {
      of << data[len - 1];
    }
  } else {
    for (int i = 0; i < len - 1; ++i) {
      of << std::to_string(data[i]) << ",";
      if (i % NUM == NUM - 1) {
        of << std::endl << left;
      }
    }
    if (len > 0) {
      of << std::to_string(data[len - 1]);
    }
  }
  of << "\n" << left << "};\n\n";
}

void PrintTensor(const lite::Tensor *tensor, std::ofstream &weightOf, std::ofstream &hOf,
                 const std::string &tensorName) {
  switch (tensor->data_type()) {
    case kNumberTypeFloat: {
      weightOf << "const float " << tensorName << "[] = ";
      hOf << "extern const float " << tensorName << "[];\n";
      PrintTensorData<float>(tensor, weightOf);
      break;
    }
    case kNumberTypeFloat32: {
      weightOf << "const float " << tensorName << "[] = ";
      hOf << "extern const float " << tensorName << "[];\n";
      PrintTensorData<float>(tensor, weightOf);
      break;
    }
    case kNumberTypeInt8: {
      weightOf << "const signed char " << tensorName << "[] = ";
      hOf << "extern const signed char " << tensorName << "[];\n";
      PrintTensorData<char>(tensor, weightOf);
      break;
    }
    case kNumberTypeInt32: {
      weightOf << "const int " << tensorName << "[] = ";
      hOf << "extern const int " << tensorName << "[];\n";
      PrintTensorData<int>(tensor, weightOf);
      break;
    }
    case kNumberTypeUInt8: {
      weightOf << "const unsigned char " << tensorName << "[] = ";
      hOf << "extern  const unsigned char " << tensorName << "[];\n";
      PrintTensorData<unsigned char>(tensor, weightOf);
      break;
    }
    case kNumberTypeInt16: {
      weightOf << "const short " << tensorName << "[] = ";
      hOf << "extern const short " << tensorName << "[];\n";
      PrintTensorData<int16_t>(tensor, weightOf);
      break;
    }
    case kNumberTypeUInt32: {
      weightOf << "const unsigned int " << tensorName << "[] = ";
      hOf << "extern const unsigned int " << tensorName << "[];\n";
      PrintTensorData<unsigned int>(tensor, weightOf);
      break;
    }
    case kNumberTypeInt64: {
      weightOf << "const long " << tensorName << "[] = ";
      hOf << "extern const long " << tensorName << "[];\n";
      PrintTensorData<int64_t>(tensor, weightOf);
      break;
    }
    case kNumberTypeUInt16: {
      weightOf << "const unsigned short " << tensorName << "[] = ";
      hOf << "extern const unsigned short " << tensorName << "[];\n";
      PrintTensorData<uint16_t>(tensor, weightOf);
      break;
    }
    case kNumberTypeFloat16: {
      MS_LOG(WARNING) << "unsupported data type: kNumberTypeFloat16";
      break;
    }
    default:
      MS_LOG(WARNING) << "unsupported data type: " << tensor->data_type();
  }
}

void PrintTensorForNet(const lite::Tensor *tensor, std::ofstream &weightOf, std::ofstream &hOf,
                       const std::string &tensorName) {
  MS_LOG(DEBUG) << "PrintTensorForNet tensor dtype: " << tensor->data_type();
  switch (tensor->data_type()) {
    case kNumberTypeFloat: {
      weightOf << "float " << tensorName << "[" << tensor->ElementsNum() << "]={0};\n";
      hOf << "extern float " << tensorName << "[];\n";
      break;
    }
    case kNumberTypeFloat32: {
      weightOf << "float " << tensorName << "[" << tensor->ElementsNum() << "]={0};\n";
      hOf << "extern float " << tensorName << "[];\n";
      break;
    }

    case kNumberTypeInt8: {
      weightOf << "signed char " << tensorName << "[" << tensor->ElementsNum() << "]={0};\n";
      hOf << "extern signed char " << tensorName << "[];\n";
      break;
    }
    case kNumberTypeInt32: {
      weightOf << "int " << tensorName << "[" << tensor->ElementsNum() << "]={0};\n";
      hOf << "extern int " << tensorName << "[];\n";
      break;
    }
    case kNumberTypeUInt8: {
      weightOf << "unsigned char " << tensorName << "[" << tensor->ElementsNum() << "]={0};\n";
      hOf << "extern  unsigned char " << tensorName << "[];\n";
      break;
    }
    case kNumberTypeInt16: {
      weightOf << "short " << tensorName << "[" << tensor->ElementsNum() << "]={0};\n";
      hOf << "extern short " << tensorName << "[];\n";
      break;
    }
    case kNumberTypeUInt32: {
      weightOf << "unsigned int " << tensorName << "[" << tensor->ElementsNum() << "]={0};\n";
      hOf << "extern unsigned int " << tensorName << "[];\n";
      break;
    }
    case kNumberTypeInt64: {
      weightOf << "long " << tensorName << "[" << tensor->ElementsNum() << "]={0};\n";
      hOf << "extern long " << tensorName << "[];\n";
      break;
    }
    case kNumberTypeUInt16: {
      weightOf << "unsigned short " << tensorName << "[" << tensor->ElementsNum() << "]={0};\n";
      hOf << "extern unsigned short " << tensorName << "[];\n";
      break;
    }
    case kNumberTypeFloat16: {
      weightOf << "float " << tensorName << "[" << tensor->ElementsNum() << "]={0};\n";
      hOf << "extern float " << tensorName << "[];\n";
      break;
    }
    default:
      MS_LOG(WARNING) << "Default  DataType_DT not support. Tensor name: " << tensorName.c_str();
  }
}

std::string GetTensorDataType(const TypeId typeId) {
  switch (typeId) {
    case kNumberTypeFloat32: {
      return "float ";
    }
    case kNumberTypeFloat: {
      return "float ";
    }
    case kNumberTypeInt8: {
      return "char ";
    }
    case kNumberTypeInt: {
      return "int ";
    }
    case kNumberTypeInt32: {
      return "int ";
    }
    case kNumberTypeUInt8: {
      return "unsigned char ";
    }
    case kNumberTypeInt16: {
      return "short ";
    }
    case kNumberTypeUInt32: {
      return "unsigned int ";
    }
    case kNumberTypeInt64: {
      return "long ";
    }
    case kNumberTypeUInt16: {
      return "unsigned short ";
    }
    case kNumberTypeFloat16: {
      MS_LOG(WARNING) << "unsupported data type: kNumberTypeFloat16";
      return "float ";
    }
    default:
      MS_LOG(WARNING) << "unsupported data type: " << typeId;
      return "int";
  }
}

std::string GetMicroTensorDataType(TypeId type) {
  switch (type) {
    case kNumberTypeFloat:
    case kNumberTypeFloat32: {
      return "DataType_DT_FLOAT";
    }
    case kNumberTypeInt8: {
      return "DataType_DT_INT8";
    }
    case kNumberTypeInt:
    case kNumberTypeInt32: {
      return "DataType_DT_INT32";
    }
    case kNumberTypeUInt8: {
      return "DataType_DT_UINT8";
    }
    case kNumberTypeInt16: {
      return "DataType_DT_INT16";
    }
    case kNumberTypeUInt32: {
      return "DataType_DT_UINT32";
    }
    case kNumberTypeInt64: {
      return "DataType_DT_INT64";
    }
    case kNumberTypeUInt16: {
      return "DataType_DT_UINT16";
    }
    case kNumberTypeFloat16: {
      MS_LOG(WARNING) << "unsupported data type: kNumberTypeFloat16";
      return "DataType_DT_FLOAT16";
    }
    default:
      MS_LOG(WARNING) << "unsupported data type: " << type << ", reference: " << kNumberTypeInt;
      return "DataType_DT_UNDEFINED";
  }
}

}  // namespace mindspore::lite::micro
