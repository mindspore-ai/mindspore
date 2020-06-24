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
#include <grpcpp/grpcpp.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "./ms_service.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using ms_serving::MSService;
using ms_serving::PredictReply;
using ms_serving::PredictRequest;
using ms_serving::Tensor;
using ms_serving::TensorShape;

enum TypeId : int {
  kTypeUnknown = 0,
  kMetaTypeBegin = kTypeUnknown,
  kMetaTypeType,  // Type
  kMetaTypeAnything,
  kMetaTypeObject,
  kMetaTypeTypeType,  // TypeType
  kMetaTypeProblem,
  kMetaTypeExternal,
  kMetaTypeNone,
  kMetaTypeNull,
  kMetaTypeEllipsis,
  kMetaTypeEnd,
  //
  // Object types
  //
  kObjectTypeBegin = kMetaTypeEnd,
  kObjectTypeNumber,
  kObjectTypeString,
  kObjectTypeList,
  kObjectTypeTuple,
  kObjectTypeSlice,
  kObjectTypeKeyword,
  kObjectTypeTensorType,
  kObjectTypeClass,
  kObjectTypeDictionary,
  kObjectTypeFunction,
  kObjectTypeJTagged,
  kObjectTypeSymbolicKeyType,
  kObjectTypeEnvType,
  kObjectTypeRefKey,
  kObjectTypeRef,
  kObjectTypeEnd,
  //
  // Number Types
  //
  kNumberTypeBegin = kObjectTypeEnd,
  kNumberTypeBool,
  kNumberTypeInt,
  kNumberTypeInt8,
  kNumberTypeInt16,
  kNumberTypeInt32,
  kNumberTypeInt64,
  kNumberTypeUInt,
  kNumberTypeUInt8,
  kNumberTypeUInt16,
  kNumberTypeUInt32,
  kNumberTypeUInt64,
  kNumberTypeFloat,
  kNumberTypeFloat16,
  kNumberTypeFloat32,
  kNumberTypeFloat64,
  kNumberTypeEnd
};

std::string RealPath(const char *path) {
  if (path == nullptr) {
    std::cout << "path is nullptr";
    return "";
  }
  if ((strlen(path)) >= PATH_MAX) {
    std::cout << "path is too long";
    return "";
  }

  std::shared_ptr<char> resolvedPath(new (std::nothrow) char[PATH_MAX]{0});
  if (resolvedPath == nullptr) {
    std::cout << "new resolvedPath failed";
    return "";
  }

  auto ret = realpath(path, resolvedPath.get());
  if (ret == nullptr) {
    std::cout << "realpath failed";
    return "";
  }
  return resolvedPath.get();
}

char *ReadFile(const char *file, size_t *size) {
  if (file == nullptr) {
    std::cout << "file is nullptr" << std::endl;
    return nullptr;
  }
  if (size == nullptr) {
    std::cout << "size should not be nullptr" << std::endl;
    return nullptr;
  }
  std::ifstream ifs(RealPath(file));
  if (!ifs.good()) {
    std::cout << "file: " << file << "is not exist";
    return nullptr;
  }

  if (!ifs.is_open()) {
    std::cout << "file: " << file << "open failed";
    return nullptr;
  }

  ifs.seekg(0, std::ios::end);
  *size = ifs.tellg();
  std::unique_ptr<char> buf(new (std::nothrow) char[*size]);
  if (buf == nullptr) {
    std::cout << "malloc buf failed, file: " << file;
    ifs.close();
    return nullptr;
  }

  ifs.seekg(0, std::ios::beg);
  ifs.read(buf.get(), *size);
  ifs.close();

  return buf.release();
}
const std::map<TypeId, ms_serving::DataType> id2type_map{
  {TypeId::kNumberTypeBegin, ms_serving::MS_UNKNOWN},   {TypeId::kNumberTypeBool, ms_serving::MS_BOOL},
  {TypeId::kNumberTypeInt8, ms_serving::MS_INT8},       {TypeId::kNumberTypeUInt8, ms_serving::MS_UINT8},
  {TypeId::kNumberTypeInt16, ms_serving::MS_INT16},     {TypeId::kNumberTypeUInt16, ms_serving::MS_UINT16},
  {TypeId::kNumberTypeInt32, ms_serving::MS_INT32},     {TypeId::kNumberTypeUInt32, ms_serving::MS_UINT32},
  {TypeId::kNumberTypeInt64, ms_serving::MS_INT64},     {TypeId::kNumberTypeUInt64, ms_serving::MS_UINT64},
  {TypeId::kNumberTypeFloat16, ms_serving::MS_FLOAT16}, {TypeId::kNumberTypeFloat32, ms_serving::MS_FLOAT32},
  {TypeId::kNumberTypeFloat64, ms_serving::MS_FLOAT64},
};

int WriteFile(const void *buf, size_t size) {
  auto fd = fopen("output.json", "a+");
  if (fd == NULL) {
    std::cout << "fd is null and open file fail" << std::endl;
    return 0;
  }
  fwrite(buf, size, 1, fd);
  fclose(fd);
  return 0;
}

PredictRequest ReadBertInput() {
  size_t size;
  auto buf = ReadFile("input206.json", &size);
  if (buf == nullptr) {
    std::cout << "read file failed" << std::endl;
    return PredictRequest();
  }
  PredictRequest request;
  auto cur = buf;
  while (size > 0) {
    if (request.data_size() == 4) {
      break;
    }
    Tensor data;
    TensorShape shape;
    // set type
    int type = *(reinterpret_cast<int *>(cur));
    cur = cur + sizeof(int);
    size = size - sizeof(int);
    ms_serving::DataType dataType = id2type_map.at(TypeId(type));
    data.set_tensor_type(dataType);

    // set shape
    size_t dims = *(reinterpret_cast<size_t *>(cur));
    cur = cur + sizeof(size_t);
    size = size - sizeof(size_t);

    for (size_t i = 0; i < dims; i++) {
      int dim = *(reinterpret_cast<int *>(cur));
      shape.add_dims(dim);
      cur = cur + sizeof(int);
      size = size - sizeof(int);
    }
    *data.mutable_tensor_shape() = shape;

    // set data
    size_t data_len = *(reinterpret_cast<size_t *>(cur));
    cur = cur + sizeof(size_t);
    size = size - sizeof(size_t);
    data.set_data(cur, data_len);
    cur = cur + data_len;
    size = size - data_len;
    *request.add_data() = data;
  }
  return request;
}

class MSClient {
 public:
  explicit MSClient(std::shared_ptr<Channel> channel) : stub_(MSService::NewStub(channel)) {}

  std::string Predict(const std::string &type) {
    // Data we are sending to the server.
    PredictRequest request;
    if (type == "add") {
      Tensor data;
      TensorShape shape;
      shape.add_dims(1);
      shape.add_dims(1);
      shape.add_dims(2);
      shape.add_dims(2);
      *data.mutable_tensor_shape() = shape;
      data.set_tensor_type(ms_serving::MS_FLOAT32);
      std::vector<float> input_data{1.1, 2.1, 3.1, 4.1};
      data.set_data(input_data.data(), input_data.size());
      *request.add_data() = data;
      *request.add_data() = data;
    } else if (type == "bert") {
      request = ReadBertInput();
    } else {
      std::cout << "type only support bert or add, but input is " << type << std::endl;
    }
    std::cout << "intput tensor size is " << request.data_size() << std::endl;
    // Container for the data we expect from the server.
    PredictReply reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // The actual RPC.
    Status status = stub_->Predict(&context, request, &reply);

    for (int i = 0; i < reply.result_size(); i++) {
      WriteFile(reply.result(i).data().data(), reply.result(i).data().size());
    }

    std::cout << "the return result size is " << reply.result_size() << std::endl;

    // Act upon its status.
    if (status.ok()) {
      return "RPC OK";
    } else {
      std::cout << status.error_code() << ": " << status.error_message() << std::endl;
      return "RPC failed";
    }
  }

 private:
  std::unique_ptr<MSService::Stub> stub_;
};

int main(int argc, char **argv) {
  // Instantiate the client. It requires a channel, out of which the actual RPCs
  // are created. This channel models a connection to an endpoint specified by
  // the argument "--target=" which is the only expected argument.
  // We indicate that the channel isn't authenticated (use of
  // InsecureChannelCredentials()).
  std::string target_str;
  std::string arg_target_str("--target");
  std::string type;
  std::string arg_type_str("--type");
  if (argc > 2) {
    {
      // parse target
      std::string arg_val = argv[1];
      size_t start_pos = arg_val.find(arg_target_str);
      if (start_pos != std::string::npos) {
        start_pos += arg_target_str.size();
        if (arg_val[start_pos] == '=') {
          target_str = arg_val.substr(start_pos + 1);
        } else {
          std::cout << "The only correct argument syntax is --target=" << std::endl;
          return 0;
        }
      } else {
        target_str = "localhost:5500";
      }
    }

    {
      // parse type
      std::string arg_val2 = argv[2];
      size_t start_pos = arg_val2.find(arg_type_str);
      if (start_pos != std::string::npos) {
        start_pos += arg_type_str.size();
        if (arg_val2[start_pos] == '=') {
          type = arg_val2.substr(start_pos + 1);
        } else {
          std::cout << "The only correct argument syntax is --target=" << std::endl;
          return 0;
        }
      } else {
        type = "add";
      }
    }

  } else {
    target_str = "localhost:5500";
    type = "add";
  }
  MSClient client(grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));
  std::string reply = client.Predict(type);
  std::cout << "client received: " << reply << std::endl;

  return 0;
}
