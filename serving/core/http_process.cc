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
#include <map>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "serving/ms_service.pb.h"
#include "util/status.h"
#include "core/session.h"
#include "core/http_process.h"

using ms_serving::MSService;
using ms_serving::PredictReply;
using ms_serving::PredictRequest;
using nlohmann::json;

namespace mindspore {
namespace serving {

const int BUF_MAX = 0x1FFFFF;
static constexpr char HTTP_DATA[] = "data";
static constexpr char HTTP_TENSOR[] = "tensor";
enum HTTP_TYPE { TYPE_DATA = 0, TYPE_TENSOR };
enum HTTP_DATA_TYPE { HTTP_DATA_NONE, HTTP_DATA_INT, HTTP_DATA_FLOAT };
static const std::map<HTTP_DATA_TYPE, ms_serving::DataType> http_to_infer_map{
  {HTTP_DATA_NONE, ms_serving::MS_UNKNOWN},
  {HTTP_DATA_INT, ms_serving::MS_INT32},
  {HTTP_DATA_FLOAT, ms_serving::MS_FLOAT32}};

Status GetPostMessage(struct evhttp_request *req, std::string *buf) {
  Status status(SUCCESS);
  size_t post_size = evbuffer_get_length(req->input_buffer);
  if (post_size == 0) {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "http message invalid");
    return status;
  } else {
    size_t copy_len = post_size > BUF_MAX ? BUF_MAX : post_size;
    buf->resize(copy_len);
    memcpy(buf->data(), evbuffer_pullup(req->input_buffer, -1), copy_len);
    return status;
  }
}
Status CheckRequestValid(struct evhttp_request *http_request) {
  Status status(SUCCESS);
  switch (evhttp_request_get_command(http_request)) {
    case EVHTTP_REQ_POST:
      return status;
    default:
      ERROR_INFER_STATUS(status, INVALID_INPUTS, "http message only support POST right now");
      return status;
  }
}

void ErrorMessage(struct evhttp_request *req, Status status) {
  json error_json = {{"error_message", status.StatusMessage()}};
  std::string out_error_str = error_json.dump();
  struct evbuffer *retbuff = evbuffer_new();
  evbuffer_add(retbuff, out_error_str.data(), out_error_str.size());
  evhttp_send_reply(req, HTTP_OK, "Client", retbuff);
  evbuffer_free(retbuff);
}

Status CheckMessageValid(const json &message_info, HTTP_TYPE *type) {
  Status status(SUCCESS);
  int count = 0;
  if (message_info.find(HTTP_DATA) != message_info.end()) {
    *type = TYPE_DATA;
    count++;
  }
  if (message_info.find(HTTP_TENSOR) != message_info.end()) {
    *type = TYPE_TENSOR;
    count++;
  }
  if (count != 1) {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "http message must have only one type of (data, tensor, text)");
    return status;
  }
  return status;
}

Status GetDataFromJson(const json &json_data, std::string *data, HTTP_DATA_TYPE *type) {
  Status status(SUCCESS);
  if (json_data.is_number_integer()) {
    if (*type == HTTP_DATA_NONE) {
      *type = HTTP_DATA_INT;
    } else if (*type != HTTP_DATA_INT) {
      ERROR_INFER_STATUS(status, INVALID_INPUTS, "the input data type should be consistent");
      return status;
    }
    auto s_data = json_data.get<int32_t>();
    data->append(reinterpret_cast<char *>(&s_data), sizeof(int32_t));
  } else if (json_data.is_number_float()) {
    if (*type == HTTP_DATA_NONE) {
      *type = HTTP_DATA_FLOAT;
    } else if (*type != HTTP_DATA_FLOAT) {
      ERROR_INFER_STATUS(status, INVALID_INPUTS, "the input data type should be consistent");
      return status;
    }
    auto s_data = json_data.get<float>();
    data->append(reinterpret_cast<char *>(&s_data), sizeof(float));
  } else {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "the input data type should be int or float");
    return status;
  }
  return SUCCESS;
}

Status RecusiveGetTensor(const json &json_data, size_t depth, std::vector<int> *shape, std::string *data,
                         HTTP_DATA_TYPE *type) {
  Status status(SUCCESS);
  if (depth >= 10) {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "the tensor shape dims is larger than 10");
    return status;
  }
  if (!json_data.is_array()) {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "the tensor is constructed illegally");
    return status;
  }
  int cur_dim = json_data.size();
  if (shape->size() <= depth) {
    shape->push_back(cur_dim);
  } else if ((*shape)[depth] != cur_dim) {
    return INFER_STATUS(INVALID_INPUTS) << "the tensor shape is constructed illegally";
  }
  if (json_data.at(0).is_array()) {
    for (const auto &item : json_data) {
      status = RecusiveGetTensor(item, depth + 1, shape, data, type);
      if (status != SUCCESS) {
        return status;
      }
    }
  } else {
    // last dim, read the data
    for (auto item : json_data) {
      status = GetDataFromJson(item, data, type);
      if (status != SUCCESS) {
        return status;
      }
    }
  }
  return status;
}

Status TransDataToPredictRequest(const json &message_info, PredictRequest *request) {
  Status status = SUCCESS;
  auto tensors = message_info.find(HTTP_DATA);
  if (tensors == message_info.end()) {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "http message do not have data type");
    return status;
  }

  if (tensors->size() == 0) {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "the input tensor list is null");
    return status;
  }
  for (const auto &tensor : *tensors) {
    std::string msg_data;
    HTTP_DATA_TYPE type{HTTP_DATA_NONE};
    if (!tensor.is_array()) {
      ERROR_INFER_STATUS(status, INVALID_INPUTS, "the tensor is constructed illegally");
      return status;
    }
    if (tensor.size() == 0) {
      ERROR_INFER_STATUS(status, INVALID_INPUTS, "the input tensor is null");
      return status;
    }
    for (const auto &tensor_data : tensor) {
      status = GetDataFromJson(tensor_data, &msg_data, &type);
      if (status != SUCCESS) {
        return status;
      }
    }
    auto iter = http_to_infer_map.find(type);
    if (iter == http_to_infer_map.end()) {
      ERROR_INFER_STATUS(status, INVALID_INPUTS, "the input type is not supported right now");
      return status;
    }

    auto infer_tensor = request->add_data();
    infer_tensor->set_tensor_type(iter->second);
    infer_tensor->set_data(msg_data.data(), msg_data.size());
  }
  // get model required shape
  std::vector<inference::InferTensor> tensor_list;
  status = Session::Instance().GetModelInputsInfo(tensor_list);
  if (status != SUCCESS) {
    ERROR_INFER_STATUS(status, FAILED, "get model inputs info failed");
    return status;
  }
  if (request->data_size() != static_cast<int64_t>(tensor_list.size())) {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "the inputs number is not equal to model required");
    return status;
  }
  for (int i = 0; i < request->data_size(); i++) {
    for (size_t j = 0;  j < tensor_list[i].shape().size(); ++j) {
      request->mutable_data(i)->mutable_tensor_shape()->add_dims(tensor_list[i].shape()[i]);
    }
  }
  return SUCCESS;
}

Status TransTensorToPredictRequest(const json &message_info, PredictRequest *request) {
  Status status(SUCCESS);
  auto tensors = message_info.find(HTTP_TENSOR);
  if (tensors == message_info.end()) {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "http message do not have tensor type");
    return status;
  }

  for (const auto &tensor : *tensors) {
    std::vector<int> shape;
    std::string msg_data;
    HTTP_DATA_TYPE type{HTTP_DATA_NONE};
    RecusiveGetTensor(tensor, 0, &shape, &msg_data, &type);
    auto iter = http_to_infer_map.find(type);
    if (iter == http_to_infer_map.end()) {
      ERROR_INFER_STATUS(status, INVALID_INPUTS, "the input type is not supported right now");
      return status;
    }
    auto infer_tensor = request->add_data();
    infer_tensor->set_tensor_type(iter->second);
    infer_tensor->set_data(msg_data.data(), msg_data.size());
    for (const auto dim : shape) {
      infer_tensor->mutable_tensor_shape()->add_dims(dim);
    }
  }
  return status;
}

Status TransHTTPMsgToPredictRequest(struct evhttp_request *http_request, PredictRequest *request, HTTP_TYPE *type) {
  Status status = CheckRequestValid(http_request);
  if (status != SUCCESS) {
    return status;
  }
  std::string post_message;
  status = GetPostMessage(http_request, &post_message);
  if (status != SUCCESS) {
    return status;
  }

  json message_info;
  try {
    message_info = nlohmann::json::parse(post_message);
  } catch (nlohmann::json::exception &e) {
    std::string json_exception = e.what();
    std::string error_message = "Illegal JSON format." + json_exception;
    ERROR_INFER_STATUS(status, INVALID_INPUTS, error_message);
    return status;
  }

  status = CheckMessageValid(message_info, type);
  if (status != SUCCESS) {
    return status;
  }
  switch (*type) {
    case TYPE_DATA:
      status = TransDataToPredictRequest(message_info, request);
      break;
    case TYPE_TENSOR:
      status = TransTensorToPredictRequest(message_info, request);
      break;
    default:
      ERROR_INFER_STATUS(status, INVALID_INPUTS, "http message must have only one type of (data, tensor)");
      return status;
  }
  return status;
}

Status GetJsonFromTensor(const ms_serving::Tensor &tensor, int len, int *pos, json *out_json) {
  Status status(SUCCESS);
  switch (tensor.tensor_type()) {
    case ms_serving::MS_INT32: {
      std::vector<int> result_tensor;
      for (int j = 0; j < len; j++) {
        int val;
        memcpy(&val, reinterpret_cast<const int *>(tensor.data().data()) + *pos + j, sizeof(int));
        result_tensor.push_back(val);
      }
      *out_json = result_tensor;
      *pos += len;
      break;
    }
    case ms_serving::MS_FLOAT32: {
      std::vector<float> result_tensor;
      for (int j = 0; j < len; j++) {
        float val;
        memcpy(&val, reinterpret_cast<const float *>(tensor.data().data()) + *pos + j, sizeof(float));
        result_tensor.push_back(val);
      }
      *out_json = result_tensor;
      *pos += len;
      break;
    }
    default:
      MSI_LOG(ERROR) << "the result type is not supported in restful api, type is " << tensor.tensor_type();
      ERROR_INFER_STATUS(status, FAILED, "reply have unsupported type");
  }
  return status;
}

Status TransPredictReplyToData(const PredictReply &reply, json *out_json) {
  Status status(SUCCESS);
  for (int i = 0; i < reply.result_size(); i++) {
    json tensor_json;
    int num = 1;
    for (auto j = 0; j < reply.result(i).tensor_shape().dims_size(); j++) {
      num *= reply.result(i).tensor_shape().dims(j);
    }
    int pos = 0;
    status = GetJsonFromTensor(reply.result(i), num, &pos, &tensor_json);
    if (status != SUCCESS) {
      return status;
    }
    (*out_json)["data"].push_back(tensor_json);
  }
  return status;
}

Status RecusiveGetJson(const ms_serving::Tensor &tensor, int depth, int *pos, json *out_json) {
  Status status(SUCCESS);
  if (depth >= 10) {
    ERROR_INFER_STATUS(status, FAILED, "result tensor shape dims is larger than 10");
    return status;
  }
  if (depth == tensor.tensor_shape().dims_size() - 1) {
    status = GetJsonFromTensor(tensor, tensor.tensor_shape().dims(depth), pos, out_json);
    if (status != SUCCESS) {
      return status;
    }
  } else {
    for (int i = 0; i < tensor.tensor_shape().dims(depth); i++) {
      json tensor_json;
      status = RecusiveGetJson(tensor, depth + 1, pos, &tensor_json);
      if (status != SUCCESS) {
        return status;
      }
      out_json->push_back(tensor_json);
    }
  }
  return status;
}

Status TransPredictReplyToTensor(const PredictReply &reply, json *out_json) {
  Status status(SUCCESS);
  for (int i = 0; i < reply.result_size(); i++) {
    json tensor_json;
    int pos = 0;
    status = RecusiveGetJson(reply.result(i), 0, &pos, &tensor_json);
    if (status != SUCCESS) {
      return status;
    }
    (*out_json)["tensor"].push_back(tensor_json);
  }
  return status;
}

Status TransPredictReplyToHTTPMsg(const PredictReply &reply, const HTTP_TYPE &type, struct evbuffer *buf) {
  Status status(SUCCESS);
  json out_json;
  switch (type) {
    case TYPE_DATA:
      status = TransPredictReplyToData(reply, &out_json);
      break;
    case TYPE_TENSOR:
      status = TransPredictReplyToTensor(reply, &out_json);
      break;
    default:
      ERROR_INFER_STATUS(status, FAILED, "http message must have only one type of (data, tensor)");
      return status;
  }

  std::string out_str = out_json.dump();
  evbuffer_add(buf, out_str.data(), out_str.size());
  return status;
}

void http_handler_msg(struct evhttp_request *req, void *arg) {
  std::cout << "in handle" << std::endl;
  PredictRequest request;
  PredictReply reply;
  HTTP_TYPE type;
  auto status = TransHTTPMsgToPredictRequest(req, &request, &type);
  if (status != SUCCESS) {
    ErrorMessage(req, status);
    MSI_LOG(ERROR) << "restful trans to request failed";
    return;
  }
  MSI_TIME_STAMP_START(Predict)
  status = Session::Instance().Predict(request, reply);
  if (status != SUCCESS) {
    ErrorMessage(req, status);
    MSI_LOG(ERROR) << "restful predict failed";
  }
  MSI_TIME_STAMP_END(Predict)
  struct evbuffer *retbuff = evbuffer_new();
  status = TransPredictReplyToHTTPMsg(reply, type, retbuff);
  if (status != SUCCESS) {
    ErrorMessage(req, status);
    MSI_LOG(ERROR) << "restful trans to reply failed";
    return;
  }
  evhttp_send_reply(req, HTTP_OK, "Client", retbuff);
  evbuffer_free(retbuff);
}

}  // namespace serving
}  // namespace mindspore
