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
#include "core/serving_tensor.h"

using ms_serving::MSService;
using ms_serving::PredictReply;
using ms_serving::PredictRequest;
using nlohmann::json;

namespace mindspore {
namespace serving {

const int BUF_MAX = 0x7FFFFFFF;
static constexpr char HTTP_DATA[] = "data";
static constexpr char HTTP_TENSOR[] = "tensor";
enum HTTP_TYPE { TYPE_DATA = 0, TYPE_TENSOR };
enum HTTP_DATA_TYPE { HTTP_DATA_NONE, HTTP_DATA_INT, HTTP_DATA_FLOAT };

static const std::map<inference::DataType, HTTP_DATA_TYPE> infer_type2_http_type{
  {inference::DataType::kMSI_Int32, HTTP_DATA_INT}, {inference::DataType::kMSI_Float32, HTTP_DATA_FLOAT}};

Status GetPostMessage(struct evhttp_request *req, std::string *buf) {
  Status status(SUCCESS);
  size_t post_size = evbuffer_get_length(req->input_buffer);
  if (post_size == 0) {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "http message invalid");
    return status;
  } else if (post_size > BUF_MAX) {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "http message is bigger than 0x7FFFFFFF.");
    return status;
  } else {
    buf->resize(post_size);
    memcpy(buf->data(), evbuffer_pullup(req->input_buffer, -1), post_size);
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
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "http message must have only one type of (data, tensor)");
    return status;
  }
  return status;
}

Status GetDataFromJson(const json &json_data_array, ServingTensor *request_tensor, size_t data_index,
                       HTTP_DATA_TYPE type) {
  Status status(SUCCESS);
  auto type_name = [](const json &json_data) -> std::string {
    if (json_data.is_number_integer()) {
      return "integer";
    } else if (json_data.is_number_float()) {
      return "float";
    }
    return json_data.type_name();
  };
  size_t array_size = json_data_array.size();
  if (type == HTTP_DATA_INT) {
    auto data = reinterpret_cast<int32_t *>(request_tensor->mutable_data()) + data_index;
    for (size_t k = 0; k < array_size; k++) {
      auto &json_data = json_data_array[k];
      if (!json_data.is_number_integer()) {
        status = INFER_STATUS(INVALID_INPUTS) << "get data failed, expected integer, given " << type_name(json_data);
        MSI_LOG_ERROR << status.StatusMessage();
        return status;
      }
      data[k] = json_data.get<int32_t>();
    }
  } else if (type == HTTP_DATA_FLOAT) {
    auto data = reinterpret_cast<float *>(request_tensor->mutable_data()) + data_index;
    for (size_t k = 0; k < array_size; k++) {
      auto &json_data = json_data_array[k];
      if (!json_data.is_number_float()) {
        status = INFER_STATUS(INVALID_INPUTS) << "get data failed, expected float, given " << type_name(json_data);
        MSI_LOG_ERROR << status.StatusMessage();
        return status;
      }
      data[k] = json_data.get<float>();
    }
  }
  return SUCCESS;
}

Status RecusiveGetTensor(const json &json_data, size_t depth, ServingTensor *request_tensor, size_t data_index,
                         HTTP_DATA_TYPE type) {
  Status status(SUCCESS);
  std::vector<int64_t> required_shape = request_tensor->shape();
  if (depth >= required_shape.size()) {
    status = INFER_STATUS(INVALID_INPUTS)
             << "input tensor shape dims is more than required dims " << required_shape.size();
    MSI_LOG_ERROR << status.StatusMessage();
    return status;
  }
  if (!json_data.is_array()) {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "the tensor is constructed illegally");
    return status;
  }
  if (json_data.size() != static_cast<size_t>(required_shape[depth])) {
    status = INFER_STATUS(INVALID_INPUTS)
             << "tensor format request is constructed illegally, input tensor shape dim " << depth
             << " not match, required " << required_shape[depth] << ", given " << json_data.size();
    MSI_LOG_ERROR << status.StatusMessage();
    return status;
  }
  if (depth + 1 < required_shape.size()) {
    size_t sub_element_cnt =
      std::accumulate(required_shape.begin() + depth + 1, required_shape.end(), 1LL, std::multiplies<size_t>());
    for (size_t k = 0; k < json_data.size(); k++) {
      status = RecusiveGetTensor(json_data[k], depth + 1, request_tensor, data_index + sub_element_cnt * k, type);
      if (status != SUCCESS) {
        return status;
      }
    }
  } else {
    status = GetDataFromJson(json_data, request_tensor, data_index, type);
    if (status != SUCCESS) {
      return status;
    }
  }
  return status;
}

std::vector<int64_t> GetJsonArrayShape(const json &json_array) {
  std::vector<int64_t> json_shape;
  const json *tmp_json = &json_array;
  while (tmp_json->is_array()) {
    if (tmp_json->empty()) {
      break;
    }
    json_shape.push_back(tmp_json->size());
    tmp_json = &tmp_json->at(0);
  }
  return json_shape;
}

Status TransDataToPredictRequest(const json &message_info, PredictRequest *request) {
  Status status = SUCCESS;
  auto tensors = message_info.find(HTTP_DATA);
  if (tensors == message_info.end()) {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "http message do not have data type");
    return status;
  }
  if (!tensors->is_array()) {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "the input tensor list is not array");
    return status;
  }
  auto const &json_shape = GetJsonArrayShape(*tensors);
  if (json_shape.size() != 2) {  // 2 is data format list deep
    status = INFER_STATUS(INVALID_INPUTS)
             << "the data format request is constructed illegally, expected list nesting depth 2, given "
             << json_shape.size();
    MSI_LOG_ERROR << status.StatusMessage();
    return status;
  }
  if (tensors->size() != static_cast<size_t>(request->data_size())) {
    status = INFER_STATUS(INVALID_INPUTS)
             << "model input count not match, model required " << request->data_size() << ", given " << tensors->size();
    MSI_LOG_ERROR << status.StatusMessage();
    return status;
  }
  for (size_t i = 0; i < tensors->size(); i++) {
    const auto &tensor = tensors->at(i);
    ServingTensor request_tensor(*(request->mutable_data(i)));
    auto iter = infer_type2_http_type.find(request_tensor.data_type());
    if (iter == infer_type2_http_type.end()) {
      ERROR_INFER_STATUS(status, FAILED, "the model input type is not supported right now");
      return status;
    }
    HTTP_DATA_TYPE type = iter->second;
    if (!tensor.is_array()) {
      ERROR_INFER_STATUS(status, INVALID_INPUTS, "the tensor is constructed illegally");
      return status;
    }
    if (tensor.empty()) {
      ERROR_INFER_STATUS(status, INVALID_INPUTS, "the input tensor is null");
      return status;
    }
    if (tensor.size() != static_cast<size_t>(request_tensor.ElementNum())) {
      status = INFER_STATUS(INVALID_INPUTS) << "input " << i << " element count not match, model required "
                                            << request_tensor.ElementNum() << ", given " << tensor.size();
      MSI_LOG_ERROR << status.StatusMessage();
      return status;
    }
    status = GetDataFromJson(tensor, &request_tensor, 0, type);
    if (status != SUCCESS) {
      return status;
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
  if (!tensors->is_array()) {
    ERROR_INFER_STATUS(status, INVALID_INPUTS, "the input tensor list is not array");
    return status;
  }
  if (tensors->size() != static_cast<size_t>(request->data_size())) {
    status =
      INFER_STATUS(INVALID_INPUTS)
      << "model input count not match or json tensor request is constructed illegally, model input count required "
      << request->data_size() << ", given " << tensors->size();
    MSI_LOG_ERROR << status.StatusMessage();
    return status;
  }

  for (size_t i = 0; i < tensors->size(); i++) {
    const auto &tensor = tensors->at(i);
    ServingTensor request_tensor(*(request->mutable_data(i)));

    // check data shape
    auto const &json_shape = GetJsonArrayShape(tensor);
    if (json_shape != request_tensor.shape()) {  // data shape not match
      status = INFER_STATUS(INVALID_INPUTS)
               << "input " << i << " shape is invalid, expected " << request_tensor.shape() << ", given " << json_shape;
      MSI_LOG_ERROR << status.StatusMessage();
      return status;
    }

    auto iter = infer_type2_http_type.find(request_tensor.data_type());
    if (iter == infer_type2_http_type.end()) {
      ERROR_INFER_STATUS(status, FAILED, "the model input type is not supported right now");
      return status;
    }
    HTTP_DATA_TYPE type = iter->second;
    size_t depth = 0;
    size_t data_index = 0;
    status = RecusiveGetTensor(tensor, depth, &request_tensor, data_index, type);
    if (status != SUCCESS) {
      MSI_LOG_ERROR << "Transfer tensor to predict request failed";
      return status;
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

  // get model required shape
  std::vector<inference::InferTensor> tensor_list;
  status = Session::Instance().GetModelInputsInfo(tensor_list);
  if (status != SUCCESS) {
    ERROR_INFER_STATUS(status, FAILED, "get model inputs info failed");
    return status;
  }
  for (auto &item : tensor_list) {
    auto input = request->add_data();
    ServingTensor tensor(*input);
    tensor.set_shape(item.shape());
    tensor.set_data_type(item.data_type());
    int64_t element_num = tensor.ElementNum();
    int64_t data_type_size = tensor.GetTypeSize(tensor.data_type());
    if (element_num <= 0 || INT64_MAX / element_num < data_type_size) {
      ERROR_INFER_STATUS(status, FAILED, "model shape invalid");
      return status;
    }
    tensor.resize_data(element_num * data_type_size);
  }
  MSI_TIME_STAMP_START(ParseJson)
  json message_info;
  try {
    message_info = nlohmann::json::parse(post_message);
  } catch (nlohmann::json::exception &e) {
    std::string json_exception = e.what();
    std::string error_message = "Illegal JSON format." + json_exception;
    ERROR_INFER_STATUS(status, INVALID_INPUTS, error_message);
    return status;
  }
  MSI_TIME_STAMP_END(ParseJson)

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
      auto data = reinterpret_cast<const int *>(tensor.data().data()) + *pos;
      std::vector<int32_t> result_tensor(len);
      memcpy_s(result_tensor.data(), result_tensor.size() * sizeof(int32_t), data, len * sizeof(int32_t));
      *out_json = std::move(result_tensor);
      *pos += len;
      break;
    }
    case ms_serving::MS_FLOAT32: {
      auto data = reinterpret_cast<const float *>(tensor.data().data()) + *pos;
      std::vector<float> result_tensor(len);
      memcpy_s(result_tensor.data(), result_tensor.size() * sizeof(float), data, len * sizeof(float));
      *out_json = std::move(result_tensor);
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
    (*out_json)["data"].push_back(json());
    json &tensor_json = (*out_json)["data"].back();
    int num = 1;
    for (auto j = 0; j < reply.result(i).tensor_shape().dims_size(); j++) {
      num *= reply.result(i).tensor_shape().dims(j);
    }
    int pos = 0;
    status = GetJsonFromTensor(reply.result(i), num, &pos, &tensor_json);
    if (status != SUCCESS) {
      return status;
    }
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
      out_json->push_back(json());
      json &tensor_json = out_json->back();
      status = RecusiveGetJson(tensor, depth + 1, pos, &tensor_json);
      if (status != SUCCESS) {
        return status;
      }
    }
  }
  return status;
}

Status TransPredictReplyToTensor(const PredictReply &reply, json *out_json) {
  Status status(SUCCESS);
  for (int i = 0; i < reply.result_size(); i++) {
    (*out_json)["tensor"].push_back(json());
    json &tensor_json = (*out_json)["tensor"].back();
    int pos = 0;
    status = RecusiveGetJson(reply.result(i), 0, &pos, &tensor_json);
    if (status != SUCCESS) {
      return status;
    }
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

  const std::string &out_str = out_json.dump();
  evbuffer_add(buf, out_str.data(), out_str.size());
  return status;
}

Status HttpHandleMsgDetail(struct evhttp_request *req, void *arg, struct evbuffer *retbuff) {
  PredictRequest request;
  PredictReply reply;
  HTTP_TYPE type;
  MSI_TIME_STAMP_START(ParseRequest)
  auto status = TransHTTPMsgToPredictRequest(req, &request, &type);
  MSI_TIME_STAMP_END(ParseRequest)
  if (status != SUCCESS) {
    MSI_LOG(ERROR) << "restful trans to request failed";
    return status;
  }
  MSI_TIME_STAMP_START(Predict)
  status = Session::Instance().Predict(request, reply);
  MSI_TIME_STAMP_END(Predict)
  if (status != SUCCESS) {
    MSI_LOG(ERROR) << "restful predict failed";
    return status;
  }
  MSI_TIME_STAMP_START(CreateReplyJson)
  status = TransPredictReplyToHTTPMsg(reply, type, retbuff);
  MSI_TIME_STAMP_END(CreateReplyJson)
  if (status != SUCCESS) {
    MSI_LOG(ERROR) << "restful trans to reply failed";
    return status;
  }
  return SUCCESS;
}

void http_handler_msg(struct evhttp_request *req, void *arg) {
  MSI_TIME_STAMP_START(TotalRestfulPredict)
  struct evbuffer *retbuff = evbuffer_new();
  if (retbuff == nullptr) {
    MSI_LOG_ERROR << "Create event buffer failed";
    return;
  }
  auto status = HttpHandleMsgDetail(req, arg, retbuff);
  if (status != SUCCESS) {
    ErrorMessage(req, status);
    evbuffer_free(retbuff);
    return;
  }
  MSI_TIME_STAMP_START(ReplyJson)
  evhttp_send_reply(req, HTTP_OK, "Client", retbuff);
  MSI_TIME_STAMP_END(ReplyJson)
  evbuffer_free(retbuff);
  MSI_TIME_STAMP_END(TotalRestfulPredict)
}

}  // namespace serving
}  // namespace mindspore
