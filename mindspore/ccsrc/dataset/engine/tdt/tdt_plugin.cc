/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "dataset/engine/tdt/tdt_plugin.h"
#include "common/utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace dataset {
static std::shared_ptr<TdtPlugin> instance_ptr_ = nullptr;

std::shared_ptr<TdtPlugin> TdtPlugin::GetInstance() {
  if (instance_ptr_ == nullptr) {
    instance_ptr_ = std::shared_ptr<TdtPlugin>(new TdtPlugin);
  }
  return instance_ptr_;
}

TdtStatus TdtPlugin::hostPush(TensorRow ts_row, bool is_wait, std::string channel_name) {
  MS_LOG(INFO) << "TDT channel name is " << channel_name << ".";
  std::vector<DataItem> items;
  auto ret = translate(ts_row, items);
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "TDT converting tensor failed!";
    return FAILED;
  }
  if (tdt::TdtHostPushData(channel_name, items) != 0) {
    MS_LOG(ERROR) << "TDT pushing data failed!";
    return FAILED;
  }
  return SUCCESS;
}

TdtStatus TdtPlugin::getTdtType(DataType d_type, std::string &datatype) {
  switch (d_type.value()) {
    case DataType::DE_BOOL:
      datatype = "bool";
      break;
    case DataType::DE_INT8:
      datatype = "int8";
      break;
    case DataType::DE_UINT8:
      datatype = "uint8";
      break;
    case DataType::DE_INT16:
      datatype = "int16";
      break;
    case DataType::DE_UINT16:
      datatype = "uint16";
      break;
    case DataType::DE_INT32:
      datatype = "int32";
      break;
    case DataType::DE_UINT32:
      datatype = "uint32";
      break;
    case DataType::DE_FLOAT16:
      datatype = "float16";
      break;
    case DataType::DE_FLOAT32:
      datatype = "float32";
      break;
    case DataType::DE_FLOAT64:
      datatype = "float64";
      break;
    case DataType::DE_INT64:
      datatype = "int64";
      break;
    case DataType::DE_UINT64:
      datatype = "uint64";
      break;
    default:
      return FAILED;
  }
  return SUCCESS;
}

TdtStatus TdtPlugin::translate(const TensorRow &ts_row, std::vector<DataItem> &items) {
  if (ts_row.size() == 0) {
    MS_LOG(ERROR) << "TDT the size of row is zero.";
    return SUCCESS;
  }
  for (auto ts : ts_row) {
    std::string datatype;
    TdtStatus status = getTdtType(ts->type(), datatype);
    if (status != SUCCESS) {
      return status;
    }
    TensorShape tsShape = ts->shape();
    std::string dataShapes = "[";
    for (auto dim : tsShape.AsVector()) {
      (void)dataShapes.append(std::to_string(dim)).append(",");
    }
    dataShapes.pop_back();
    (void)dataShapes.append("]");
    DataItem data_item;
    data_item.dataType_ = tdt::TDT_TENSOR;
    data_item.tensorShape_ = dataShapes;
    data_item.tensorType_ = datatype;
    data_item.dataLen_ = ts->SizeInBytes();
    data_item.dataPtr_ = std::shared_ptr<void>(reinterpret_cast<void *>(ts->GetMutableBuffer()), [](void *elem) {});
    items.emplace_back(data_item);
    MS_LOG(INFO) << "TDT data type is " << datatype << ", data shape is " << dataShapes << ", data length is "
                 << ts->Size() << ".";
  }
  return SUCCESS;
}
}  // namespace dataset
}  // namespace mindspore
