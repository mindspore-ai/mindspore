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
#include "minddata/dataset/engine/tdt/tdt_plugin.h"
#include "utils/ms_utils.h"
#include "minddata/dataset/engine/perf/profiling.h"
#include "minddata/dataset/util/log_adapter.h"
#if ENABLE_D
#include "ps/ps_cache/ps_data/ps_data_prefetch.h"
#endif

namespace mindspore {
namespace dataset {
TdtPlugin::TdtPlugin(const std::string &channel_name, int32_t device_id) {
  // create acl tdt handle
  acl_handle_ = acltdtCreateChannel(device_id, channel_name.c_str());
  if (acl_handle_ == nullptr) {
    MS_LOG(ERROR) << "Failed to create channel for tdt queue.";
  }
  TdtHandle::AddHandle(&acl_handle_);
}

TdtPlugin::~TdtPlugin() {
  if (acl_handle_ != nullptr && acltdtDestroyChannel(acl_handle_) != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Failed to destroy channel for tdt queue.";
  }
}

Status TdtPlugin::hostPush(TensorRow ts_row, bool is_wait, std::string channel_name, bool profiling, int32_t &time,
                           acltdtTensorType tdt_type) {
  MS_LOG(DEBUG) << "TDT channel name is " << channel_name << ".";

  acltdtDataset *acl_dataset = nullptr;
  double start_time;
  auto ret = translate(tdt_type, ts_row, &acl_dataset);
  if (ret != Status::OK()) {
    DestroyAclDataset(acl_dataset);
    RETURN_STATUS_UNEXPECTED("Converting into TDT tensor failed!");
  }

  if (profiling) {
    start_time = ProfilingTime::GetCurMilliSecond();
  }
#if ENABLE_D
  // Data prefetch only when PS mode enables cache.
  if (acltdtGetDatasetSize(acl_dataset) > 0) {
    acltdtDataItem *item0 = acltdtGetDataItem(acl_dataset, 0);
    std::string item_type = "unsupported";
    if (acltdtGetDataTypeFromItem(item0) == ACL_INT32) {
      item_type = "int32";
    }
    if (!ps::PsDataPrefetch::GetInstance().PrefetchData(channel_name, acltdtGetDataAddrFromItem(item0),
                                                        acltdtGetDataSizeFromItem(item0), item_type)) {
      RETURN_STATUS_UNEXPECTED("PrefetchData failed in when pre-processing sending data.");
    }
  }
#endif
  auto status = acltdtSendTensor(acl_handle_, acl_dataset, -1);
  DestroyAclDataset(acl_dataset);
  if (status != ACL_SUCCESS) {
    RETURN_STATUS_UNEXPECTED("Tdt Send data failed.");
  }
  if (profiling) {
    double end_time = ProfilingTime::GetCurMilliSecond();
    time = (int32_t)(end_time - start_time);
  }
  return Status::OK();
}

Status TdtPlugin::getTdtType(DataType d_type, aclDataType &datatype) {
  switch (d_type.value()) {
    case DataType::DE_BOOL:
      datatype = ACL_BOOL;
      break;
    case DataType::DE_INT8:
      datatype = ACL_INT8;
      break;
    case DataType::DE_UINT8:
      datatype = ACL_UINT8;
      break;
    case DataType::DE_INT16:
      datatype = ACL_INT16;
      break;
    case DataType::DE_UINT16:
      datatype = ACL_UINT16;
      break;
    case DataType::DE_INT32:
      datatype = ACL_INT32;
      break;
    case DataType::DE_UINT32:
      datatype = ACL_UINT32;
      break;
    case DataType::DE_FLOAT16:
      datatype = ACL_FLOAT16;
      break;
    case DataType::DE_FLOAT32:
      datatype = ACL_FLOAT;
      break;
    case DataType::DE_FLOAT64:
      datatype = ACL_DOUBLE;
      break;
    case DataType::DE_INT64:
      datatype = ACL_INT64;
      break;
    case DataType::DE_UINT64:
      datatype = ACL_UINT64;
      break;
    default:
      RETURN_STATUS_UNEXPECTED("Invalid data, got unexpected data type.");
  }
  return Status::OK();
}

Status TdtPlugin::translate(acltdtTensorType tdt_type, const TensorRow &ts_row, acltdtDataset **output_acl_dataset) {
  auto acl_dataset = acltdtCreateDataset();
  if (acl_dataset == nullptr) {
    RETURN_STATUS_UNEXPECTED("Create tdt dataset failed.");
  }
  auto status = AssembleTensor2AclDataset(tdt_type, ts_row, acl_dataset);
  if (status != Status::OK()) {
    DestroyAclDataset(acl_dataset);
    RETURN_STATUS_UNEXPECTED("Assemble tensor row to tdt dataset failed.");
  }

  *output_acl_dataset = acl_dataset;
  return Status::OK();
}

Status TdtPlugin::AssembleTensor2AclDataset(acltdtTensorType tdt_type, const TensorRow &ts_row,
                                            acltdtDataset *acl_dataset) {
  if (tdt_type != ACL_TENSOR_DATA_TENSOR || ts_row.size() == 0) {
    acltdtDataItem *acl_data = acltdtCreateDataItem(tdt_type, nullptr, 0, ACL_BOOL, nullptr, 0);
    if (acl_data == nullptr) {
      RETURN_STATUS_UNEXPECTED("Create data item failed when send data with type:" + std::to_string(tdt_type));
    }
    if (acltdtAddDataItem(acl_dataset, acl_data) != ACL_SUCCESS) {
      if (acltdtDestroyDataItem(acl_data) != ACL_SUCCESS) {
        MS_LOG(ERROR) << "Destroy data item failed when send data with type: " << tdt_type;
      }
      RETURN_STATUS_UNEXPECTED("Add data item to tdt dataset failed when send data.");
    }
    return Status::OK();
  }

  for (auto ts : ts_row) {
    aclDataType datatype;
    acltdtDataItem *acl_data = nullptr;
    RETURN_IF_NOT_OK(getTdtType(ts->type(), datatype));

    TensorShape tsShape = ts->shape();
    std::string dataShapes = "[";
    for (auto dim : tsShape.AsVector()) {
      (void)dataShapes.append(std::to_string(dim)).append(",");
    }
    dataShapes.pop_back();
    (void)dataShapes.append("]");

    std::shared_ptr<void> dataPtr =
      std::shared_ptr<void>(reinterpret_cast<uchar *>(&(*ts->begin<uint8_t>())), [](const void *elem) {});
    size_t dataLen = ts->SizeInBytes();
    const dsize_t dims = tsShape.Rank();
    std::vector<int64_t> dataShape;
    for (auto i = 0; i < dims; i++) {
      dataShape.emplace_back(tsShape[i]);
    }
    acl_data = acltdtCreateDataItem(ACL_TENSOR_DATA_TENSOR, (tsShape.empty() ? nullptr : &dataShape[0]), dims, datatype,
                                    dataPtr.get(), dataLen);
    if (acl_data == nullptr) {
      RETURN_STATUS_UNEXPECTED("Create data item failed when send data.");
    }
    if (acltdtAddDataItem(acl_dataset, acl_data) != ACL_SUCCESS) {
      if (acltdtDestroyDataItem(acl_data) != ACL_SUCCESS) {
        MS_LOG(ERROR) << "Destroy data item failed when send data with type ACL_TENSOR_DATA_TENSOR.";
      }
      RETURN_STATUS_UNEXPECTED("Add data item to tdt dataset failed when send data.");
    }

    MS_LOG(DEBUG) << "TDT data type is TDT_TENSOR, tensor type is " << datatype << ", tensor shape is " << dataShapes
                  << ", data length is " << ts->Size() << ".";
  }

  return Status::OK();
}

Status TdtPlugin::DestroyAclDataset(acltdtDataset *acl_dataset, bool include_data_item) {
  if (include_data_item) {
    for (size_t i = 0; i < acltdtGetDatasetSize(acl_dataset); i++) {
      if (acltdtDestroyDataItem(acltdtGetDataItem(acl_dataset, i)) != ACL_SUCCESS) {
        RETURN_STATUS_UNEXPECTED("Destroy data item failed when send data.");
      }
    }
  }
  if (acltdtDestroyDataset(acl_dataset) != ACL_SUCCESS) {
    RETURN_STATUS_UNEXPECTED("Destroy tdt dataset failed when send data.");
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
