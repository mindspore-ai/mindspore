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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_TDT_TDT_PLUGIN_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_TDT_TDT_PLUGIN_H_

#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "acl/acl_tdt.h"
#include "minddata/dataset/engine/tdt/tdt_handle.h"

#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_row.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

class TdtPlugin {
 public:
  static std::shared_ptr<TdtPlugin> GetInstance();

  Status hostPush(TensorRow ts_row, bool is_wait, std::string channel_name, bool profilig, int32_t &time,
                  acltdtTensorType tdt_type = ACL_TENSOR_DATA_TENSOR);

  TdtPlugin(const std::string &channel_name, int32_t device_id);

  ~TdtPlugin();

 private:
  Status DestroyAclDataset(acltdtDataset *acl_dataset, bool include_data_item = true);

  Status AssembleTensor2AclDataset(acltdtTensorType tdt_type, const TensorRow &ts_row, acltdtDataset *acl_dataset);

  Status getTdtType(DataType d_type, aclDataType &datatype);

  Status translate(acltdtTensorType tdt_type, const TensorRow &ts_row, acltdtDataset **output_acl_dataset);

  void *tdt_handle_ = nullptr;

  acltdtChannelHandle *acl_handle_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_TDT_TDT_PLUGIN_H_
