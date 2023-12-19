/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/device/tensordump_utils.h"
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include "debug/data_dump/npy_header.h"
#include "mindspore/core/utils/file_utils.h"
#include "plugin/device/ascend/hal/device/ascend_data_queue.h"

namespace mindspore::device::ascend {
namespace {
const size_t kMbufCapacitySize = 128;
const int32_t kMbufDestroyDelayTime = 500;

const std::map<aclDataType, TypeId> kTensorDumpAclDataTypeMap = {
  {ACL_INT8, TypeId::kNumberTypeInt8},       {ACL_UINT8, TypeId::kNumberTypeUInt8},
  {ACL_INT16, TypeId::kNumberTypeInt16},     {ACL_UINT16, TypeId::kNumberTypeUInt16},
  {ACL_INT32, TypeId::kNumberTypeInt32},     {ACL_UINT32, TypeId::kNumberTypeUInt32},
  {ACL_INT64, TypeId::kNumberTypeInt64},     {ACL_UINT64, TypeId::kNumberTypeUInt64},
  {ACL_FLOAT16, TypeId::kNumberTypeFloat16}, {ACL_FLOAT, TypeId::kNumberTypeFloat32},
  {ACL_DOUBLE, TypeId::kNumberTypeFloat64},  {ACL_BOOL, TypeId::kNumberTypeBool},
};

void SaveTensor2NPY(std::string file_name, mindspore::tensor::TensorPtr tensor_ptr) {
  std::string npy_header = GenerateNpyHeader(tensor_ptr->shape(), tensor_ptr->data_type());
  if (!npy_header.empty()) {
    ChangeFileMode(file_name, S_IWUSR);
    std::fstream output{file_name, std::ios::out | std::ios::trunc | std::ios::binary};
    if (!output.is_open()) {
      MS_LOG(ERROR) << "For 'TensorDump' ops, open " << file_name << " file failed, the args of 'file' is invalid.";
      return;
    }
    output << npy_header;
    (void)output.write(reinterpret_cast<const char *>(tensor_ptr->data_c()), SizeToLong(tensor_ptr->Size()));
    if (output.bad()) {
      output.close();
      MS_LOG(ERROR) << "For 'TensorDump' ops, write mem to " << file_name << " failed.";
      return;
    }
    output.close();
    ChangeFileMode(file_name, S_IRUSR);
  } else {
    MS_LOG(ERROR) << "For 'TensorDump' ops, the type of " << TypeIdToType(tensor_ptr->data_type())->ToString()
                  << " not support dump.";
  }
}

bool EndsWith(const std::string &s, const std::string &sub) {
  return s.rfind(sub) == (s.length() - sub.length()) ? true : false;
}

}  // namespace

AsyncFileWriter::AsyncFileWriter(size_t thread_nums) { threads.reserve(thread_nums); }

AsyncFileWriter::~AsyncFileWriter() {
  stop.store(true, std::memory_order_acq_rel);
  cv.notify_all();
  for (auto &thread : threads) {
    if (thread.joinable()) {
      MS_LOG(INFO) << "TensorDump join file writer threads";
      thread.join();
    }
  }
}

void AsyncFileWriter::Submit(std::function<void()> func) {
  if (!threads_started.exchange(true)) {
    MS_LOG(INFO) << "Create AsyncFileWriter threads.";
    for (size_t i = 0; i < threads.capacity(); ++i) {
      threads.emplace_back(&AsyncFileWriter::WorkerThread, this);
    }
  }
  {
    std::lock_guard<std::mutex> lock(queue_mutex);
    tasks.push(func);
  }
  cv.notify_one();
}

void AsyncFileWriter::WorkerThread() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      cv.wait(lock, [this] { return stop || !tasks.empty(); });
      if (stop && tasks.empty()) {
        return;
      }
      task = tasks.front();
      tasks.pop();
    }
    task();
  }
}

std::string TensorDumpUtils::TensorNameToArrayName(const std::string &tensor_path) {
  static size_t name_id = 0;
  std::string npy_suffix{".npy"};
  std::string separator{"_"};
  std::optional<std::string> parent_path;
  std::optional<std::string> file_name;
  FileUtils::SplitDirAndFileName(tensor_path, &parent_path, &file_name);
  if (!parent_path.has_value()) {
    parent_path = ".";
  }
  std::optional<std::string> realpath = FileUtils::CreateNotExistDirs(parent_path.value());
  std::optional<std::string> new_file_name = std::to_string(name_id++) + separator + file_name.value();
  if (!EndsWith(new_file_name.value(), npy_suffix)) {
    new_file_name.value() += npy_suffix;
  }
  std::optional<std::string> new_file_path;
  FileUtils::ConcatDirAndFileName(&realpath, &new_file_name, &new_file_path);
  MS_LOG(INFO) << "For 'TensorDump' ops, dump file path is " << new_file_path.value();
  return new_file_path.value();
}

TensorDumpUtils &TensorDumpUtils::GetInstance() {
  static TensorDumpUtils instance;
  return instance;
}

void TensorDumpUtils::AsyncSaveDatasetToNpyFile(acltdtDataset *acl_dataset) {
  std::string tensor_name = std::string{acltdtGetDatasetName(acl_dataset)};
  MS_LOG(INFO) << "For 'TensorDump' ops, acltdt received Tensor name is " << tensor_name;
  if (tensor_name.empty()) {
    MS_LOG(ERROR) << "For 'TensorDump' ops, the of args of 'file' is empty, skip this data.";
    return;
  }
  size_t acl_dataset_size = acltdtGetDatasetSize(acl_dataset);

  for (size_t i = 0; i < acl_dataset_size; i++) {
    acltdtDataItem *item = acltdtGetDataItem(acl_dataset, i);
    MS_EXCEPTION_IF_NULL(item);
    if (acltdtGetTensorTypeFromItem(item) == ACL_TENSOR_DATA_END_OF_SEQUENCE) {
      MS_LOG(INFO) << "end of sequence" << std::endl;
      break;
    }

    size_t dim_num = acltdtGetDimNumFromItem(item);
    void *acl_addr = acltdtGetDataAddrFromItem(item);
    size_t acl_data_size = acltdtGetDataSizeFromItem(item);
    aclDataType acl_data_type = acltdtGetDataTypeFromItem(item);

    auto acl_data = reinterpret_cast<uint8_t *>(acl_addr);
    if (acl_data_size > 0) {
      MS_EXCEPTION_IF_NULL(acl_data);
    }

    ShapeVector tensor_shape;
    tensor_shape.resize(dim_num);

    if (acltdtGetDimsFromItem(item, tensor_shape.data(), dim_num) != ACL_SUCCESS) {
      MS_LOG(ERROR) << "ACL failed to get dim-size from acl channel data";
      continue;
    }

    auto type_iter = kTensorDumpAclDataTypeMap.find(acl_data_type);
    if (type_iter == kTensorDumpAclDataTypeMap.end()) {
      MS_LOG(ERROR) << "For 'TensorDump' ops, the type of tensor not support: " << acl_data_type;
      continue;
    }
    auto type_id = type_iter->second;
    auto tensor_ptr = std::make_shared<mindspore::tensor::Tensor>(type_id, tensor_shape);
    auto file_name = TensorNameToArrayName(tensor_name);

    if (acl_data_size == 0) {
      file_writer.Submit(std::bind(SaveTensor2NPY, file_name, tensor_ptr));
    } else if (CopyDataToTensor(acl_data, tensor_ptr, acl_data_size)) {
      file_writer.Submit(std::bind(SaveTensor2NPY, file_name, tensor_ptr));
    } else {
      // do nothing
    }
  }
}

bool TensorDumpUtils::CopyDataToTensor(const uint8_t *src, mindspore::tensor::TensorPtr tensor_ptr, const size_t size) {
  MS_EXCEPTION_IF_NULL(src);
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  auto *dest = reinterpret_cast<uint8_t *>(tensor_ptr->data_c());
  MS_EXCEPTION_IF_NULL(dest);

  size_t dest_size = static_cast<size_t>(tensor_ptr->data().nbytes());
  auto cp_ret = memcpy_s(dest, dest_size, src, size);
  if (cp_ret != EOK) {
    MS_LOG(ERROR) << "TensorDump op failed to copy the memory to py::tensor. Error code is " << cp_ret;
    return false;
  }
  return true;
}

}  // namespace mindspore::device::ascend
