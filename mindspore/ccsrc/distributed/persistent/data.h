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

#ifndef MIINDSPORE_CCSRC_DISTRIBUTED_PERSISTENT_DATA_H_
#define MIINDSPORE_CCSRC_DISTRIBUTED_PERSISTENT_DATA_H_

#include <map>
#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <utility>

#include "distributed/persistent/storage/local_file.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace distributed {
namespace persistent {
// The data class is used to save and manage the tensor in memory, and provides
// interfaces for persistence and disaster recovery.
template <typename T>
class Data {
 public:
  explicit Data(const std::shared_ptr<std::vector<T>> &data, const std::shared_ptr<std::vector<int>> &shape = nullptr)
      : data_(data), shape_(shape) {}

  virtual ~Data() = default;

  // Get the memory data of Data
  T *data() const {
    MS_EXCEPTION_IF_NULL(data_);
    return data_->data();
  }

  // Get the mutable memory data of Data
  std::shared_ptr<std::vector<T>> MutableData() const { return data_; }

  // Get the element number of Data
  size_t size() const {
    MS_EXCEPTION_IF_NULL(data_);
    return data_->size();
  }

  // Get the dimension information of Data.
  std::shared_ptr<std::vector<int>> shape() const { return shape_; }

 protected:
  // Container used to store continuous memory buffer of Data.
  std::shared_ptr<std::vector<T>> data_;

  // Container used to record the dimension information of Data which persists a tensor.
  std::shared_ptr<std::vector<int>> shape_;
};

// Implementation of the class Data to complete the function of persistence and disaster tolerance.
template <typename T>
class PersistentData : public Data<T> {
 public:
  explicit PersistentData(const std::shared_ptr<std::vector<T>> &data,
                          const std::shared_ptr<std::vector<int>> &shape = nullptr)
      : Data<T>(data, shape) {}

  ~PersistentData() override = default;

  // Initialize storage module.
  // Custom storage config, you can choose different configurations according to different storage forms,
  // such as using file storage by configuring the file storage path,
  // and config can be like this: std::map<std::string, std::string> config = {{kFileStoragePath, "real_path_of_dir"}};
  void Initialize(const std::map<std::string, std::string> &storage_config);

  // In disaster recovery mode, memory of tensor need to be saved into disk file periodically.
  void Persist(const storage::DirtyInfo &dirty_info) const;

  // In disaster recovery mode, server node or worker node need to restore persistent data when restart.
  void Restore() const;

 private:
  // The following variables are used in disaster recovery mode:
  // The threads used to execute persistence task.
  std::thread persist_thread_;

  // The file storage handle used to persist data.
  std::shared_ptr<storage::StorageBase> storage_;
};

template <typename T>
void PersistentData<T>::Initialize(const std::map<std::string, std::string> &storage_config) {
  storage_ = std::make_shared<storage::LocalFile>(storage_config);
}

template <typename T>
void PersistentData<T>::Persist(const storage::DirtyInfo &dirty_info) const {
  MS_EXCEPTION_IF_NULL(storage_);
  storage::InputData input = std::make_tuple(*Data<T>::shape_, Data<T>::data(), Data<T>::size() * sizeof(T));
  storage_->Write(input, dirty_info);
}

template <typename T>
void PersistentData<T>::Restore() const {
  storage::OutputData output = std::make_pair(Data<T>::data(), Data<T>::size() * sizeof(T));
  MS_EXCEPTION_IF_NULL(storage_);
  storage_->Read(output);
}
}  // namespace persistent
}  // namespace distributed
}  // namespace mindspore

#endif  // MIINDSPORE_CCSRC_DISTRIBUTED_PERSISTENT_DATA_H_
