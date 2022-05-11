/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_RECOVERY_CONFIGURATION_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_RECOVERY_CONFIGURATION_H_

#include <string>

namespace mindspore {
namespace distributed {
namespace recovery {
// An abstract configuration class to store and recover key-value style metadata, which could be stored in a local file
// or other storages.
class Configuration {
 public:
  Configuration() = default;
  virtual ~Configuration() = default;

  // These two methods should be implemented in sub-class to allocate and release resources owned by this configuration
  // instance.
  virtual bool Initialize() = 0;
  virtual bool Finalize() { return true; }

  // Get the configuration item value by the specified key, returns the default value if the key does not
  // exist.
  virtual std::string Get(const std::string &key, const std::string &defaultvalue) const = 0;

  // Persist the key-value pair metadata.
  virtual void Put(const std::string &key, const std::string &value) = 0;

  // Check whether the specified configuration key exists.
  virtual bool Exists(const std::string &key) const = 0;

  // Check whether the configuration contains any key-value pairs.
  virtual bool Empty() const = 0;

  // Flush all the key-value pairs in memory into the specific sub-class's storage.
  virtual bool Flush() = 0;
};
}  // namespace recovery
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_RECOVERY_CONFIGURATION_H_
