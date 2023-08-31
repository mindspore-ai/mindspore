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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_AUTO_REGISTRATION_FACTORY_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_AUTO_REGISTRATION_FACTORY_H_

#include <unordered_map>

namespace mindspore::kernel {
template <typename KeyType, typename CreatorType>
class AutoRegistrationFactory {
 public:
  struct AutoRegister {
    AutoRegister(KeyType k, CreatorType creator) {
      AutoRegistrationFactory<KeyType, CreatorType>::Get().Insert(k, creator);
    }
  };
  static AutoRegistrationFactory<KeyType, CreatorType> &Get();
  bool HasKey(KeyType k) const { return key2creator_.find(k) != key2creator_.end(); }
  CreatorType GetCreator(KeyType k) { return key2creator_[k]; }

 private:
  bool Insert(KeyType k, CreatorType creator) {
    if (HasKey(k)) {
      return false;
    }
    return key2creator_.emplace(k, creator).second;
  }
  std::unordered_map<KeyType, CreatorType> key2creator_;
};

#define AUTO_REGISTRATION_FACTORY_JOIN(a, b) a##b

#define AUTO_REGISTRATION_FACTORY_UNIQUE_NAME_JOIN(a, b) AUTO_REGISTRATION_FACTORY_JOIN(a, b)

#define AUTO_REGISTRATION_FACTORY_UNIQUE_NAME AUTO_REGISTRATION_FACTORY_UNIQUE_NAME_JOIN(g_, __COUNTER__)

#define REGISTER_CLASS_CREATOR(KeyType, k, CreatorType, creator) \
  static AutoRegistrationFactory<KeyType, CreatorType>::AutoRegister AUTO_REGISTRATION_FACTORY_UNIQUE_NAME(k, creator);
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_AUTO_REGISTRATION_FACTORY_H_
