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

#ifndef PREDICT_COMMON_FLAG_PARSER_H_
#define PREDICT_COMMON_FLAG_PARSER_H_

#include <functional>
#include <map>
#include <utility>
#include <string>

#include "common/utils.h"
#include "common/option.h"

namespace mindspore {
namespace predict {
struct FlagInfo;

struct Nothing {};

class FlagParser {
 public:
  FlagParser() { AddFlag(&FlagParser::help, "help", "print usage message", false); }

  virtual ~FlagParser() = default;

  // only support read flags from command line
  virtual Option<std::string> ParseFlags(int argc, const char *const *argv, bool supportUnknown = false,
                                         bool supportDuplicate = false);
  std::string Usage(const Option<std::string> &usgMsg = Option<std::string>(None())) const;

  template <typename Flags, typename T1, typename T2>
  void AddFlag(T1 *t1, const std::string &flagName, const std::string &helpInfo, const T2 &t2);

  template <typename Flags, typename T1, typename T2>
  void AddFlag(T1 Flags::*t1, const std::string &flagName, const std::string &helpInfo, const T2 &t2);

  template <typename Flags, typename T>
  void AddFlag(T Flags::*t, const std::string &flagName, const std::string &helpInfo);

  // Option-type fields
  template <typename Flags, typename T>
  void AddFlag(Option<T> Flags::*t, const std::string &flagName, const std::string &helpInfo);
  bool help;

 protected:
  std::string binName;
  Option<std::string> usageMsg;

 private:
  struct FlagInfo {
    std::string flagName;
    bool isRequired;
    bool isBoolean;
    std::string helpInfo;
    bool isParsed;
    std::function<Option<Nothing>(FlagParser *, const std::string &)> parse;
  };

  inline void AddFlag(const FlagInfo &flag);

  // construct a temporary flag
  template <typename Flags, typename T>
  void ConstructFlag(Option<T> Flags::*t, const std::string &flagName, const std::string &helpInfo, FlagInfo *flag);

  // construct a temporary flag
  template <typename Flags, typename T1>
  void ConstructFlag(T1 Flags::*t1, const std::string &flagName, const std::string &helpInfo, FlagInfo *flag);

  Option<std::string> InnerParseFlags(std::multimap<std::string, Option<std::string>> *values);

  bool GetRealFlagName(const std::string &oriFlagName, std::string *flagName);

  std::map<std::string, FlagInfo> flags;
};

// convert to std::string
template <typename Flags, typename T>
Option<std::string> ConvertToString(T Flags::*t, const FlagParser &baseFlag) {
  const Flags *flag = dynamic_cast<Flags *>(&baseFlag);
  if (flag != nullptr) {
    return std::to_string(flag->*t);
  }

  return Option<std::string>(None());
}

// construct for a Option-type flag
template <typename Flags, typename T>
void FlagParser::ConstructFlag(Option<T> Flags::*t1, const std::string &flagName, const std::string &helpInfo,
                               FlagInfo *flag) {
  if (flag == nullptr) {
    MS_LOGE("FlagInfo is nullptr");
    return;
  }
  flag->flagName = flagName;
  flag->helpInfo = helpInfo;
  flag->isBoolean = typeid(T) == typeid(bool);
  flag->isParsed = false;
}

// construct a temporary flag
template <typename Flags, typename T>
void FlagParser::ConstructFlag(T Flags::*t1, const std::string &flagName, const std::string &helpInfo, FlagInfo *flag) {
  if (flag == nullptr) {
    MS_LOGE("FlagInfo is nullptr");
    return;
  }
  if (t1 == nullptr) {
    MS_LOGE("t1 is nullptr");
    return;
  }
  flag->flagName = flagName;
  flag->helpInfo = helpInfo;
  flag->isBoolean = typeid(T) == typeid(bool);
  flag->isParsed = false;
}

inline void FlagParser::AddFlag(const FlagInfo &flagItem) { flags[flagItem.flagName] = flagItem; }

template <typename Flags, typename T>
void FlagParser::AddFlag(T Flags::*t, const std::string &flagName, const std::string &helpInfo) {
  if (t == nullptr) {
    MS_LOGE("t1 is nullptr");
    return;
  }

  Flags *flag = dynamic_cast<Flags *>(this);
  if (flag == nullptr) {
    MS_LOGI("dynamic_cast failed");
    return;
  }

  FlagInfo flagItem;

  // flagItem is as a output parameter
  ConstructFlag(t, flagName, helpInfo, &flagItem);
  flagItem.parse = [t](FlagParser *base, const std::string &value) -> Option<Nothing> {
    Flags *flag = dynamic_cast<Flags *>(base);
    if (base != nullptr) {
      Option<T> ret = Option<T>(GenericParseValue<T>(value));
      if (ret.IsNone()) {
        return Option<Nothing>(None());
      } else {
        flag->*t = ret.Get();
      }
    }

    return Option<Nothing>(Nothing());
  };

  flagItem.isRequired = true;
  flagItem.helpInfo +=
    !helpInfo.empty() && helpInfo.find_last_of("\n\r") != helpInfo.size() - 1 ? " (default: " : "(default: ";
  flagItem.helpInfo += ")";

  // add this flag to a std::map
  AddFlag(flagItem);
}

template <typename Flags, typename T1, typename T2>
void FlagParser::AddFlag(T1 *t1, const std::string &flagName, const std::string &helpInfo, const T2 &t2) {
  if (t1 == nullptr) {
    MS_LOGE("t1 is nullptr");
    return;
  }

  FlagInfo flagItem;

  // flagItem is as a output parameter
  ConstructFlag(t1, flagName, helpInfo, flagItem);
  flagItem.parse = [t1](FlagParser *base, const std::string &value) -> Option<Nothing> {
    if (base != nullptr) {
      Option<T1> ret = Option<T1>(GenericParseValue<T1>(value));
      if (ret.IsNone()) {
        return Option<T1>(None());
      } else {
        *t1 = ret.Get();
      }
    }

    return Option<Nothing>(Nothing());
  };

  flagItem.isRequired = false;
  *t1 = t2;

  flagItem.helpInfo +=
    !helpInfo.empty() && helpInfo.find_last_of("\n\r") != helpInfo.size() - 1 ? " (default: " : "(default: ";
  flagItem.helpInfo += ToString(t2).Get();
  flagItem.helpInfo += ")";

  // add this flag to a std::map
  AddFlag(flagItem);
}

template <typename Flags, typename T1, typename T2>
void FlagParser::AddFlag(T1 Flags::*t1, const std::string &flagName, const std::string &helpInfo, const T2 &t2) {
  if (t1 == nullptr) {
    MS_LOGE("t1 is nullptr");
    return;
  }

  Flags *flag = dynamic_cast<Flags *>(this);
  if (flag == nullptr) {
    MS_LOGI("dynamic_cast failed");
    return;
  }

  FlagInfo flagItem;

  // flagItem is as a output parameter
  ConstructFlag(t1, flagName, helpInfo, &flagItem);
  flagItem.parse = [t1](FlagParser *base, const std::string &value) -> Option<Nothing> {
    Flags *flag = dynamic_cast<Flags *>(base);
    if (base != nullptr) {
      Option<T1> ret = Option<T1>(GenericParseValue<T1>(value));
      if (ret.IsNone()) {
        return Option<Nothing>(None());
      } else {
        flag->*t1 = ret.Get();
      }
    }

    return Option<Nothing>(Nothing());
  };

  flagItem.isRequired = false;
  flag->*t1 = t2;

  flagItem.helpInfo +=
    !helpInfo.empty() && helpInfo.find_last_of("\n\r") != helpInfo.size() - 1 ? " (default: " : "(default: ";
  flagItem.helpInfo += ToString(t2).Get();
  flagItem.helpInfo += ")";

  // add this flag to a std::map
  AddFlag(flagItem);
}

// option-type add flag
template <typename Flags, typename T>
void FlagParser::AddFlag(Option<T> Flags::*t, const std::string &flagName, const std::string &helpInfo) {
  if (t == nullptr) {
    MS_LOGE("t is nullptr");
    return;
  }

  Flags *flag = dynamic_cast<Flags *>(this);
  if (flag == nullptr) {
    MS_LOGE("dynamic_cast failed");
    return;
  }

  FlagInfo flagItem;
  // flagItem is as a output parameter
  ConstructFlag(t, flagName, helpInfo, &flagItem);
  flagItem.isRequired = false;
  flagItem.parse = [t](FlagParser *base, const std::string &value) -> Option<Nothing> {
    Flags *flag = dynamic_cast<Flags *>(base);
    if (base != nullptr) {
      Option<T> ret = Option<std::string>(GenericParseValue<T>(value));
      if (ret.IsNone()) {
        return Option<Nothing>(None());
      } else {
        flag->*t = Option<T>(Some(ret.Get()));
      }
    }

    return Option<Nothing>(Nothing());
  };

  // add this flag to a std::map
  AddFlag(flagItem);
}
}  // namespace predict
}  // namespace mindspore

#endif  // PREDICT_COMMON_FLAG_PARSER_H_
