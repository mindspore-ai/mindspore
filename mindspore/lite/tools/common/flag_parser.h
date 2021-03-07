/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_COMMON_FLAG_PARSER_H
#define MINDSPORE_LITE_TOOLS_COMMON_FLAG_PARSER_H

#include <functional>
#include <map>
#include <utility>
#include <string>
#include "src/common/utils.h"
#include "tools/common/option.h"

namespace mindspore {
namespace lite {
struct Nothing {};

class FlagParser {
 public:
  FlagParser() { AddFlag(&FlagParser::help, helpStr, "print usage message", ""); }

  virtual ~FlagParser() = default;

  // only support read flags from command line
  virtual Option<std::string> ParseFlags(int argc, const char *const *argv, bool supportUnknown = false,
                                         bool supportDuplicate = false);
  std::string Usage(const Option<std::string> &usgMsg = Option<std::string>(None())) const;

  template <typename Flags, typename T1, typename T2>
  void AddFlag(T1 *t1, const std::string &flagName, const std::string &helpInfo, const T2 *t2);

  template <typename Flags, typename T1, typename T2>
  void AddFlag(T1 *t1, const std::string &flagName, const std::string &helpInfo, const T2 &t2);

  // non-Option type fields in class
  template <typename Flags, typename T1, typename T2>
  void AddFlag(T1 Flags::*t1, const std::string &flagName, const std::string &helpInfo, const T2 *t2);

  template <typename Flags, typename T1, typename T2>
  void AddFlag(T1 Flags::*t1, const std::string &flagName, const std::string &helpInfo, const T2 &t2);

  template <typename Flags, typename T>
  void AddFlag(T Flags::*t, const std::string &flagName, const std::string &helpInfo);

  // Option-type fields
  template <typename Flags, typename T>
  void AddFlag(Option<T> Flags::*t, const std::string &flagName, const std::string &helpInfo);
  bool help{};

 protected:
  template <typename Flags>
  void AddFlag(std::string Flags::*t1, const std::string &flagName, const std::string &helpInfo, const char *t2) {
    AddFlag(t1, flagName, helpInfo, std::string(t2));
  }

  std::string binName;
  Option<std::string> usageMsg;
  std::string helpStr = "help";

 private:
  struct FlagInfo {
    std::string flagName;
    bool isRequired = false;
    bool isBoolean = false;
    std::string helpInfo;
    bool isParsed = false;
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

  static bool GetRealFlagName(std::string *flagName, const std::string &oriFlagName);

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
    MS_LOG(ERROR) << "FlagInfo is nullptr";
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
    MS_LOG(ERROR) << "FlagInfo is nullptr";
    return;
  }
  if (t1 == nullptr) {
    MS_LOG(ERROR) << "t1 is nullptr";
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
    MS_LOG(ERROR) << "t1 is nullptr";
    return;
  }
  AddFlag(t, flagName, helpInfo, static_cast<const T *>(nullptr));
}

template <typename Flags, typename T1, typename T2>
void FlagParser::AddFlag(T1 Flags::*t1, const std::string &flagName, const std::string &helpInfo, const T2 &t2) {
  if (t1 == nullptr) {
    MS_LOG(ERROR) << "t1 is nullptr";
    return;
  }
  AddFlag(t1, flagName, helpInfo, &t2);
}

// just for test
template <typename Flags, typename T1, typename T2>
void AddFlag(T1 *t1, const std::string &flagName, const std::string &helpInfo, const T2 &t2) {
  if (t1 == nullptr) {
    MS_LOG(ERROR) << "t1 is nullptr";
    return;
  }
  AddFlag(t1, flagName, helpInfo, &t2);
}

template <typename Flags, typename T1, typename T2>
void FlagParser::AddFlag(T1 *t1, const std::string &flagName, const std::string &helpInfo, const T2 *t2) {
  if (t1 == nullptr) {
    MS_LOG(ERROR) << "t1 is nullptr";
    return;
  }

  FlagInfo flagItem;

  // flagItem is as an output parameter
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

  if (t2 != nullptr) {
    flagItem.isRequired = false;
    *t1 = *t2;
  }

  flagItem.helpInfo +=
    !helpInfo.empty() && helpInfo.find_last_of("\n\r") != helpInfo.size() - 1 ? " (default: " : "(default: ";
  if (t2 != nullptr) {
    flagItem.helpInfo += ToString(*t2).Get();
  }
  flagItem.helpInfo += ")";

  // add this flag to a std::map
  AddFlag(flagItem);
}

template <typename Flags, typename T1, typename T2>
void FlagParser::AddFlag(T1 Flags::*t1, const std::string &flagName, const std::string &helpInfo, const T2 *t2) {
  if (t1 == nullptr) {
    MS_LOG(ERROR) << "t1 is nullptr";
    return;
  }

  auto *flag = dynamic_cast<Flags *>(this);
  if (flag == nullptr) {
    return;
  }

  FlagInfo flagItem;

  // flagItem is as a output parameter
  ConstructFlag(t1, flagName, helpInfo, &flagItem);
  flagItem.parse = [t1](FlagParser *base, const std::string &value) -> Option<Nothing> {
    auto *flag = dynamic_cast<Flags *>(base);
    if (flag == nullptr) {
      return Option<Nothing>(None());
    }
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

  if (t2 != nullptr) {
    flagItem.isRequired = false;
    flag->*t1 = *t2;
  } else {
    flagItem.isRequired = true;
  }

  flagItem.helpInfo +=
    !helpInfo.empty() && helpInfo.find_last_of("\n\r") != helpInfo.size() - 1 ? " (default: " : "(default: ";
  if (t2 != nullptr) {
    flagItem.helpInfo += ToString(*t2).Get();
  }
  flagItem.helpInfo += ")";

  // add this flag to a std::map
  AddFlag(flagItem);
}

// option-type add flag
template <typename Flags, typename T>
void FlagParser::AddFlag(Option<T> Flags::*t, const std::string &flagName, const std::string &helpInfo) {
  if (t == nullptr) {
    MS_LOG(ERROR) << "t is nullptr";
    return;
  }

  auto *flag = dynamic_cast<Flags *>(this);
  if (flag == nullptr) {
    MS_LOG(ERROR) << "dynamic_cast failed";
    return;
  }

  FlagInfo flagItem;
  // flagItem is as a output parameter
  ConstructFlag(t, flagName, helpInfo, &flagItem);
  flagItem.isRequired = false;
  flagItem.parse = [t](FlagParser *base, const std::string &value) -> Option<Nothing> {
    auto *flag = dynamic_cast<Flags *>(base);
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
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_COMMON_FLAG_PARSER_H
