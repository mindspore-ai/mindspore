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

#include "tools/common/flag_parser.h"
#include "src/common/log_adapter.h"

namespace mindspore {
namespace lite {
// parse flags read from command line
Option<std::string> FlagParser::ParseFlags(int argc, const char *const *argv, bool supportUnknown,
                                           bool supportDuplicate) {
  MS_ASSERT(argv != nullptr);
  const int FLAG_PREFIX_LEN = 2;
  binName = GetFileName(argv[0]);

  std::multimap<std::string, Option<std::string>> keyValues;
  for (int i = 1; i < argc; i++) {
    std::string tmp = argv[i];
    Trim(&tmp);
    const std::string flagItem(tmp);

    if (flagItem == "--") {
      break;
    }

    if (flagItem.find("--") == std::string::npos) {
      return Option<std::string>("Failed: flag " + flagItem + " is not valid.");
    }

    std::string key;
    Option<std::string> value = Option<std::string>(None());

    size_t pos = flagItem.find_first_of('=');
    if (pos == std::string::npos) {
      key = flagItem.substr(FLAG_PREFIX_LEN);
    } else {
      key = flagItem.substr(FLAG_PREFIX_LEN, pos - FLAG_PREFIX_LEN);
      value = Option<std::string>(flagItem.substr(pos + 1));
    }

    keyValues.insert(std::pair<std::string, Option<std::string>>(key, value));
  }

  Option<std::string> ret = Option<std::string>(InnerParseFlags(&keyValues));
  if (ret.IsSome()) {
    return Option<std::string>(ret.Get());
  }

  return Option<std::string>(None());
}

bool FlagParser::GetRealFlagName(std::string *flagName, const std::string &oriFlagName) {
  MS_ASSERT(flagName != nullptr);
  const int BOOL_TYPE_FLAG_PREFIX_LEN = 3;
  bool opaque = false;
  if (StartsWithPrefix(oriFlagName, "no-")) {
    *flagName = oriFlagName.substr(BOOL_TYPE_FLAG_PREFIX_LEN);
    opaque = true;
  } else {
    *flagName = oriFlagName;
  }
  return opaque;
}

// Inner parse function
Option<std::string> FlagParser::InnerParseFlags(std::multimap<std::string, Option<std::string>> *keyValues) {
  MS_ASSERT(keyValues != nullptr);
  for (auto &keyValue : *keyValues) {
    std::string flagName;
    bool opaque = GetRealFlagName(&flagName, keyValue.first);
    Option<std::string> flagValue = keyValue.second;

    auto item = flags.find(flagName);
    if (item == flags.end()) {
      return Option<std::string>(std::string(flagName + " is not a valid flag"));
    }
    FlagInfo *flag = &(item->second);
    if (flag == nullptr) {
      return Option<std::string>("Failed: flag is nullptr");
    }
    if (flag->isParsed) {
      return Option<std::string>("Failed: already parsed flag: " + flagName);
    }
    std::string tmpValue;
    if (!flag->isBoolean) {
      if (opaque) {
        return Option<std::string>(flagName + " is not a boolean type");
      }
      if (flagValue.IsNone()) {
        return Option<std::string>("No value provided for non-boolean type: " + flagName);
      }
      tmpValue = flagValue.Get();
    } else {
      if (flagValue.IsNone() || flagValue.Get().empty()) {
        tmpValue = !opaque ? "true" : "false";
      } else if (!opaque) {
        tmpValue = flagValue.Get();
      } else {
        return Option<std::string>(std::string("Boolean flag can not have non-empty value"));
      }
    }
    // begin to parse value
    Option<Nothing> ret = flag->parse(this, tmpValue);
    if (ret.IsNone()) {
      return Option<std::string>("Failed to parse value for: " + flag->flagName);
    }
    flag->isParsed = true;
  }

  // to check flags not given in command line but added as in constructor
  for (auto &flag : flags) {
    if (flag.second.isRequired && !flag.second.isParsed) {
      return Option<std::string>("Error, value of '" + flag.first + "' not provided");
    }
  }

  return Option<std::string>(None());
}

void ReplaceAll(std::string *str, const std::string &oldValue, const std::string &newValue) {
  if (str == nullptr) {
    MS_LOG(ERROR) << "Input str is nullptr";
    return;
  }
  while (true) {
    std::string::size_type pos(0);
    if ((pos = str->find(oldValue)) != std::string::npos) {
      str->replace(pos, oldValue.length(), newValue);
    } else {
      break;
    }
  }
}

std::string FlagParser::Usage(const Option<std::string> &usgMsg) const {
  // first line, brief of the usage
  std::string usageString = usgMsg.IsSome() ? usgMsg.Get() + "\n" : "";
  // usage of bin name
  usageString += usageMsg.IsNone() ? "\nusage: " + binName + " [options]\n" : usageMsg.Get() + "\n";
  // help line of help message, usageLine:message of parameters
  std::string helpLine;
  std::string usageLine;
  uint32_t i = 0;
  for (auto flag = flags.begin(); flag != flags.end(); flag++) {
    std::string flagName = flag->second.flagName;
    std::string helpInfo = flag->second.helpInfo;
    // parameter line
    std::string thisLine = flagName == "help" ? " --" + flagName : " --" + flagName + "=VALUE";
    if (++i <= flags.size()) {
      // add parameter help message of each line
      thisLine += " " + helpInfo;
      ReplaceAll(&helpInfo, "\n\r", "\n");
      usageLine += thisLine + "\n";
    } else {
      // brief help message
      helpLine = thisLine + " " + helpInfo + "\n";
    }
  }
  // total usage is brief of usage+ brief of bin + help message + brief of
  // parameters
  return usageString + helpLine + usageLine;
}
}  // namespace lite
}  // namespace mindspore
