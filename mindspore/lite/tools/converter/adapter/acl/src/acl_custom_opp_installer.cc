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

#include "tools/converter/adapter/acl/src/acl_custom_opp_installer.h"

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <algorithm>

#include "src/common/log_util.h"

namespace mindspore {
namespace opt {
constexpr size_t kOutBufSize = 1024;

bool AclCustomOppInstaller::InstallCustomOpp(const std::string &custom_opp_path, const std::string &cann_opp_path) {
  MS_LOG(INFO) << "AclCustomOppInstaller::InstallCustomOpp: Install opp from " << custom_opp_path << " to "
               << cann_opp_path;

  if (!custom_opp_path.empty()) {
    // if set the custom_opp_path, install the custom opp to cann opp library
    if (!cann_opp_path.empty()) {
      // if set the cann_opp_path, set the ASCEND_OPP_PATH env to cann_opp_path for targe dir
      if (setenv("ASCEND_OPP_PATH", cann_opp_path.c_str(), 1) != 0) {
        MS_LOG(ERROR) << "AclCustomOppInstaller::InstallCustomOpp: Set ASCEND_OPP_PATH env failed";
        return false;
      }
    }

    // call the install.sh script to install custom opp
    FILE *fp;
    std::string install_path = custom_opp_path + "/install.sh";
    std::string cmd = "bash " + install_path;
    if ((fp = popen(cmd.c_str(), "r")) == NULL) {
      MS_LOG(ERROR) << "AclCustomOppInstaller::InstallCustomOpp: Install Custom Opp failed";
      return false;
    } else {
      char buf_ps[kOutBufSize];
      while (fgets(buf_ps, kOutBufSize, fp) != NULL) {
        printf("%s\n", buf_ps);
      }
    }
    pclose(fp);
  }

  return true;
}
}  // namespace opt
}  // namespace mindspore
