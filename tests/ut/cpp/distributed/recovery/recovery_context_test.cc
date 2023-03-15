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

#include "common/common_test.h"
#include "include/backend/distributed/recovery/recovery_context.h"
#include "utils/ms_utils.h"
#include "distributed/persistent/storage/file_io_utils.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace distributed {
namespace recovery {
constexpr char kEnvMSRole[] = "MS_ROLE";

class TestRecoveryContext : public UT::Common {
 public:
  TestRecoveryContext() = default;
  virtual ~TestRecoveryContext() = default;

  void SetUp() override {}
  void TearDown() override {}
};

/// Feature: test recovery context.
/// Description:Test all interfaces in the recovery context.
/// Expectation: The return value of the normal interface is as expected, and the exception branch is expected to catch
/// the exception.
TEST_F(TestRecoveryContext, all_interface) {
  common::SetEnv(kEnvEnableRecovery, "1");
  common::SetEnv(kEnvRecoveryInterval, "10");
  std::string recovery_path = "./recovery_path";
  if (!storage::FileIOUtils::IsFileOrDirExist(recovery_path)) {
    storage::FileIOUtils::CreateDir(recovery_path);
  }
  auto ret = FileUtils::GetRealPath(recovery_path.c_str());
  if (!ret.has_value()) {
    MS_LOG(EXCEPTION) << "Cannot get real path of recovery path.";
  }
  std::string real_recovery_path = ret.value();
  common::SetEnv(kEnvRecoveryPath, real_recovery_path.c_str());
  common::SetEnv(kEnvMSRole, "MS_SCHED");

  EXPECT_NO_THROW(RecoveryContext::GetInstance()->enable_recovery());

  EXPECT_EQ(RecoveryContext::GetInstance()->recovery_path(), real_recovery_path);
  EXPECT_EQ(RecoveryContext::GetInstance()->recovery_interval(), 10);

  EXPECT_NO_THROW(RecoveryContext::GetInstance()->SetCkptPath(real_recovery_path));
  EXPECT_NO_THROW(RecoveryContext::GetInstance()->GetCkptPath());

  EXPECT_EQ(RecoveryContext::GetInstance()->latest_ckpt_epoch(), -1);
  EXPECT_EQ(RecoveryContext::GetInstance()->latest_ckpt_step(), -1);
  EXPECT_EQ(RecoveryContext::GetInstance()->latest_ckpt_file(), std::string());

  EXPECT_NO_THROW(RecoveryContext::GetInstance()->set_need_reset(false));
  EXPECT_EQ(RecoveryContext::GetInstance()->need_reset(), false);

  EXPECT_NO_THROW(RecoveryContext::GetInstance()->set_need_sync_weight_to_device(false));
  EXPECT_EQ(RecoveryContext::GetInstance()->need_sync_weight_to_device(), false);

  EXPECT_NO_THROW(RecoveryContext::GetInstance()->set_global_rank_id(1));
  EXPECT_NO_THROW(RecoveryContext::GetInstance()->set_global_rank_size(2));

  EXPECT_THROW(RecoveryContext::GetInstance()->persistent_json(), std::runtime_error);

  auto real_recovery_config = real_recovery_path + "/config.json";
  EXPECT_EQ(remove(real_recovery_config.c_str()), 0);
  EXPECT_EQ(remove(real_recovery_path.c_str()), 0);
}
}  // namespace recovery
}  // namespace distributed
}  // namespace mindspore
