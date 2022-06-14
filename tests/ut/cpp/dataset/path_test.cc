/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/util/path.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include <cstdio>

using namespace mindspore::dataset;

class MindDataTestPath : public UT::Common {
 public:
    MindDataTestPath() {}
};

/// Feature: Path
/// Description: Test Path on a directory and on jpeg file extension
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPath, Test1) {
  Path f("/tmp");
  ASSERT_TRUE(f.Exists());
  ASSERT_TRUE(f.IsDirectory());
  ASSERT_EQ(f.ParentPath(), "/");
  // Print out the first few items in the directory
  auto dir_it = Path::DirIterator::OpenDirectory(&f);
  ASSERT_NE(dir_it.get(), nullptr);
  int i = 0;
  while (dir_it->HasNext()) {
    Path v = dir_it->Next();
    MS_LOG(DEBUG) << v.ToString() << "\n";
    i++;
    if (i == 10) {
      break;
    }
  }
  // Test extension.
  Path g("file.jpeg");
  MS_LOG(DEBUG) << g.Extension() << "\n";
  ASSERT_EQ(g.Extension(), ".jpeg");
}

/// Feature: Path
/// Description: Test Path with various assignments using a Path, on empty string, std::move on Path
///     and with concatenation
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPath, Test2) {
  Path p("/tmp");
  Path p2(p);
  ASSERT_TRUE(p2.Exists());
  ASSERT_TRUE(p2.IsDirectory());
  ASSERT_EQ(p2.ParentPath(), "/");

  p2 = p;
  ASSERT_TRUE(p2.Exists());
  ASSERT_TRUE(p2.IsDirectory());
  ASSERT_EQ(p2.ParentPath(), "/");

  Path p3("");
  p3 = std::move(p2);
  ASSERT_TRUE(p3.Exists());
  ASSERT_TRUE(p3.IsDirectory());
  ASSERT_EQ(p3.ParentPath(), "/");

  Path p4(std::move(p3));
  ASSERT_TRUE(p4.Exists());
  ASSERT_TRUE(p4.IsDirectory());
  ASSERT_EQ(p4.ParentPath(), "/");

  Path p5("/");
  std::string s = "tmp";
  ASSERT_TRUE((p5 + "tmp").Exists());
  ASSERT_TRUE((p5 + "tmp").IsDirectory());
  ASSERT_EQ((p5 + "tmp").ParentPath(), "/");

  ASSERT_TRUE((p5 / "tmp").Exists());
  ASSERT_TRUE((p5 / "tmp").IsDirectory());
  ASSERT_EQ((p5 / "tmp").ParentPath(), "/");

  ASSERT_TRUE((p5 + s).Exists());
  ASSERT_TRUE((p5 + s).IsDirectory());
  ASSERT_EQ((p5 + s).ParentPath(), "/");

  ASSERT_TRUE((p5 / s).Exists());
  ASSERT_TRUE((p5 / s).IsDirectory());
  ASSERT_EQ((p5 / s).ParentPath(), "/");

  Path p6("tmp");
  ASSERT_TRUE((p5 + p6).Exists());
  ASSERT_TRUE((p5 + p6).IsDirectory());
  ASSERT_EQ((p5 + p6).ParentPath(), "/");
  ASSERT_TRUE((p5 / p6).Exists());
  ASSERT_TRUE((p5 / p6).IsDirectory());
  ASSERT_EQ((p5 / p6).ParentPath(), "/");
  p5 += p6;
  ASSERT_TRUE(p5.Exists());
  ASSERT_TRUE(p5.IsDirectory());
  ASSERT_EQ(p5.ParentPath(), "/");

  Path p7("/");
  Path p8(p7);
  p7 += s;
  p8 += "tmp";
  ASSERT_TRUE(p7.Exists());
  ASSERT_TRUE(p7.IsDirectory());
  ASSERT_EQ(p7.ParentPath(), "/");
  ASSERT_TRUE(p8.Exists());
  ASSERT_TRUE(p8.IsDirectory());
  ASSERT_EQ(p8.ParentPath(), "/");

  Path p9("/tmp/test_path");
  ASSERT_TRUE(p9.CreateDirectories().IsOk());
  ASSERT_EQ(remove("/tmp/test_path"), 0);
}
