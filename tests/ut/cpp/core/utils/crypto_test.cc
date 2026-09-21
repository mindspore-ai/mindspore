/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "common/common_test.h"
#include "utils/crypto.h"

#if !defined(_MSC_VER) && !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace mindspore {
#if !defined(_MSC_VER) && !defined(_WIN32)
namespace {
constexpr Byte kTestKey[] = "0123456789012345";  // 16-byte AES key
constexpr size_t kTestKeyLen = sizeof(kTestKey) - 1;
constexpr int32_t kIvLen = 16;  // AES_BLOCK_SIZE, the fixed iv length required by ParseEncryptData

void AppendInt32(std::vector<Byte> *buf, int32_t v) {
  const auto *p = reinterpret_cast<const Byte *>(&v);
  buf->insert(buf->end(), p, p + sizeof(v));
}

// Full 24-byte block header: iv_len(4B) + iv(16B) + cipher_len(4B), with cipher_len = 1
std::vector<Byte> MakeBlockHeader() {
  std::vector<Byte> header;
  AppendInt32(&header, kIvLen);
  header.insert(header.end(), kIvLen, 0);
  AppendInt32(&header, 1);
  return header;
}

// Cipher data layout of AES-CBC: magic(4B) + block_size(4B) + block
std::vector<Byte> MakeAesCbcData(size_t block_size, const std::vector<Byte> &block) {
  std::vector<Byte> data;
  AppendInt32(&data, static_cast<int32_t>(CBC_MAGIC_NUM));
  AppendInt32(&data, static_cast<int32_t>(block_size));
  data.insert(data.end(), block.begin(), block.end());
  return data;
}

bool WriteFile(const std::string &path, const std::vector<Byte> &content) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) {
    return false;
  }
  f.write(reinterpret_cast<const char *>(content.data()), static_cast<std::streamsize>(content.size()));
  return f.good();
}

std::vector<Byte> MakePlainData(size_t len) {
  std::vector<Byte> plain(len);
  for (size_t i = 0; i < len; ++i) {
    plain[i] = static_cast<Byte>(i % 251);
  }
  return plain;
}
}  // namespace

class TestCryptoDecrypt : public UT::Common {
 public:
  TestCryptoDecrypt() = default;
  void SetUp() {}
};

// Feature: Decrypt file boundary check
// Description: Decrypting an empty file should fail instead of returning an empty buffer.
// Expectation: Nullptr is returned without crash.
TEST_F(TestCryptoDecrypt, DecryptFileRejectsEmptyFile) {
  const std::string path = "crypto_ut_empty.ckpt";
  ASSERT_TRUE(WriteFile(path, {}));

  size_t decrypt_len = 0;
  auto decrypt_data = Decrypt(&decrypt_len, path, kTestKey, kTestKeyLen, "AES-GCM");
  EXPECT_EQ(decrypt_data, nullptr);

  (void)std::remove(path.c_str());
}

// Feature: Decrypt file boundary check
// Description: Decrypting a file larger than MAX_DECRYPT_FILE_SIZE should fail before allocation.
// Expectation: Nullptr is returned without out-of-memory or crash.
TEST_F(TestCryptoDecrypt, DecryptFileRejectsOversizedFile) {
  const std::string path = "crypto_ut_oversized.ckpt";
  int fd = open(path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0600);
  ASSERT_GE(fd, 0);
  // Create a sparse file of MAX_DECRYPT_FILE_SIZE + 1 bytes, no actual disk usage.
  ASSERT_EQ(ftruncate(fd, static_cast<off_t>(MAX_DECRYPT_FILE_SIZE + 1)), 0);
  (void)close(fd);

  size_t decrypt_len = 0;
  auto decrypt_data = Decrypt(&decrypt_len, path, kTestKey, kTestKeyLen, "AES-GCM");
  EXPECT_EQ(decrypt_data, nullptr);

  (void)std::remove(path.c_str());
}

// Feature: Decrypt data boundary check
// Description: A truncated block whose header cannot hold iv_len, iv and cipher_len must be rejected
//              before header parsing, otherwise out-of-bounds reads occur.
// Expectation: Nullptr is returned without crash.
TEST_F(TestCryptoDecrypt, DecryptDataRejectsTruncatedBlock) {
  const std::vector<Byte> header = MakeBlockHeader();
  for (size_t block_size : {size_t(1), size_t(4), size_t(23)}) {
    std::vector<Byte> block(header.begin(), header.begin() + static_cast<std::ptrdiff_t>(block_size));
    auto data = MakeAesCbcData(block_size, block);

    size_t decrypt_len = 0;
    auto decrypt_data = Decrypt(&decrypt_len, data.data(), data.size(), kTestKey, kTestKeyLen, "AES-CBC");
    EXPECT_EQ(decrypt_data, nullptr) << "block_size: " << block_size;
  }
}

// Feature: Encrypt and decrypt
// Description: Regression test that valid data can still be encrypted and decrypted via both the
//              data path and the file path.
// Expectation: Decrypted content equals the original plain content.
TEST_F(TestCryptoDecrypt, EncryptDecryptRoundtrip) {
  const auto plain = MakePlainData(1024);
  for (const std::string &mode : {"AES-GCM", "AES-CBC", "SM4-CBC"}) {
    size_t encrypt_len = 0;
    auto cipher = Encrypt(&encrypt_len, plain.data(), plain.size(), kTestKey, kTestKeyLen, mode);
    ASSERT_NE(cipher, nullptr) << "mode: " << mode;
    ASSERT_GT(encrypt_len, 0) << "mode: " << mode;

    size_t decrypt_len = 0;
    auto decrypt_data = Decrypt(&decrypt_len, cipher.get(), encrypt_len, kTestKey, kTestKeyLen, mode);
    ASSERT_NE(decrypt_data, nullptr) << "mode: " << mode;
    ASSERT_EQ(decrypt_len, plain.size()) << "mode: " << mode;
    EXPECT_EQ(std::memcmp(decrypt_data.get(), plain.data(), plain.size()), 0) << "mode: " << mode;

    const std::string path = "crypto_ut_roundtrip.ckpt";
    ASSERT_TRUE(WriteFile(path, std::vector<Byte>(cipher.get(), cipher.get() + encrypt_len)));

    size_t file_decrypt_len = 0;
    auto file_decrypt_data = Decrypt(&file_decrypt_len, path, kTestKey, kTestKeyLen, mode);
    ASSERT_NE(file_decrypt_data, nullptr) << "mode: " << mode;
    ASSERT_EQ(file_decrypt_len, plain.size()) << "mode: " << mode;
    EXPECT_EQ(std::memcmp(file_decrypt_data.get(), plain.data(), plain.size()), 0) << "mode: " << mode;

    (void)std::remove(path.c_str());
  }
}
#endif
}  // namespace mindspore
