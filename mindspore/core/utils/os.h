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

#ifndef MINDSPORE_CORE_UTILS_OS_H_
#define MINDSPORE_CORE_UTILS_OS_H_

#ifdef _MSC_VER
#include <io.h>
#include <process.h>
#include <BaseTsd.h>
#include "dirent.h"  // NOLINT
#else
#include <dirent.h>
#include <unistd.h>
#endif

#ifdef _MSC_VER

// https://docs.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=cmd
#ifndef PATH_MAX
#define PATH_MAX (260)
#endif

#ifndef F_OK
#define F_OK (0) /* Check for file existence */
#endif

#ifndef X_OK
#define X_OK (1) /* Check for execute permission. */
#endif

#ifndef W_OK
#define W_OK (2) /* Check for write permission */
#endif

#ifndef R_OK
#define R_OK (4) /* Check for read permission */
#endif

using mode_t = int;

using ssize_t = SSIZE_T;

using off64_t = off_t;

using pid_t = int;

#ifndef _S_IRWXU
#define _S_IRWXU (_S_IREAD | _S_IWRITE | _S_IEXEC)
#endif

#ifndef S_IRWXU
#define S_IRWXU (_S_IRWXU)
#endif

#ifndef S_IRWXG
#define S_IRWXG (S_IRWXU >> 3)
#endif

#ifndef S_IRWXO
#define S_IRWXO (S_IRWXG >> 3)
#endif
#endif  // _MSC_VER

#ifndef _WIN32
#define PATH_SEPARATOR '/'
#else
#define PATH_SEPARATOR '\\'
#endif

#endif  // MINDSPORE_CORE_UTILS_OS_H_
