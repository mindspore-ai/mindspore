/*
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

package com.mindspore.config;

import java.util.logging.Logger;

/**
 * Define mindspore version info.
 *
 * @since v1.0
 */
public class Version {
    private static final Logger LOGGER = MindsporeLite.GetLogger();
    static {
        LOGGER.info("Version init ...");
        init();
    }

    /**
     * Init function.
     */
    public static void init() {
        LOGGER.info("Version init load ...");
        try {
            NativeLibrary.load();
        } catch (UnsatisfiedLinkError e) {
            LOGGER.severe("Failed to load MindSporLite native library.");
        }
    }

    /**
     * Get MindSpore Lite version info.
     *
     * @return MindSpore Lite version info.
     */
    public static native String version();
}