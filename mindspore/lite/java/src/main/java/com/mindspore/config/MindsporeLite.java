/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
 * MSLite Init Class
 *
 * @since v1.0
 */
public final class MindsporeLite {
    private static final Object lock = new Object();
    private static Logger LOGGER = GetLogger();

    public static Logger GetLogger() {
        if (LOGGER == null) {
            synchronized (lock) {
                if (LOGGER == null) {
                    LOGGER = Logger.getLogger(MindsporeLite.class.toString());
                }
            }
        }
        return LOGGER;
    }

    /**
     * Init function.
     */
    public static void init() {
        LOGGER.info("MindsporeLite init load ...");
        try {
            NativeLibrary.load();
        } catch (Exception e) {
            LOGGER.severe("Failed to load MindSporLite native library.");
            throw e;
        }
    }

    static {
        LOGGER.info("MindsporeLite init ...");
        init();
    }
}
