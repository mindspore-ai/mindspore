package com.mindspore.config;

import java.util.logging.Logger;

public final class MindsporeLite {
    private static final Logger LOGGER = Logger.getLogger(MindsporeLite.class.toString());

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
