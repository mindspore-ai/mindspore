package com.mindspore.config;

import java.util.logging.Logger;

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
