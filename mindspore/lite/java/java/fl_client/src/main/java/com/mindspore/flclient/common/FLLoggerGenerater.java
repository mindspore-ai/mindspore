package com.mindspore.flclient.common;

import java.util.logging.*;

/**
 * Generate Logger for FL
 *
 * @author : zhangzhaoju
 * @since : 2022/4/14
 */
public class FLLoggerGenerater {
    private static Level level = Level.INFO;
    private static final AnonymousFilter filter = new AnonymousFilter();

    public static void setAnonymousFlg(boolean anonymousFlg) {
        AnonymousFilter.anonymousFlg = anonymousFlg;
    }

    /**
     * Log filter for anonymous
     *
     * @author : zhangzhaoju
     * @since : 2022/4/14
     */
    public static class AnonymousFilter implements Filter {
        private static boolean anonymousFlg = true;
        private final MsgAnonymous anonymous = new MsgAnonymous();

        public boolean isLoggable(LogRecord record) {
            String originMsg = record.getMessage();
            String anoyedMsg = anonymousFlg ? anonymous.doAnonymous(originMsg) : originMsg;
            record.setMessage("<FLClient> " + anoyedMsg);
            return true;
        }
    }

    /**
     * Create Logger with anonymousFilter
     *
     * @param clsName
     * @return
     */
    public static Logger getModelLogger(String clsName) {
        Logger logger = Logger.getLogger(clsName);
        logger.setFilter(filter);
        return logger;
    }
}
