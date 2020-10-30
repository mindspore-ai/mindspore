~/cache/cache_admin --start
session_id=$(~/cache/cache_admin -g | awk '{print $NF}')
export SESSION_ID=${session_id}
pytest dataset/test_cache_nomap.py::test_cache_nomap_server_stop &
pid=("$!")

sleep 2
~/cache/cache_admin --stop
sleep 1
wait ${pid}
