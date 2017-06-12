#!/bin/bash
check_process(){
        # check the args
        if [ "$1" = "" ];
        then
                return 0
        fi

        #PROCESS_NUM => get the process number regarding the given thread name
        PROCESS_NUM=$(ps -ef | grep "$1" | grep -v "grep" | wc -l)
        # for degbuging...
        if [ $PROCESS_NUM -eq 1 ];
        then
                return 1
        else
                return 0
        fi
}

# check wheter the instance of thread exsits
while [ 1 ] ; do
        echo 'begin checking...'
        check_process "superset runserver" # the thread name
        CHECK_RET=$?
        if [ $CHECK_RET -eq 0 ]; # none exist
        then
                cd /home/felipealvesdias/druid-0.10.0
                ./startCoruja.sh
                cd /home/felipealvesdias/.superset
                nohup superset runserver
        fi
        sleep 60
done