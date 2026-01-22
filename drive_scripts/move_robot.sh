#cd .. || exit 1
cd hardware_launch_scripts || exit 1
 ./vesc.sh &

cd .. || exit 1
cd drive_scripts || exit 1

./mockodom.sh
