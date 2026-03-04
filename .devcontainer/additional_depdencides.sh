sudo apt install ros-noetic-autoware-msgs ros-noetic-ros-numpy ros-noetic-derived-object-msgs ros-noetic-grid-map-core ros-noetic-grid-map-ros python3-empy ros-noetic-ackermann-msgs ros-noetic-grid-map-filters ros-noetic-grid-map-rviz-plugin -y

sudo apt install ros-noetic-lanelet2 ros-noetic-lanelet2-python ros-noetic-jsk-rviz-plugins -y

sudo apt-get install -y cuda-toolkit-11-8
sudo apt-get install -y \
    libnvinfer-dev libnvinfer-plugin-dev

pip3 install pickle5

pip3 install -r .devcontainer/requirements.txt