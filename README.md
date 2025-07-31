# Visuals
curl -sSL https://raw.githubusercontent.com/McShoothy/visuals/main/main.cpp -o main.cpp && sudo apt update && sudo apt install -y build-essential libopencv-dev libopencv-contrib-dev && g++ -std=c++17 -O3 main.cpp -o visuals $(pkg-config --cflags --libs opencv4) -lpthread && ./visuals# visuals
