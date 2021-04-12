#include <array>
#include <iostream>
#include <queue>
#include <vector>
#include <cassert>


class Filtering{
private:
    static const int n_regions = 3;
    int current_region;
    int switch_delay;
    int queue_length;
    int cnt = 0;
    std::queue<int> regions;
    std::array<int, n_regions> frequencies;
public:
    Filtering(int n_frames, int delay){
        queue_length = n_frames;
        switch_delay = delay;
        std::fill(frequencies.begin(), frequencies.end(), 0);
    }
    void update(int new_region){
        if (regions.empty())
            current_region = new_region;
        if (regions.size() == queue_length){
            frequencies[regions.front()]--;
            regions.pop();
        }
        regions.push(new_region);
        frequencies[new_region]++;
        assert(frequencies[0]+frequencies[1]+frequencies[2] <= queue_length);
    }
    int get_region(){
        int max_val=-1, max_region=0;
        for(int i=0;i<n_regions;i++)
            if(frequencies[i]>max_val){
                max_val = frequencies[i];
                max_region = i;
            }
        if ((current_region != max_region) && (cnt < switch_delay))
            cnt++;
        if ((current_region != max_region) && (cnt == switch_delay)){
            cnt = 0;
            current_region = max_region;
        }
        if (current_region == max_region)
            cnt = 0;

        return current_region;
    }
    void print_freq(){
        for (int i = 0; i <3; i++)
            std::cout << i << ":" << frequencies[i] << " ";
        std::cout << "\n";
    }
};