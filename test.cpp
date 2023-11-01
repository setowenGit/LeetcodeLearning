#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_map>

using namespace std;


int qSort(vector<int>& nums, int l, int r, int k){
    if(l >= r) {return nums[k];}
    int i = rand() % (r-l+1) + l;
    swap(nums[i], nums[r]);
    i = l - 1;
    for(int j = l; j <= r-1; j++){
        if(nums[j] <= nums[r]){
            i++;
            swap(nums[j], nums[i]);
        }
    }
    swap(nums[i+1], nums[r]);
    if(k <= i) {return qSort(nums, l, i, k);}
    else {return qSort(nums, i+2, r, k);}
}

int quickSort(vector<int>& nums, int k){
    return qSort(nums, 0, nums.size()-1, k);
}


int findKthLargest(vector<int>& nums, int k) {
    return quickSort(nums, k);
}


void adjust(vector<pair<int, int>>& nums, int cur, int len){
    while(2 * cur + 1 < len){
        int c_left = 2 * cur + 1;
        int c_right = 2 * cur + 2;
        int c_large = (c_right < len && nums[c_right].second < nums[c_left].second) ? c_right : c_left;
        if(nums[c_large].second < nums[cur].second){
            swap(nums[c_large], nums[cur]);
            cur = c_large;
        }
        else {break;}
    }
}


vector<int> topKFrequent(vector<int>& nums, int k) {
    unordered_map<int, int> nums_hash;// 存储数字及其出现的次数
    for(auto& n : nums){
        nums_hash[n]++;
    }
    vector<pair<int, int>> numsp;// 将哈希表放入数组中，用于排序
    for(auto& n : nums_hash){
        numsp.emplace_back(n);
    }
    vector<pair<int, int>> nums_heap;// 排序用到的堆，关于次数的小根堆
    for(int p = 0; p < numsp.size(); p++){
        // 固定堆的大小为k，当装入的元素数量小于k时，确保父节点一定小于其子节点
        if(nums_heap.size() < k){
            nums_heap.emplace_back(numsp[p]);
            int n = nums_heap.size();
            for(int i = n/2 - 1; i >= 0; i--){
                adjust(nums_heap, i, n);
            }
        }
        // 当堆已经满了后，比较堆顶元素和遍历到的元素，若遍历到的元素更大则取代原本的堆顶，再将新的堆顶向下调整
        else{
            if(numsp[p].second > numsp[0].second){
                nums_heap[0] = numsp[p];
                adjust(nums_heap, 0, nums_heap.size());
            }
        }
    }
    vector<int> ans;
    for(int i = 0; i < nums_heap.size(); i++){
        ans.emplace_back(nums_heap[i].first);
    }
    return ans;
}

int main(){
    vector<int> nums = {5,3,1,1,1,3,73,1};
    vector<int> c = topKFrequent(nums, 1);
    return 0;
}