#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

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

int main(){
    vector<int> nums = {3,2,1,5,6,4};
    int c = findKthLargest(nums, 2);
    return 0;
}