#include <iostream>
#include <vector>

using namespace std;

// 搜索第一个大于等于target的数的索引（左闭右闭写法）
int lower_bound1(vector<int>& nums, int target){
    //左闭右闭区间 [left, right]
    int left = 0;
    int right = nums.size() - 1;
    while(left <= right){//区间不为空则继续
        int mid = (left + right)/2;
        if(nums[mid] < target) {left = mid + 1;}//[mid+1, right]
        else {right = mid - 1;}//[left, mid-1]
    }
    return left;
}

// 搜索第一个大于等于target的数的索引（左闭右开写法）
int lower_bound2(vector<int>& nums, int target){
    //左闭右开区间 [left, right)
    int left = 0;
    int right = nums.size();
    while(left < right){//不为空则继续。由于right是开的，所以left=right时该区间为空，
        int mid = (left + right)/2;
        if(nums[mid] < target) {left = mid + 1;}//[mid+1, right)
        else {right = mid;}//[left, mid)
    }
    return left; //输出right也行
}

// 搜索第一个大于等于target的数的索引（左开右开写法）
int lower_bound3(vector<int>& nums, int target){
    //左开右开区间 (left, right)
    int left = -1;
    int right = nums.size();
    while(left + 1 < right){//不为空则继续。由于left和right是开的，所以left+1>=right时该区间为空
        int mid = (left + right)/2;
        if(nums[mid] < target) {left = mid;}//(mid, right)
        else {right = mid;}//(left, mid)
    }
    return right;
}




// 搜索第一个大于target的数的索引（左闭右闭写法）
int upper_bound1(vector<int>& nums, int target){
    //左闭右闭区间 [left, right]
    int left = 0;
    int right = nums.size() - 1;
    while(left <= right){//区间不为空则继续
        int mid = (left + right)/2;
        if(nums[mid] <= target) {left = mid + 1;}//[mid+1, right]
        else {right = mid - 1;}//[left, mid-1]
    }
    return left;
}

// 搜索第一个大于target的数的索引（左闭右开写法）
int upper_bound2(vector<int>& nums, int target){
    //左闭右开区间 [left, right)
    int left = 0;
    int right = nums.size();
    while(left < right){//不为空则继续。由于right是开的，所以left=right时该区间为空，
        int mid = (left + right)/2;
        if(nums[mid] <= target) {left = mid + 1;}//[mid+1, right)
        else {right = mid;}//[left, mid)
    }
    return left; //输出right也行
}

// 搜索第一个大于target的数的索引（左开右开写法）
int upper_bound3(vector<int>& nums, int target){
    //左开右开区间 (left, right)
    int left = -1;
    int right = nums.size();
    while(left + 1 < right){//不为空则继续。由于left和right是开的，所以left+1>=right时该区间为空
        int mid = (left + right)/2;
        if(nums[mid] <= target) {left = mid;}//(mid, right)
        else {right = mid;}//(left, mid)
    }
    return right;
}

 
int main(){
    vector<int> nums = {0, 1, 2, 3, 3, 3, 4, 5, 6};
    int ans = upper_bound3(nums, 3);
    return 0;
}