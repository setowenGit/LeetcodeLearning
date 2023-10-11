#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace std;

struct ListNode{
    int val;
    ListNode* next;
    ListNode() {this->val = 0; this->next = nullptr;}
    ListNode(int val) {this->val = val; this->next = nullptr;}
    ListNode(int val, ListNode* next) {this->val = val; this->next = next;}
};

ListNode* createList(vector<int>& nums){
    ListNode* pre = new ListNode();
    ListNode* cur = pre;
    for(auto& num:nums){
        cur->next = new ListNode(num);
        cur = cur->next;
    }
    return pre->next;
}

// 快速排序
void qSort(vector<int>& nums, int l, int r){
    if(l >= r) {return;}
    int i = rand() % (r - l + 1) + l;//随机选择主元
    swap(nums[r], nums[i]);
    int pivot = nums[r];
    i = l - 1;
    for(int j = l; j <= r - 1; j++){
        if(nums[j] <= pivot){
            i++;
            swap(nums[j], nums[i]);
        }
    }
    swap(nums[i+1], nums[r]);
    qSort(nums, l, i);
    qSort(nums, i + 2, r);
}

vector<int> quickSort(vector<int>& nums){
    qSort(nums, 0, nums.size()-1);
    return nums;
}

// 归并排序
void mSort(vector<int>& nums, int l, int r){
    if(l >= r) {return;}
    int mid = (l + r) / 2;
    mSort(nums, l, mid);
    mSort(nums, mid + 1, r);
    vector<int> res;//暂时存储有序数组
    int i = l, j = mid + 1;
    while(i <= mid && j <= r){
        if(nums[i] <= nums[j]) {res.emplace_back(nums[i]); i++;}
        else {res.emplace_back(nums[j]); j++;}
    }
    while(i <= mid) {res.emplace_back(nums[i]); i++;}
    while(j <= r) {res.emplace_back(nums[j]); j++;}
    for(int i = 0; i < r - l + 1; i ++){
        nums[i + l] = res[i];
    }
}

vector<int> mergeSort(vector<int>& nums){
    mSort(nums, 0, nums.size()-1);
    return nums;
}

// 堆排序
//以cur节点为出发点，调整cur及其以下的节点，以满足父节点大于其子节点
void adjust(vector<int>& nums, int cur, int len){
    while(2 * cur + 1 < len){//节点cur的左节点不超过len
        int c_left = 2 * cur + 1;//左子节点
        int c_right = 2 * cur + 2;//右子节点
        // 当右子节点存在且大于左子节点时，则右子节点为c_large，否则左子节点为c_large
        int c_large = (c_right < len && nums[c_right] > nums[c_left]) ? c_right : c_left;
        // 当父节点比c_large小时，c_large将成为父节点
        if(nums[c_large] > nums[cur]) {swap(nums[c_large], nums[cur]);}
        else {break;}//若父节点仍然大于其子节点， 则退出循环
        cur = c_large;
    }
}

vector<int> heapSort(vector<int>& nums){
    int n = nums.size();
    for(int i = n/2 - 1; i >= 0; i--){//从⼦节点是叶⼦节点的⽗节点开始向下调整，确保父节点一定大于其子节点
        adjust(nums, i, n);
    }
    for(int i = n - 1; i > 0; i--){//将堆顶节点挪到最末尾位置，然后选出新的堆顶
        swap(nums[0], nums[i]);
        adjust(nums, 0, i);//新的堆顶向下沉
    }
    return nums;
}

// 选择排序
//数组
vector<int> selectSort(vector<int>& nums){
    int n = nums.size();
    for(int i = 0; i < n - 1; i++){
        int min = i;
        for(int j = i + 1; j < n; j++){
            if(nums[j] < nums[min]) {min = j;}
        }
        swap(nums[i], nums[min]);
    }
    return nums;
}
//单向链表
ListNode* selectSort(ListNode* head){
    ListNode* slow;
    ListNode* fast;
    for(slow = head; slow != nullptr; slow = slow->next){
        for(fast = slow->next; fast != nullptr; fast = fast->next){
            if(slow->val > fast->val){
                int tmp = slow->val;
                slow->val = fast->val;
                fast->val = tmp;
            }
        }
    }
    return head;
}

// 冒泡排序
vector<int> bubbleSort(vector<int>& nums){
    int n = nums.size();
    for(int i = 0; i < n - 1; i++){
        for(int j = 0; j < n - i - 1; j++){
            if(nums[j] > nums[j+1]){
                swap(nums[j], nums[j+1]);
            }
        }
    }
}

// 插入排序
//数组
vector<int> insertSort(vector<int>& nums){
    int n = nums.size();
    for(int i = 1; i < n; i++){
        int tmp = nums[i];
        int j = i;
        while(j > 0 && nums[j - 1] > tmp){
            nums[j] = nums[j - 1];
            j --;
        }
        nums[j] = tmp;
    }
    return nums;
}

//单向链表
ListNode* insertSort(ListNode* head){
    if(head == nullptr) {return head;}
    ListNode* dummy = new ListNode();
    dummy->next = head;
    ListNode* pre = dummy, *cur = head->next, *final = head;
    while(final->next != nullptr){
        if(final->val <= cur->val){
            final = final->next;
        }
        else{
            pre = dummy;
            while(cur->val > pre->next->val){
                pre = pre->next;
            }
            final->next = cur->next;
            cur->next = pre->next;
            pre->next = cur;
        }
        cur = final->next;
    }
    return dummy->next;
}

// 希尔排序
vector<int> shellSort(vector<int>& nums){
    int n = nums.size();
    int gap = 1;
    while(gap < n / 3) {gap = gap * 3 + 1;}
    while(gap > 0){
        for(int i = gap; i < n; i++){
            int tmp = nums[i];
            int j = i;
            while(j >= gap && nums[j - gap] > tmp){
                nums[j] = nums[j - gap];
                j -= gap;
            }
            nums[j] = tmp;
        }
        gap = gap / 3;
    }
    return nums;
}

// 计数排序
vector<int> countingSort(vector<int>& nums){
    int n = nums.size();
    if(n == 0) {return nums;}
    int min_ele = *min_element(nums.begin(), nums.end());
    int max_ele = *max_element(nums.begin(), nums.end());
    vector<int> count_nums(max_ele - min_ele + 1, 0);
    for(auto num : nums){
        count_nums[num - min_ele] ++;
    }
    int j = 0;
    for(int i = 0; i < n; i++){
        while(count_nums[j] == 0) {j ++;}
        nums[i] = j + min_ele;
        count_nums[j] --;
    }
    return nums;
}

// 桶排序
vector<int> bucketSort(vector<int>& nums, int bucket_size){
    int n = nums.size();
    if(n == 0) {return nums;}
    int min_ele = *min_element(nums.begin(), nums.end());
    int max_ele = *max_element(nums.begin(), nums.end());
    int bucket_num = (max_ele - min_ele) / bucket_size + 1;
    vector<vector<int>> bucket(bucket_num);
    for(int i = 0; i < n; i++){
        bucket[(nums[i] - min_ele) / bucket_size].emplace_back(nums[i]);
    }
    int i = 0;
    for(int j = 0; j < bucket_num; j++){
        int size = bucket[j].size();
        bucket[j] = quickSort(bucket[j]);// 每个桶里面借助其他排序算法进行排序，在这里使用了快排
        for(auto b : bucket[j]){
            nums[i] = b;
            i++;
        }
    }
    return nums;
}

int main(){
    int n = 5;
    vector<int> indices(n);
    iota(indices.begin(), indices.end(), 0);

    vector<int> nums = {2,8,4,1,3,5,6,7};
    ListNode* head = createList(nums);
    nums = bucketSort(nums, 5);
    return 0;
}