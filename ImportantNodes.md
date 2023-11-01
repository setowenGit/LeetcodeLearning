# 重要编程知识汇总

## 1. 二分查找

* 找到**大于等于**目标值的**下界**的位置：lower_bound
* 找到**大于**目标值的**下界**的位置：upper_bound
* 找到**小于**目标值的**上界**的位置：lower_bound - 1
* 找到**小于等于**目标值的**上界**的位置：upper_bound - 1

![](./fig/9.jpg)

### lower_bound的三种写法

```c++
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

// 搜索第一个大于等于target的数的索引（左闭右开写法）【常用】
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
```

### upper_bound的三种写法（在lower_bound基础上加多了一个等于号判断即可）
```c++
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

// 搜索第一个大于target的数的索引（左闭右开写法）【常用】
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
```

> 视频 https://www.bilibili.com/video/BV1AP41137w7/?vd_source=1dc1f5616fa7bc05e0def8e62f42c924

## 2. 重点字符串匹配算法——[KMP算法](https://leetcode.cn/problems/implement-strstr/solution/duo-tu-yu-jing-xiang-jie-kmp-suan-fa-by-w3c9c/)

寻找字符串中的匹配字符串最先出现的位置
```c++
// KMP字符串匹配算法——haystack为原字符串，needle为匹配串
int KMP_process(string haystack, string needle){
    if(needle.size() == 0) {return 0;}
    vector<int> next(needle.size());//前缀表，存放当前长度下的最长相同前后缀的长度
    //填写前缀表
    for(int left = 0, right = 1; right < needle.size(); right++){
        //当left大于0且与right的字符不相等时，则回退到next数组所指示的位置
        while(left > 0 && needle[right] != needle[left]){
            left = next[left - 1];
        }
        //当right和left的元素相同时，left加一
        if(needle[right] == needle[left]){  
            left++;
        }
        //将left填入到这时right对应的next位置
        next[right] = left;
    }
    //找寻原字符串中匹配串的起始位置
    //j相当于上面的left，并在needle中访问
    //i相当于上面的right，haystack相当于上面的needle，并在haystack中访问，i从0开始而不是从1开始
    for(int j = 0, i = 0; i < haystack.size(); i++){
        while(j > 0 && haystack[i] != needle[j]){
            j = next[j-1];
        }
        if(haystack[i] == needle[j]){
            j++;
        }
        //若j已经在匹配串中走完，则表示匹配完成
        if(j == needle.size()) {return i - needle.size() + 1;}
    }
    return -1;
}
```

寻找字符串中的匹配字符串所有出现的位置
```c++
vector<int> KMP_process_mul(string haystack, string needle){
    vector<int> pos;// 改动的地方
    if(needle.size() == 0) {return pos;}
    vector<int> next(needle.size());
    for(int left = 0, right = 1; right < needle.size(); right++){
        while(left > 0 && needle[right] != needle[left]){
            left = next[left - 1];
        }
        if(needle[right] == needle[left]){
            left++;
        }
        next[right] = left;
    }
    for(int j = 0, i = 0; i < haystack.size(); i++){
        while(j > 0 && haystack[i] != needle[j]){
            j = next[j-1];
        }
        if(haystack[i] == needle[j]){
            j++;
        }
        if(j == needle.size()) {
            pos.emplace_back(i - j + 1);// 改动的地方
            j = next[j - 1];// 改动的地方
        }
    }
    return pos;
}
```

## 3. 内置sort函数的自定义排序

默认的sort是升序排序，要想把他改为降序排序，则如下

```c++
static bool cmp(int& a, int& b){
    return a > b;
}

int main(){
    ...
    sort(nums.begin(), nums.end(), cmp);
    ...
}
```

* 记住自定义的cmp要加上static
* cmp函数的返回值是一个bool值，当返回值为true时不改变元素顺序，反之则需要调换元素。

## 4. unordered_map用法

* 将unordered_map元素放入vector中，其中每一个元素是一个pair，所以vector的类型是```pair<int, int>```
* 当使用范围for或是普通for遍历时，不能保证遍历获得的元素的顺序就是元素加入到unordered_map中的顺序

```c++
unordered_map<int, int> nums_hash;
for(auto& n : nums){
    nums_hash[n]++;
}
vector<pair<int, int>> numsp;
for(auto& n : nums_hash){
    numsp.emplace_back(n);
}
```

| 成员方法      | 功能                                                                                                                                                                                                        |
|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| begin()       | 返回指向容器中第一个键值对的正向迭代器。                                                                                                                                                                    |
| end()         | 返回指向容器中最后一个键值对之后位置的正向迭代器。                                                                                                                                                          |
| cbegin()      | 和 begin() 功能相同，只不过在其基础上增加了 const 属性，即该方法返回的迭代器不能用于修改容器内存储的键值对。                                                                                                |
| cend()        | 和 end() 功能相同，只不过在其基础上，增加了 const 属性，即该方法返回的迭代器不能用于修改容器内存储的键值对。                                                                                                |
| empty()       | 若容器为空，则返回 true；否则 false。                                                                                                                                                                       |
| size()        | 返回当前容器中存有键值对的个数。                                                                                                                                                                            |
| max_size()    | 返回容器所能容纳键值对的最大个数，不同的操作系统，其返回值亦不相同。                                                                                                                                        |
| operator[key] | 该模板类中重载了 [] 运算符，其功能是可以向访问数组中元素那样，只要给定某个键值对的键 key，就可以获取该键对应的值。注意，如果当前容器中没有以 key 为键的键值对，则其会使用该键向当前容器中插入一个新键值对。 |
| at(key)       | 返回容器中存储的键 key 对应的值，如果 key 不存在，则会抛出 out_of_range 异常。                                                                                                                              |
| find(key)     | 查找以 key 为键的键值对，如果找到，则返回一个指向该键值对的正向迭代器；反之，则返回一个指向容器中最后一个键值对之后位置的迭代器（如果 end() 方法返回的迭代器）。                                            |
| count(key)    | 在容器中查找以 key 键的键值对的个数。                                                                                                                                                                       |
| emplace()     | 向容器中添加新键值对，效率比 insert() 方法高。                                                                                                                                                              |
| insert()      | 向容器中添加新键值对。                                                                                                                                                                                      |
| erase()       | 删除指定键值对。                                                                                                                                                                                            |
| clear()       | 清空容器，即删除容器中存储的所有键值对。                                                                                                                                                                    |
| swap()        | 交换 2 个 unordered_map 容器存储的键值对，前提是必须保证这 2 个容器的类型完全相等。                                                                                                                         |

## 5. priority_queue用法

* cmp比较参数是```less<T>```表示大根堆，```greater<T>```表示小根堆
* 自定义优先队列排序的格式如下，下面是实现了针对pair下的second的小根堆，记住cmp函数要加static，且使用decltype来在初始化时指定cmp函数

```c++
static bool cmp(pair<int, int>& a, pair<int, int>& b){
    return a.second > b.second;
}

int main(){
    ...
    priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(&cmp)> que(cmp);
    ...
}
```

| 成员方法 | 功能                       |
|----------|----------------------------|
| top      | 访问队头元素               |
| empty    | 队列是否为空               |
| size     | 返回队列内元素个数         |
| push     | 插入元素到队尾 (并排序)    |
| emplace  | 原地构造一个元素并插入队列 |
| pop      | 弹出队头元素               |
| swap     | 交换内容                   |