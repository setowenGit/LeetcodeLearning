#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <queue>

using namespace std;

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
        if(j == needle.size()) { return i - needle.size() + 1;}
    }
    return -1;
}

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



vector<int> topStudents(vector<string>& positive_feedback, vector<string>& negative_feedback, vector<string>& report, vector<int>& student_id, int k) {
    priority_queue<pair<int, int>, vector<pair<int, int>>> que;
    for(int i = 0; i < student_id.size(); i++){
        int score = 0;
        for(int p = 0; p < positive_feedback.size(); p++){
            vector<int> pos = KMP_process_mul(report[i], positive_feedback[p]);
            score += 3 * pos.size();
        }
        for(int n = 0; n < negative_feedback.size(); n++){
            vector<int> pos = KMP_process_mul(report[i], negative_feedback[n]);
            score -= 1 * pos.size();
        }
        pair<int, int> stu(score, -student_id[i]);
        que.emplace(stu);
    }
    vector<int> ans;
    for(int num = 0; num < k; num++){
        ans.emplace_back(-que.top().second);
        que.pop();
    }
    return ans;
}

    int check(string& s, int left, int right){
        int len = right - left + 1;
        if(len > 1 && s[left] == '0') {return -1;}
        int num = 0;
        for(int i = left; i <= right; i++){
            num = num * 10 + (s[i] - '0');
        }
        if(num > 255) {return -1;}
        return num;
    }
    

void dfs(string& s, int index, int len, vector<string>& path, vector<string>& ans, int splitimes){
        if(index == len){
            if(splitimes == 4){
                string tmp = "";
                for(int j = 0; j < path.size(); j++){
                    tmp += path[j];
                    if(j != path.size()-1) {tmp += ".";}
                }
                ans.emplace_back(tmp);
            }
            return;
        }
         if (len - index < (4 - splitimes) || len - index > 3 * (4 - splitimes)) {
            return;
        }
        for(int i = 0; i < 3; i++){
            if(index + i >= len) {break;}
            int ipSegment = check(s, index, index + i);
            if (ipSegment != -1) {
                // 在判断是 ip 段的情况下，才去做截取
                path.emplace_back(s.substr(index, i + 1));
                dfs(s, len, index + i + 1, path, ans, splitimes + 1);
                path.pop_back();
            }
        }
    }



vector<string> restoreIpAddresses(string s) {
        int len = s.length();
        vector<string> ans;
        if(len < 4 || len > 12) {return ans;} //超出IP长度直接返回
        vector<string> path;
        dfs(s, 0, len, path, ans, 0);
        
        return ans;
    }

int main(){
    // restoreIpAddresses("25525511135");
    // return 0;
    string s;
    cin >> s;
    cout << s << endl;
    return 0;
}