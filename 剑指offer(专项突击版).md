## 剑指offer专项突击版刷题记录

[TOC]



### 链表，队列，栈

#### 链表排序（以后重做尝试归并排序）

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* sortList(ListNode* head) {
        if(head == NULL) return NULL;
        vector<int> arr;
        ListNode * node = head;
        while(node){
            arr.push_back(node->val);
            node = node->next;
        }
        int size = arr.size();
        sort(arr.begin(),arr.end());
        node = head;
        for(int i=0;i<size;i++){
            node->val = arr[i];
            node = node->next;
        }
        return head;
    }
};
```

#### 删除链表的倒数第n个结点

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        if(head == NULL) return NULL;
        int size = 0;
        ListNode * node;
        while(node){
            size++;
            node = node->next;
        }
        int cnt = 0;
        node = new ListNode(-1);//哨兵结点
        node->next = head;
        ListNode * fast = head;
        ListNode * low = node;
        while(fast){
            cnt++;
            if(cnt == size-n+1){
                low->next = fast->next;
            }
            fast = fast->next;
            low = low->next;
        }
        return node->next;
    }
};
```

#### 链表中环的入口结点 

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
       unordered_set<ListNode *> vis;
       ListNode * node = head;
       while(node){
           if(vis.count(node)){
               return node;
           }
           vis.insert(node);
           node = node->next;
       } 
       return NULL;
    }
};
```

快慢指针实现  空间O(1)

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode * MeetingNode(ListNode * head){
        if(head == NULL)
            return NULL;
        ListNode * pSlow = head->next;
        if(pSlow == NULL)
            return NULL;
        ListNode * pFast = pSlow->next;
        while(pFast != NULL && pSlow!= NULL){
            if(pFast == pSlow)
                return pFast;
            pSlow = pSlow->next;
             pFast = pFast->next;
            if(pFast!=NULL)
                pFast = pFast->next;
            //pFast 的移动速度是pSlow的两倍
        }
        return NULL;
    }
    ListNode *detectCycle(ListNode *head) {
       ListNode * meetingNode = MeetingNode(head);
       if(meetingNode == NULL)
            return NULL;
        int nodesLoop = 1;//得到环中结点的数目
        ListNode * pNode1 = meetingNode;
        while(pNode1->next!=meetingNode){
            pNode1 = pNode1->next;
            nodesLoop++;
        }
        pNode1 = head;
        for(int i=0;i<nodesLoop;i++)
            pNode1 = pNode1->next;
        ListNode * pNode2 = head;
        while(pNode1 != pNode2){
            pNode1 = pNode1->next;
            pNode2 = pNode2->next;
        }
        return pNode1;
    }
};
```



####  排序的循环链表

```c++
/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* next;

    Node() {}

    Node(int _val) {
        val = _val;
        next = NULL;
    }

    Node(int _val, Node* _next) {
        val = _val;
        next = _next;
    }
};
*/

class Solution {
public:
    Node* insert(Node* head, int insertVal) {
        Node * slow = head;
        if(head == NULL){
            head = new Node(insertVal);
            head->next = head;
            return head; 
        }
        int flag = 0;
        Node * fast = head->next;
        while(1){
            if(slow->val <= insertVal && fast->val >insertVal){
                Node * temp = new Node(insertVal);
                slow->next = temp;
                temp->next = fast;
                return head;
            }
            if(fast->val < slow->val && insertVal >= slow->val){
                Node * temp = new Node(insertVal);
                slow->next = temp;
                temp->next = fast;
                return head;
            }
            if(fast->val < slow->val && insertVal <=fast->val){
                Node * temp = new Node(insertVal);
                slow->next = temp;
                temp->next = fast;
                return head;
            }
            if(slow == head && flag){
                Node * temp = new Node(insertVal);
                slow->next = temp;
                temp->next = fast;
                return head;
            }
            slow = slow->next;
            fast = fast->next;
            flag = 1;
        }
        return head;
    }
};
```

#### 最近最少使用缓存(RE)

```c++
class LRUCache {
public:
    class Node{
        public:
        int key  , value ;
        Node * left; 
        Node * right;
        Node(){
            left = NULL;
            right = NULL;
            key = -1, value = -1;
        }
    };
    int n;//表示哈希表长度
    Node * head, * tail;
    unordered_map<int,Node *> hash;
    LRUCache(int capacity) {
        n = capacity;
        head = new Node ,tail = new Node;
        head->right = tail;
        tail->left = head;
    }

    void remove(Node * node){
        node->left->right = node->right;
        node->right->left = node->left;
    }

    void insertHead(Node * node){
        node->right = head->right;
        node->left = head;
        head->right->left = node;
        head->right = node;
    }
    
    int get(int key) {
        if(!hash.count(key)) return -1;
        Node * p = hash[key];
        remove(p);
        insertHead(p);
        return p->value;
    }
       
    void put(int key, int value) {
        if(hash.count(key)){
            Node * p = hash[key];
            p->value = value;
            remove(p);
            insertHead(p);
        }
        else{
            if(hash.size() == n){
                //链表已满删除尾元素
                Node * p = tail->left;
                remove(p);
                insertHead(p);
                hash.erase(p->key);
                delete p;
            }
            Node * p = new Node;
            p->key = key;
            p->value = value;
            hash[key] = p;
            insertHead(p);
        }
    }
};

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache* obj = new LRUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */
```

### 树

#### 往完全二叉树添加节点（WA）

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */

class CBTInserter {
private:
    TreeNode * root;
    queue<TreeNode*> Q;
public:
    CBTInserter(TreeNode* root) {
        this->root = root;
        Q.push(this->root);
        while(Q.front()->left && Q.front()->right)
        {
            TreeNode * node = Q.front();
            Q.pop();
            Q.push(node->left);
            Q.push(node->right);
        }
    }
    
    int insert(int v) {
            TreeNode * node = new TreeNode(v);
            TreeNode * parent = Q.front();
            if(node->left == NULL)
            {
                parent->left = node;
            }
            else
            {
                parent->right = node;
                Q.pop();
                Q.push(node->left);
                Q.push(node->right);
            }
            return parent->val;
    }
    
    TreeNode* get_root() {
        return this->root;
    }
};

```

#### 二叉树的右侧视图

层序遍历

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<int> res;
    void levelOrder(TreeNode* root)
    {
        queue<TreeNode*> Q;
        Q.push(root);
        while(!Q.empty())
        {
            int size = Q.size();
            for(int i=0;i<size;i++)
            {
                TreeNode* tmp = Q.front();
                if(tmp->left) 
                    Q.push(tmp->left);
                if(tmp->right)
                    Q.push(tmp->right);
                if(i == size-1)
                    res.push_back(tmp->val);
                Q.pop();
            }
        }
    }
    vector<int> rightSideView(TreeNode* root) {
        if(root == NULL)
            return res;
        levelOrder(root);
        return res;c
    }
};
```

#### 二叉树剪枝

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    bool check(TreeNode* root)
    {
        bool flag = root->val;
        if(root->left)
            flag = flag||check(root->left);
        if(root->right)
            flag = flag||check(root->right);
        return flag;
    }
    TreeNode* pruneTree(TreeNode* root) {
        if(root == NULL)
            return NULL;
        if(!check(root))
            return NULL;
        if(root->left)
        {
            root->left = pruneTree(root->left);
        }
        if(root->right)
        {
            root->right = pruneTree(root->right);
        }
        return root;
    }
};
```

#### 向下路径节点之和

深搜穷举  时间复杂度：n^2   空间复杂度  n

```c++
class Solution {
public:
    int rootSum(TreeNode* root, int targetSum) {
        if (!root) {
            return 0;
        }

        int ret = 0;
        if (root->val == targetSum) {
            ret++;
        } 

        ret += rootSum(root->left, targetSum - root->val);
        ret += rootSum(root->right, targetSum - root->val);
        return ret;
    }

    int pathSum(TreeNode* root, int targetSum) {
        if (!root) {
            return 0;
        }
        
        int ret = rootSum(root, targetSum);
        ret += pathSum(root->left, targetSum);
        ret += pathSum(root->right, targetSum);
        return ret;
    }
};

 
```



###   二分法

#### 排序数组中两个数字的和(WA)

```c++
class Solution {
public:
    bool flag = true;
    int find(vector<int> numbers,int num)
    {
        int mid,L,R;
        L = 0,R = numbers.size()-1;
        while(L <= R)
        {
            mid = (L+R)>>1;
            if(num == numbers[mid])
                return mid;
            if(num > numbers[mid])
                L = mid+1;
            if(num < numbers[mid])
                R = mid-1;
        }
        if(L > R)
        //cout<<L<<" "<<R<<" "<<mid<<endl;
            flag = false;
           // cout<<mid<<endl;
        return mid;
    }
    vector<int> twoSum(vector<int>& numbers, int target) {
        int index1,index2;
        vector<int> ret;
        if(numbers.size() == 0)
            return ret;
        ret.push_back(0),ret.push_back(0);
        int n=numbers.size();
        for(int i=0;i<n;i++)
        {
            int num = target - numbers[i];
            if(num == numbers[i])
                continue;
            index2 = find(numbers,num);
            if(flag == true)
            {
                index1 = i;
                break;
            }
        }
        ret[0] = index1;
        ret[1] = index2;
       // cout<<numbers[index1]<<" "<<numbers[index2]<<endl;
        return ret;
    }
};
```

#### 山峰数组的顶部

```c++
class Solution {
public:
    int peakIndexInMountainArray(vector<int>& arr) {
        int mid,L,R;
        L = 0,R=arr.size()-1;
        int ans = 0;
        while(L <= R)
        {
            mid = (L+R)>>1;
            if(arr[mid] > arr[mid+1])
            {
                ans = mid;
                R = mid-1;
            }  
            else
            {
                L = mid+1;
            }
        }
        return ans;
    }
};
```

#### 查找插入位置

```c++
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int n = nums.size();
        int L,R,mid,ans=0;
        L = 0,R = n-1;
        while(L <= R)
        {
            mid = (L+R)>>1;
            if(nums[mid] == target)
            {
                ans = mid;
                break;
            }
            if(nums[mid] > target)
            {
                R = mid-1;
            }
            if(nums[mid] < target)
            {
                L = mid+1;
                ans = L;
            }
        }
        return ans;
    }
};
```

#### 狒狒吃香蕉

```c++
class Solution {
public:
    int timeConsume(vector<int>& piles,int speed)
    {
        int sumTime = 0;
        for(int i=piles.size()-1;i>=0;i--)
        {
            sumTime += piles[i]/speed;
            sumTime+=((piles[i]%speed)>0);
        }
        return sumTime;
    }
    int minEatingSpeed(vector<int>& piles, int h) {
        //最大速度是max(piles[i]) 最小速度是min(piles[i])
        //二分速度
        int n = piles.size();
         int maxSpeed = *max_element(piles.begin(),piles.end());
         int minSpeed=1;
        int L=minSpeed,R=maxSpeed,mid;
        while(L <= R)
        {
            mid = (L+R)>>1;
            int time = timeConsume(piles,mid);
            if(time <= h)//时间短了，速度快了，速度应该慢一点
            {
                if(mid == 1 || timeConsume(piles,mid-1)>h)
                {
                    return mid;
                }
                R = mid-1;
            }
            if(time > h)
            {
                L = mid + 1;
            }
        }
        return -1;
    }
};
```

#### 和大于等于target的最短子数组的长度 

法一：二分法+双指针  时间复杂度 nlogn

```c++
class Solution {
public:
    int maxSubArraySum(vector<int> nums,int len)
    {
        if(len == 0)
            return 0;
        int ans = 0 , L = 0,sum = 0;
        queue<int> Q;//此处用队列是保证 n 时间复杂度
        for(int i=0;i<nums.size();i++)
        {
            if(L >= len)
            {
                int tmp = Q.front();
                sum-=tmp;
                Q.pop();
                L--;
            }
            sum += nums[i];
            ans = max(ans,sum);
            Q.push(nums[i]);
            L++;
        }
        //cout<<endl;
        return ans;
    }
    int minSubArrayLen(int target, vector<int>& nums) {
        if(nums.size() == 0)
            return 0;
         int L = 0,R = nums.size(),mid;
         int maxSum;
         int minLen = -1;
         while(L <= R)
         {
             mid = (L+R)>>1;
             int maxSum = maxSubArraySum(nums,mid);
             //cout<<mid<<" "<<L<<" "<<R<<" "<<maxSum<<endl;
            if(maxSum < target)
            {
                L = mid+1;
            }
            if(maxSum >= target)
            {
                minLen = mid;
                R = mid-1;
            }
         }
         if(minLen == -1)
            return 0;
        return minLen;
    }
};
```

法二：二分法 + 前缀和   时间复杂度：nlogn

```c++
class Solution {
public:
    int minSubArrayLen(int s, vector<int>& nums) {
        int n = nums.size();
        if (n == 0) {
            return 0;
        }
        int ans = INT_MAX;
        vector<int> sums(n + 1, 0); 
        // 为了方便计算，令 size = n + 1 
        // sums[0] = 0 意味着前 0 个元素的前缀和为 0
        // sums[1] = A[0] 前 1 个元素的前缀和为 A[0]
        // 以此类推
        for (int i = 1; i <= n; i++) {
            sums[i] = sums[i - 1] + nums[i - 1];
        }
        for (int i = 1; i <= n; i++) {
            int target = s + sums[i - 1];
            auto bound = lower_bound(sums.begin(), sums.end(), target);
            if (bound != sums.end()) {
                ans = min(ans, static_cast<int>((bound - sums.begin()) - (i - 1)));
            }
        }
        return ans == INT_MAX ? 0 : ans;
    }
};
```

法三：滑动窗口  时间复杂度：n

```c++
class Solution {
public:
    int minSubArrayLen(int s, vector<int>& nums) {
        int n = nums.size();
        if (n == 0) {
            return 0;
        }
        int ans = INT_MAX;
        int start = 0, end = 0;
        int sum = 0;
        while (end < n) {
            sum += nums[end];
            while (sum >= s) {
                ans = min(ans, end - start + 1);
                sum -= nums[start];
                start++;
            }
            end++;
        }
        return ans == INT_MAX ? 0 : ans;
    }
};

```

#### 排序数组中只出现一次的数字

时间复杂度 logn  空间复杂度 O(1)

```c++
class Solution {
public:
    int singleNonDuplicate(vector<int>& nums) {
        int n = nums.size();
        int L = 0,R = n-1,mid;
        //当区间只剩下一个数即L==R时即为所求
        while(L < R)
        {
            mid = L+((R-L)>>1);
            if(mid%2 == 1)
                mid--;//支队偶数进行检索
            if(nums[mid] == nums[mid+1])
            {
                L = mid+2;
            }
            else
            {
                R = mid;
            }
        }
        return nums[L];
    }
};
```

