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

#### 二叉树最底层最左边的值

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
    vector<int> tmp;
    void levelOrder(TreeNode* root)
    {
        queue<TreeNode*> Q;
        Q.push(root);
        while(!Q.empty())
        {
            int size = Q.size();
            tmp.clear();
            for(int i=0;i<size;i++)
            {
                TreeNode* node = Q.front();
                if(node->left)
                    Q.push(node->left);
                if(node->right)
                    Q.push(node->right);
                tmp.push_back(node->val);
                Q.pop();
            }
        }
    }
    int findBottomLeftValue(TreeNode* root) {
         levelOrder(root);
         return tmp[0];
    }
};
```

#### 展平二叉搜索树

时间复杂度：n  空间复杂度 1

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
    TreeNode* cur;
    void dfs(TreeNode* root)
     {
         if(root == NULL)
            return;
        dfs(root->left);
        cur->right = root;
        cur = cur->right;
        cur->left = NULL;
        dfs(root->right);
     }
    TreeNode* increasingBST(TreeNode* root) {
        if(root == NULL)
            return NULL;
        TreeNode* head = new TreeNode(0,NULL,NULL);
        cur = head;
        dfs(root);
        return head->right;
    }
};
```

#### 二叉树中的中序后继

时间复杂度为树的层数

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* inorderSuccessor(TreeNode* root, TreeNode* p) {
        TreeNode* cur;TreeNode* result;
        result = NULL;
        cur = root;
        while(cur)
        {
            if(cur->val > p->val)
            {
                result = cur;
                cur = cur->left;
            }
            else
            {
                cur = cur->right;
            }
        }
        return result;
    }
};
```

#### 所有大于等于节点的值之和

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
    int sum;
    TreeNode* convertBST(TreeNode* root) {
         sum = 0;
         dfs(root);
         return root;
    }
private:
    void dfs(TreeNode* root)
    {
        if(root == NULL)
            return;
        dfs(root->right);
        sum += root->val;
        root->val = sum;
        dfs(root->left);

    }
};
```

#### 从根节点到叶节点的路径之和

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
    int sum = 0;
    void PreOrder(TreeNode* root,vector<int> num)
    {
        cout<<root->val<<endl;
        num.push_back(root->val);
        if(root->left == NULL && root->right ==NULL)
        {
            int size = num.size();
            int tmpNum=0;
            for(int i=size-1;i>=0;i--)
            {
                tmpNum+=pow(10,size-i-1)*num[i];
            }
            sum += tmpNum;
            //cout<<tmpNum<<endl;
            return;
        }
        if(root->left)
        {
            PreOrder(root->left,num);
        }
        if(root->right)
        {
            PreOrder(root->right,num);
        }
    }
    int sumNumbers(TreeNode* root) {
        if(root == NULL)
        {
            return 0;
        }
        vector<int> num;
        PreOrder(root,num);
        return sum;
    }
};
```

#### 二叉搜索树迭代器

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
class BSTIterator {
private:
    int index;
    vector<int> arr;
    void inorder(TreeNode* root,vector<int>& res)
    {
        if(root == NULL)
            return;
        if(root->left)
            inorder(root->left,res);
        res.push_back(root->val);
        if(root->right)
            inorder(root->right,res);
    }
    vector<int> inorderTraversal(TreeNode* root)
    {
        vector<int> res;
        inorder(root,res);
        return res;
    }
public:
    TreeNode * node;
    BSTIterator(TreeNode* root) {
        index = 0;
        arr = inorderTraversal(root);
    }
    
    int next() {
        return arr[index++];
    }
    
    bool hasNext() {
        if(index < arr.size())
            return true;
        else
            return false;
    }
};

/**
 * Your BSTIterator object will be instantiated and called as such:
 * BSTIterator* obj = new BSTIterator(root);
 * int param_1 = obj->next();
 * bool param_2 = obj->hasNext();
 */
```

#### 结点之和最大的路径

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
    int maxn = INT_MIN;
    int maxPathSum(TreeNode* root) {
        if(root == NULL)
            return 0;
        int n = dfs(root);
        return maxn;
    }
    int dfs(TreeNode* root)
    {
        if(root == NULL)
            return 0;
        int left = dfs(root->left);
        int right = dfs(root->right);
        left = max(0,left);
        right = max(0,right);
        maxn = max(maxn,left + right + root->val);//如果以该节点为参考点则把左右子树的最大路径都算上，因为该节点不论如何在本路径中，结点仅访问一次
        return max(left,right) + root->val;//如果从上面的祖先结点访问该根节点则返回通过该节点最大的路径（因为每个结点只能访问一次）
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

### 动态规划

#### 回文子字符串的个数（WA）

时间复杂度 n^2  空间复杂度 n

```c++
class Solution {
public:
    int countSubstrings(string s) {
        int sum = 0;
        int n = s.size();
        sum = n;
        int dp[n+1][n+1];
        for(int i=0;i<n;i++)
            dp[i][i] = 1;
        for(int k = 2;k <= n;k++)
        {
            for(int i=0;i+k-1<n;i++)
            {
                int j = i+k-1;
                if(s[i] == s[j])
                {
                    if(i < j-1 && dp[i+1][j-1] == 1)
                    {
                        dp[i][j] = 1;
                        sum++;
                    }
                    else if(i == j-1)
                    {
                        dp[i][j] = 1;
                        sum++;
                    }
                }
                else
                    continue;
            }
        }
        return sum;   
    }
};
```

#### 分割回文字符串

dfs回溯法

```c++
class Solution {
private:
    vector<vector<string>> res;
public:
    bool Palindrome(string& s,int left,int right)
    {
        for(;left<right;left++,right--)
        {
            if(s[left] != s[right])
                return false;
        }
        return true;
    }
    void helper(string& s,vector<string>& temp,int index)
    {
        if(index == s.size())
        {
            res.push_back(temp);
            return;
        }
        for(int end = index;end <s.size();end++)
        {
            //以Index为起点end为终点的字符串
            if(Palindrome(s,index,end))//如果是一个回文字符串
            {
                temp.push_back(s.substr(index,end-index+1));
                helper(s,temp,end+1);
                temp.pop_back();//恢复现场
            }

        }
    }
    vector<vector<string>> partition(string s) {
         vector<string> temp;c
         helper(s,temp,0);
         return res;
    }
};
```

#### 房屋偷盗

时间复杂度为 On^2 空间复杂度为 On

```c++
class Solution {
public:
    int rob(vector<int>& nums) {
        //设dpi为以第i间房屋结尾的最高金额
        int n = nums.size();
        vector<int> dp(n,0);
        dp[0] = nums[0];
        int ans = dp[0];
        if(n >= 2){
             dp[1] = nums[1];
              ans = max(dp[0],dp[1]);
        }
        for(int i=2;i<n;i++)
        {
            for(int j=0;j<i;j++)
            {
                if(j < i-1)
                    dp[i] = max(dp[i],dp[j] + nums[i]);
                ans = max(ans,dp[i]);
            } 
        }
        return ans;
    }
};c
```

时间复杂度On ，空间复杂度 O1  (标答)

```c++
class Solution {
public:
    int rob(vector<int>& nums) {
        int cur=0,pre=0;//一前一后
        for(int i=0;i<nums.size();i++)
        {
            int tmp = cur;
            cur = max(cur,pre + nums[i]);
            pre = tmp;
        }c
        return cur;
    }
};
```

####  环形房屋偷盗

```c++
class Solution {
public:
    int rob(vector<int>& nums) {
         if(nums.size() == 0) return 0;
         if(nums.size() == 1) return nums[0];
         int n = nums.size();
         int ans = 0;
         vector<int> tmp;
         for(int i=0;i<n-1;i++)
            tmp.push_back(nums[i]);
         ans = myrob(tmp);
         tmp.clear();
         for(int i=1;i<n;i++)
            tmp.push_back(nums[i]);
         ans = max(ans,myrob(tmp));
         return ans;
    }
private:
    int myrob(vector<int>& nums)
    {
        int cur=0,pre=0;
        for(int i=0;i<nums.size();i++)
        {
            int tmp = cur;
            cur = max(cur,pre + nums[i]);
            pre = tmp;
        }
        return cur;
    }
};
```

#### 粉刷房子

```c++
class Solution {
public:
    int minCost(vector<vector<int>>& costs) {
        //设dp[i][j]表示第i个房子被刷成i色需要的最少总花费
        int n = costs.size();
        int dp[n+1][4];
        dp[0][0] = costs[0][0];
        dp[0][1] = costs[0][1];
        dp[0][2] = costs[0][2];
        for( int i = 1 ; i < n ; i++ )
        {
            dp[i][0] = min(dp[i-1][1],dp[i-1][2]) + costs[i][0];
            dp[i][1] = min(dp[i-1][0],dp[i-1][2]) + costs[i][1];
            dp[i][2] = min(dp[i-1][0],dp[i-1][1]) + costs[i][2];
        }
        int ans = dp[n-1][0];
        ans = min(ans,dp[n-1][1]);
        ans = min(ans,dp[n-1][2]);
        return ans;
    }
};
```

#### 翻转字符

超时做法  时间复杂度On^2  空间复杂度 On

```c++
class Solution {
public:
    int minFlipsMonoIncr(string s) {
        //答案  = 总长度 - 最长上升子序列
        return s.size() - maxSubUpString(s);
    }
private:
    int maxSubUpString(string s)
    {
        int n = s.size();
        int maxlen[n+1];
        for(int i=0;i<n;i++)
            maxlen[i] = 1;
        int ans = 1;
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<i;j++)
            {
                if(s[i] >= s[j])
                    maxlen[i] = max(maxlen[i],maxlen[j] + 1);
                ans = max(ans,maxlen[i]);
            }
        }
        return ans;
    }
};
```

时间复杂度 On 空间复杂度On

```c++
class Solution {
public:
    int minFlipsMonoIncr(string s) {
        int n = s.size();
        int f0[n+1];
        int f1[n+1];
        if(s[0] == '0')
        {
            f0[0] = 0;
            f1[0] = 1;
        }
        else
        {
            f0[0] = 1;
            f1[0] = 0;
        }
        for(int i=1;i<n;i++)
        {
            f0[i] = f0[i-1] + (s[i] != '0');
            f1[i] = min(f0[i-1],f1[i-1]) + (s[i]!='1');
        }
        return min(f1[n-1],f0[n-1]);
    }
};
```

标答  时间复杂度：On  空间复杂度 O1

```c++
class Solution {
public:
    int minFlipsMonoIncr(string s) {
        int n = s.size();
        int f0pre,f0cur,f1pre,f1cur;
        if(s[0] == '0')
        {
            f0pre = 0;
            f1pre = 1;
        }
        else
        {
            f0pre = 1;
            f1pre = 0;
        }
        f1cur = f1pre;f0cur = f0pre;
        for(int i=1;i<n;i++)
        {
            f0cur = f0pre + (s[i] != '0');
            f1cur = min(f0pre,f1pre) + (s[i]!='1');
            f0pre = f0cur,f1pre = f1cur;
        }
        return min(f1cur,f0cur);
    }
};
```

#### 最长斐波那契数列

动态规划 + 哈希表  时间复杂度 ；n^2  空间复杂度 ： n^2

```c++
class Solution {
public:
    int lenLongestFibSubseq(vector<int>& arr) {
        vector<vector<int>> dp(arr.size(),vector<int>(arr.size()));
        unordered_map<int,int> mp;
        for(int i=0;i<arr.size();i++)
            mp[arr[i]] = i;
        int ans = 0;
        for(int i=0;i<arr.size();i++)
        {
            for(int j = 0;j<i;j++)
            {
                int temp = arr[i] - arr[j];
                if(mp.count(temp) == 1 && mp[temp] < j)
                {
                    dp[i][j] = dp[j][mp[temp]] + 1;
                }
                else
                {
                    dp[i][j] = 2;
                }
                ans = max(ans,dp[i][j]);
            }
        }
        if(ans <= 2)
            return 0;
        else
            return ans;
    }
};
```

动态规划 + 二分法 时间复杂度 n^2logn  空间复杂度 n^2  超时

```c++
class Solution {
public:
        int BinarySearch(vector<int> arr,int L,int R,int target)
        {
            int mid;
            while(L <= R)
            {
                mid = (L + R)>>1;
                if(arr[mid] == target)
                    return mid;
                if(arr[mid] < target)
                    L = mid+1;
                if(arr[mid] > target)
                    R = mid-1;
            }
            return -1;
        }
    int lenLongestFibSubseq(vector<int>& arr) {
        vector<vector<int>> dp(arr.size(),vector<int>(arr.size()));
        int ans = 0;
        for(int i=0;i<arr.size();i++)
        {
            for(int j = 0;j<i;j++)
            {
                int temp = arr[i] - arr[j]; 
                int pos = BinarySearch(arr,0,j-1,temp);
                if(pos !=-1)
                {
                    dp[i][j] = dp[j][pos] + 1;
                }
                else
                {
                    dp[i][j] = 2;
                }
                ans = max(ans,dp[i][j]);
            }
        }
        if(ans <= 2)
            return 0;
        else
            return ans;
    }
};
```

### 图：

#### 二分图（并查集 + DFS）

```c++
class Solution {
public:
        vector<int> tree;
        int find(int x)
        {
            if(x == tree[x])
                return x;
            tree[x] = find(tree[x]);
            return tree[x];
        }
        void Union(int x,int y)
        {
            int rootx=find(x);
            int rooty=find(y);
            if(rootx != rooty)
            {
                tree[x] = rooty;
            }
        }
    bool isBipartite(vector<vector<int>>& graph) {
        int size = graph.size();
        for(int i=0;i<size;i++)
            tree.push_back(i);
        for(int i=0;i<size;i++)
        {
            int roota = find(i);
            int edgeSize = graph[i].size();
            if(edgeSize > 0)
            {
                int rootb = find(graph[i][0]);
                if(roota == rootb)
                {
                    return false;
                }
                for(int j=1;j<edgeSize;j++)
                {
                    int rootc = find(graph[i][j]);
                    if(rootc == roota)
                        return false;
                     Union(rootc,rootb);
                }
            }
        }
        return true;
    }
};
```

#### 多余的边（并查集 ）

```c++
class Solution {
public:
    vector<int> parent;
    int res=-1;
    int cnt=-1;
    int find(int x)
    {
        if(x == parent[x])
            return x;
        parent[x] = find(parent[x]);
        return parent[x];
    }
    void Union(int x,int y)
    {
        int rootx=find(x);
        int rooty=find(y);
        if(rootx == rooty)
        {
            res = cnt;
        }
        parent[rootx] = rooty;
    }
    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        int size = edges.size();
        if(size == 0)
            return {};
        parent.push_back(0);
        for(int i=1;i<=size;i++)
            parent.push_back(i);
        for(int i=0;i<size;i++)
        {
            cnt++;
             Union(edges[i][0],edges[i][1]);
        }
        if(res == -1)
            return {};
        else
            return edges[res];
    }
};
```

#### 省份数量（纯并查集）

```c++
class Solution {
public:
    vector<int> parent;
    int find(int x)
    {
        if(parent[x] == x)
            return x;
        parent[x] = find(parent[x]);
        return parent[x];
    }
    void Union(int x,int y)
    {
        int rootx = find(x);
        int rooty = find(y);
        if(rootx != rooty)
            parent[rootx] = rooty;
    }
    int findCircleNum(vector<vector<int>>& isConnected) {
        int size = isConnected.size();
        if(size == 0)
            return 0;
        for(int i=0;i<size;i++)
            parent.push_back(i);
        for(int i=0;i<size;i++)
        {
            for(int j=0;j<isConnected[i].size();j++)
            {
                if(isConnected[i][j] == 1)
                    Union(i,j);
            }
        }
        //sort(parent.start(),parent.end());
        set<int> province;
        for(int i=0;i<size;i++)
        {
            province.insert(find(i));
        }
        // for(int i=0;i<size;i++)
        //     cout<<find(i)<<endl;
        return province.size();
    }
};
```

#### 重建序列（判断是否DAG+拓扑排序）

```c++
class Solution {
public:
    bool isOnlyOne = true;
    vector<int> ans;
    bool isDAG(vector<vector<int>>& edge,vector<int> in,vector<int> visted,int node_num)
    {
        int cnt=0;
        int start;
        for(int i=1;i<=node_num;i++)
        {
            if(in[i] == 0)
            {
                start = i;
                break;
            }
        }
        queue<int> q;
        q.push(start);
        while(!q.empty())
        {
            cnt++;
            int now=q.front();
            q.pop();
            for(int j=0;j<edge[now].size();j++)
            {
                int v=edge[now][j];
                in[v]--;
                if(in[v]==0)
                    q.push(v);
            }
        }
        if(cnt == node_num)//是否是DAG
            return true;
        else
            return false;

    }
    void build(vector<vector<int>>& edge,vector<vector<int>> seqs,int size,vector<int>& in,int node_num)
    { 
        set<pair<int,int>> used;
        for(int i=0;i<size;i++)
        {
            for(int j=0;j<seqs[i].size()-1;j++)
            {
                 int u=seqs[i][j],v=seqs[i][j+1];
                  pair<int,int> p;
                 p.first=u,p.second=v;
                 if(used.find(p) != used.end()) //判断是否重复边
                    continue;
                 edge[u].push_back(v);
                 used.insert(p);
                 in[v]++;
            }
        }
    }
    void topu(int start,vector<vector<int>> edge,vector<int> in,int size,int node_num,vector<int> visted) //拓扑排序
    {
        queue<int> Q;
        Q.push(start);
        while(!Q.empty())
        {
            if(Q.size()>1)
                isOnlyOne = false;
            int node = Q.front();
            Q.pop();
            ans.push_back(node);
            visted[node] = 1;
            for(int i=0;i<edge[node].size();i++)
            {
                in[edge[node][i]]--;
                if(in[edge[node][i]] == 0)
                    Q.push(edge[node][i]);
            }
        }
    }
    bool sequenceReconstruction(vector<int>& org, vector<vector<int>>& seqs) {
        set<int> s;
        int size = seqs.size();
        if(size == 0)
            return false;
        for(int i=0;i<size;i++)
            for(int j=0;j<seqs[i].size();j++)
                s.insert(seqs[i][j]);
        if(org.size() != s.size())
            return false;
        for(int i=0;i<org.size();i++)
        {
            if(s.find(org[i]) == s.end())
                return false;
        }
        int node_num = org.size();
        vector<vector<int>> edge(node_num + 1);
        vector<int> in(node_num + 1,0);//记录入度
        vector<int> visted(node_num + 1,0);
        build(edge,seqs,size,in,node_num);//建图
        int in_zero = 0;
        for(int i=1;i<=node_num;i++)
        {
            if(in[i] == 0)
                in_zero++;
        }
        if(in_zero != 1)
            return false;
        if(!isDAG(edge,in,visted,node_num))
            return false;
        for(int i=1;i<=node_num;i++)
        {
            if(in[i] == 0)
                topu(i,edge,in,size,node_num,visted);
        }
        if(!isOnlyOne)
            return false;
        for(int i=0;i<org.size();i++)
        {
            if(ans[i] != org[i])
                return false;
        }
        return true;
    }
};
```

#### 课程顺序（拓扑排序）未AC

```c++
class Solution {
public:
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        vector<int> ans;
        if(prerequisites.size() == 0)
            return ans;
        vector<int> in(numCourses+1,0);
        vector<vector<int>> G(numCourses+1,vector<int>);
        int size = prerequisites.size();
        for(int i=0;i<size;i++)
        {
            in[prerequisites[i][0]]++;
            G[prerequisites[i][1]].push_back(prerequisites[i][0]);
        }
        for(int i=0;i<numCourses;i++)
        {
            if(in[i] == 0)
            {
                queue<int> Q;
                Q.push(i);
                while(!Q.empty())
                {
                    int now = Q.front();
                    Q.pop();
                    ans.push_back(now);
                    for(int j=0;j<G[now].size();j++)
                    {
                        in[G[now][j]]--;
                        if(in[G[now][j]] == 0)
                            Q.push(G[now][j]);
                    }
                }
            }
        }
        for(int i=0;i<numCourses;i++)
        {
            if(in[i]!=0)
                return {};
        }
        return ans;
    }
};
```

#### 最长递增路径

（dfs求最长路径）做法

```c++
class Solution {
public:
    static constexpr int dirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    int rows, columns;

    int longestIncreasingPath(vector< vector<int> > &matrix) {
        if (matrix.size() == 0 || matrix[0].size() == 0) {
            return 0;
        }
        rows = matrix.size();
        columns = matrix[0].size();
        auto memo = vector< vector<int> > (rows, vector <int> (columns));
        int ans = 0;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                ans = max(ans, dfs(matrix, i, j, memo));
            }
        }
        return ans;
    }

    int dfs(vector< vector<int> > &matrix, int row, int column, vector< vector<int> > &memo) {
        if (memo[row][column] != 0) {
            return memo[row][column];
        }
        ++memo[row][column];
        for (int i = 0; i < 4; ++i) {
            int newRow = row + dirs[i][0], newColumn = column + dirs[i][1];
            if (newRow >= 0 && newRow < rows && newColumn >= 0 && newColumn < columns && matrix[newRow][newColumn] > matrix[row][column]) {
                memo[row][column] = max(memo[row][column], dfs(matrix, newRow, newColumn, memo) + 1);
            }
        }
        return memo[row][column];
    }
};

```

拓扑排序做法（WA）

```c++
class Solution {
public:
    struct node
    {
        int x;int y;
        int len;
    };
    int row,col;
    int dir[4][2]={{1,0},{-1,0},{0,1},{0,-1}};
    int tuopu(vector<vector<int>>& matrix, vector<vector<int>> in,int x,int y)
    {
        int ans = 0;
        queue<node> Q;
        node tmp;
        tmp.x = x,tmp.y = y;
        tmp.len = 1;
        Q.push(tmp);
        while(!Q.empty())
        {
            node now = Q.front();
            Q.pop();
            ans = max(ans,now.len);
            for(int k=0;k<4;k++)
            {
                int ux = now.x+dir[k][0];
                int uy = now.y+dir[k][1];
                if(ux>=0&&uy>=0&&ux<row&&uy<col&&matrix[ux][uy]>matrix[now.x][now.y]){
                    in[ux][uy]--;
                    if(in[ux][uy]==0)
                    {
                        tmp.x=ux;
                        tmp.y=uy;
                        tmp.len=now.len+1;
                        Q.push(tmp);
                    }
                }
            }
        }
        return ans;
    }
    int longestIncreasingPath(vector<vector<int>>& matrix) {
        int maxLen = 0;
        row = matrix.size();
        if(row == 0)
            return 0;
        col = matrix[0].size();
        if(col ==0 )
            return 0;
        vector<vector<int>> in(row+1,vector<int>(col+1,0));
        for(int i=0;i<row;i++)
        {
            for(int j=0;j<col;j++)
            {
                for(int k=0;k<4;k++)
                {
                    int ux=i+dir[k][0];
                    int uy=j+dir[k][1];
                    if(ux>=0&&uy>=0&&ux<row&&uy<col&&matrix[ux][uy]>matrix[i][j])
                        in[ux][uy]++;
                }
            }
        }
        for(int i=0;i<row;i++)
        {
            for(int j=0;j<col;j++)
            {
                if(in[i][j] == 0){
                    maxLen = max(tuopu(matrix,in,i,j),maxLen);
                   }
            }
        }
        return maxLen;
    }
};
```

