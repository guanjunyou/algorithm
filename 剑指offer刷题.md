

## 剑指offer刷题记录

[TOC]



### 栈与队列

#### 用两个栈实现队列

```c++
class CQueue {
	stack<int> stack1, stack2;
public:
	//CQueue() {
		//while (!stack1.empty()) {
		//	stack1.pop();
//}
	//	while (!stack2.empty()) {
	//		stack2.pop();      
	//	}
	//}

	void appendTail(int value) {
		stack1.push(value);
	}

	int deleteHead() {
		// 如果第二个栈为空
		if (stack2.empty()) {
			while (!stack1.empty()) {
				stack2.push(stack1.top());
				stack1.pop();
			}
		}
		if (stack2.empty()) {
			return -1;
		}
		else {
			int deleteItem = stack2.top();
			stack2.pop();
			return deleteItem;
		}
	}
};
```

#### 包含min函数的栈（编译失败）

```c++
class MinStack {
public:
    /** initialize your data structure here. */
    stack<long long> Stack;
    long long min_number = LONG_LONG_MAX;
    MinStack() {
    }
    
    void push(long long x) {
         if(min_number > x) min_number = x;
         Stack.push(x);
         Stack.push(min_number);
    }
    
    void pop() {
        Stack.pop();
        Stack.pop();
    }
    
    long long top() {
         long long minn = Stack.top();
         Stack.pop();
         long long res = Stack.top();
         Stack.push(minn);
         return res;
    }
    
    long long min() {
        return Stack.top();
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(x);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->min();
 */
```

#### 队列的最大值



```c++
class MaxQueue {
public:
    deque<int> d;
    queue<int> q;
    MaxQueue() {

    }
    
    int max_value() {
        if(d.empty()) return -1;
        return d.front();
    }
    
    void push_back(int value) {
        while(!d.empty() && d.back() < value)
            d.pop_back();
        d.push_back(value);
        q.push(value);
    }
    
    int pop_front() {
        if(q.empty()) return -1;
        int ans = q.front();
        q. pop();
        if(d.front() == ans) d.pop_front();
        return ans;
    }
};

/**
 * Your MaxQueue object will be instantiated and called as such:
 * MaxQueue* obj = new MaxQueue();
 * int param_1 = obj->max_value();
 * obj->push_back(value);
 * int param_3 = obj->pop_front();
 */
```

#### 丑数（小顶堆+set判重）

```c++
class Solution {
public:
    int nthUglyNumber(int n) {
        priority_queue<long,vector<long>,greater<long>> q;
        unordered_set<long> seen;
        int dir[3] = {2,3,5};
        q.push(1L);
        seen.insert(1L);
        long ugly;
        for(int i=1;i<=n;i++){
            ugly = q.top();
            q.pop();
            for(int k=0;k<3;k++){
                long temp = ugly*dir[k];
                if(!seen.count(temp)){
                    q.push(temp);
                    seen.insert(temp);
                }
            }
        }
        return ugly;
    }
};
```



### 链表

#### 从头到尾打印链表

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
    vector<int> reversePrint(ListNode* head) {
        vector<int> result;
        stack<int> arr;
        ListNode * p = head;
        while(p!= NULL){
            arr.push(p->val);
            p=p->next;
        }
        int len = arr.size();
        for(int i=0;i<len;i++)
        {
            result.push_back(arr.top());
            arr.pop();
        }
        return result;
    }
};
```

#### 反转链表

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
    ListNode* reverseList(ListNode* head) {
        ListNode *pre = NULL;
        ListNode *cur = head;
        while(cur) {
            ListNode *tmp = cur -> next;
            cur -> next = pre;
            pre = cur;
            cur = tmp;
        }
        head = pre;
        return head;
    }
};
```

#### 删除链表的节点

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
    ListNode* deleteNode(ListNode* head, int val) {
        if(head == NULL) return NULL;
        ListNode * node = head;
        if(head->val == val){
            ListNode * node = head->next;
            return node;
        }
        while(node->next){
            if(node->next->val == val){
                node->next = node->next->next;
                return head;
            }
            node = node->next;
        }
        return head;
    }
};
```

#### 链表中倒数第k个节点

```c++
class Solution {
public:
    ListNode* getKthFromEnd(ListNode* head, int k) {
        ListNode* fast = head;
        ListNode* slow = head;

        while (fast && k > 0) {
            fast = fast->next;
            k--;
        }
        while (fast) {
            fast = fast->next;
            slow = slow->next;
        }

        return slow;
    }
};

```

更鲁棒的代码：

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
    ListNode* getKthFromEnd(ListNode* pListHead, int k) {
          if(pListHead == NULL||k==0)
            return NULL;
          ListNode * pAhead = pListHead;
          ListNode * pBehind = NULL;

          for(int i=0;i<k-1;i++){
              if(pAhead->next!=NULL)
                pAhead = pAhead->next;
            else
            {
                return NULL;
            }
          }
          pBehind = pListHead;
          while(pAhead->next!=NULL){
              pAhead = pAhead->next;
              pBehind = pBehind->next;
          }
          return pBehind;
    }
};
```



#### 合并两个排序的链表

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
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if(l1 == NULL) return l2;
        if(l2 == NULL) return l1;
        ListNode * minNode ;ListNode * maxNode;
        if(l1->val <= l2->val){
            minNode = l1 , maxNode = l2;
        }else{
            minNode = l2 , maxNode = l1;
        }
        ListNode * p1;ListNode * p2;
        p1 = minNode,p2 = maxNode;
        while(p1 && p2){
            if(p1->next == NULL && p2->val >= p1->val){
                p1->next = p2;
                break;
            }
            
            if(p2->val >= p1->val && p2->val <p1->next->val){
                ListNode * temp = p2;
                p2 = p2->next;
                temp->next = p1->next;
                p1->next = temp;
            }
            p1 = p1->next;
        }
        return minNode;
    }
};
```

#### 复杂链表的复制(超时代码)

```c++
/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* next;
    Node* random;
    
    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};
*/
class Solution {
public:
    Node* copyRandomList(Node* head) {
         if(head == NULL) return NULL;
        Node * ans = new Node(head->val);
         Node * low = head;
         ans->next = head->next;
         Node * pre = head->next;
         low->next = ans;
         low = low->next;
         Node * newStart = ans;
         while(pre){
             low = low->next;
             Node * temp = new Node(low->val);
             low->next = temp;
             temp->next = pre;
             pre = pre->next;
         }

         low = head;
         while(low){
             if(low->random == NULL)
                low->next->random = NULL;
            else{
                low->next->random = low->random->next;
            }
            low = low->next->next;
         }
         low = head;
         pre = head->next;

         while(pre->next){
             low->next = low->next->next;
             pre->next = pre->next->next;
             low = low->next;
             pre = pre->next;
         }
         low->next = NULL;

         return newStart;

    }
};
```



### 字符串

#### 替换空格

```c++
class Solution {
public:
    string replaceSpace(string s) {
        int space_num = 0;
        char ans[30010];
        string ANS="";
        for(int i=0;i<s.size();i++){
            if(s[i] == ' ')
            space_num++;
        }
        int sum_num = s.size() + space_num*2;
        int tail = sum_num-1;
        for(int i = s.size()-1;i>=0;i--)
        {
            if(s[i] == ' ')
            {
                ans[tail] = '0';
                ans[tail-1] = '2';
                ans[tail-2] = '%';
                tail-=3;
            }
            else
            {
                ans[tail] = s[i];
                tail--;
            }
        }
        for(int i=0;i<sum_num;i++)
            ANS+=ans[i];
        return ANS;
    }
};
```

### 查找算法

#### 在排序数组中查找数字 I（二分查找）

```c++
class Solution {
public:
    int binarySearch(vector<int>& nums, int target, bool lower) {
        int left = 0, right = (int)nums.size() - 1, ans = (int)nums.size();
        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums[mid] > target || (lower && nums[mid] >= target)) {
                right = mid - 1;
                ans = mid;
            } else {
                left = mid + 1;
            }
        }
        return ans;
    }

    int search(vector<int>& nums, int target) {
        int leftIdx = binarySearch(nums, target, true);
        int rightIdx = binarySearch(nums, target, false) - 1;
        if (leftIdx <= rightIdx && rightIdx < nums.size() && nums[leftIdx] == target && nums[rightIdx] == target) {
            return rightIdx - leftIdx + 1;
        }
        return 0;
    }
};

```

#### 查找0~n-1中缺失的数字（二分，哈希）

```c++
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        int L = 0,R = nums.size()-1;
        
        while(L<=R){
            int mid = (L+R)>>1;
            if(nums[mid] == mid){
                L = mid + 1;
            }
            else
            {
                R = mid - 1;
            }
        }
        return L;
    }
};
```

#### 二维数组中查找

```c++
class Solution {
public:
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
        if (matrix.empty()||matrix[0].empty()) return false;
        int rows = matrix.size()-1;
        int columns = matrix[0].size()-1;
        int row = 0,column = columns;
        while(row>=0&&row<=rows&&column>=0&&column<=columns)
        {
            if(matrix[row][column] == target) return true;
            else if(matrix[row][column] > target)
                column--;
            else    
                row++;
        }
        return false;
    }
};
```

#### 数组中的逆序对（二分，归并排序）

时间复杂度：nlogn  空间复杂度： n

```c++
class Solution {
public:
    int reversePairs(vector<int>& nums) {
        vector<int> tmp(nums.size());
        return mergeSort(0,nums.size()-1,nums,tmp);
    }
private:
    int mergeSort(int L,int R,vector<int>& nums,vector<int>& tmp)
    {
        if(L >= R)
            return 0;//终止条件
        int mid = L+((R-L)>>1);
        int res = mergeSort(L,mid,nums,tmp) + mergeSort(mid+1,R,nums,tmp);

        int leftIndex=L,rightIndex=mid+1;
        for(int k=L;k<=R;k++)
            tmp[k] = nums[k];
        for(int k=L;k<=R;k++)
        {
            if(leftIndex == mid+1)
                nums[k] = tmp[rightIndex++];
            else if(rightIndex == R+1 || tmp[leftIndex]<=tmp[rightIndex])
                nums[k] = tmp[leftIndex++];
            else{ // tmp[leftIndex] > tmp[rightIndex]
                nums[k] = tmp[rightIndex++];
                res += mid-leftIndex+1;//统计逆序对
            }
        }
        return res;
    }
};
```



### 搜索与回溯算法（简单）

#### 从上到下打印二叉树 I（BFS）

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
    vector<int> levelOrder(TreeNode* root) {
        if(root == NULL) return {};
        queue<TreeNode*>Queue;
        vector<int> res;//储存二叉树的结点
        Queue.push(root);
        while(!Queue.empty()){
            TreeNode * node;
            node = Queue.front();
            Queue.pop();
            res.push_back(node->val);
            if(node->left){
                Queue.push(node->left);
            }
            if(node->right){
                Queue.push(node->right);
            }
        }
        return res;
    }
};
```

#### 从上到下打印二叉树 II

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
    vector<vector<int>> levelOrder(TreeNode* root) {
        queue<TreeNode*> Queue;
        vector<vector<int>> result;
        if(root)
            Queue.push(root);
        else
            return result;
        while(!Queue.empty()){
            vector<int> temp;
            int length = Queue.size();
            for(int i=0;i < length;i++){
                if(Queue.front()->left)
                    Queue.push(Queue.front()->left);
                if(Queue.front()->right)
                    Queue.push(Queue.front()->right);
                temp.push_back(Queue.front()->val);
                Queue.pop();
            }
            result.push_back(temp);
        }
        return result;
    }
};
```

#### 从上到下打印二叉树 III

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
    vector<vector<int>> levelOrder(TreeNode* root) {
        queue<TreeNode*> Queue;
        vector<vector<int>> result;
        if(root)
            Queue.push(root);
        else
            return result;
        while(!Queue.empty()){
            vector<int> temp;
            int length = Queue.size();
            for(int i=0;i < length;i++){
                if(Queue.front()->left) 
                    Queue.push(Queue.front()->left);
                if(Queue.front()->right)
                    Queue.push(Queue.front()->right);
                temp.push_back(Queue.front()->val);
                Queue.pop();
            }
          if(result.size() % 2 == 0)
            result.push_back(temp);
          else{
            reverse(temp.begin(),temp.end());  
            result.push_back(temp);
          }
        }
        return result;
    }
};
```



#### 二叉树的深度（dfs)

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
    int maxn = 0;
    void dfs(TreeNode * node,int Depth){
        maxn = max(maxn,Depth);
        if(node->left) 
            dfs(node->left,Depth+1);
        if(node->right)
            dfs(node->right,Depth+1);
        return;
    }
    int maxDepth(TreeNode* root) {
        if(root == NULL) return 0;
        dfs(root,1);
        return maxn;
    }
};
```

#### 对称的二叉树（中序遍历的做法）（未AC）

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
    vector<int> MidArr;
    void InOrder(TreeNode * node){
        if(!(node->left||node->right)){
            MidArr.push_back(node->val);
            return;
        }

        if(node->left) 
            InOrder(node->left);
        else 
            MidArr.push_back(-1);
        MidArr.push_back(node->val);
        if(node->right)
            InOrder(node->right);
        else
            MidArr.push_back(-1);
    }

    bool isSymmetric(TreeNode* root) {
        if(!root) return true;
        InOrder(root);
    
    int size = MidArr.size();
    
    for(int i=0;i<size/2;i++){
        if(MidArr[i] != MidArr[size-i-1])
            return false;
    }
    return true;
    }
};
```

#### 对称二叉树（二叉树镜像做法）

空间和时间复杂度太大

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
	bool isSymmetric(TreeNode* pRoot)
	{
        if (pRoot == nullptr)
			return true;
		TreeNode* cloneRoot = clone(pRoot);
		Mirror(cloneRoot);
        return isSameTree(pRoot,cloneRoot);
	}
	TreeNode* clone(TreeNode* pRoot) {   //深拷贝该二叉树
		if (pRoot == nullptr)
			return nullptr;
		TreeNode* cloneRoot = new TreeNode(pRoot->val);
		cloneRoot->left = clone(pRoot->left);
		cloneRoot->right = clone(pRoot->right);
		return cloneRoot;
	}
	void Mirror(TreeNode* cloneNode) {   //二叉树的镜像
		if (cloneNode == nullptr) 
			return;
		swap(cloneNode->left, cloneNode->right);
		Mirror(cloneNode->left);
		Mirror(cloneNode->right);
	}
	 bool isSameTree(TreeNode* t1,TreeNode* t2){   //判断两棵树是否相同
        if(t1==nullptr && t2==nullptr)
            return true;
        if(t1!=nullptr && t2!=nullptr && t1->val==t2->val) {
            bool left = isSameTree(t1->left, t2->left);
            bool right = isSameTree(t1->right, t2->right);
            return left && right;
        }
        return false;
    }
};
```



### 树（数据结构）

#### 二叉搜索树第k大结点

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
    vector<int> order;
    void InOrder(TreeNode * root){
        if(root->left) InOrder(root->left);
        order.push_back(root->val);
        if(root->right) InOrder(root->right);
    }
    int kthLargest(TreeNode* root, int k) {
        InOrder(root);
        reverse(order.begin(),order.end());
        return order[k-1];
    }
};
```

#### 判断是否为平衡二叉树

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
    bool flag = true;
    int depth(TreeNode * root){
        if(root == NULL) return 1;

        int left = depth(root->left);
        int right = depth(root->right);
        if(abs(left - right) > 1) 
            flag = false;
        int maxDepth = max(left,right) + 1;
        return maxDepth;
    }
    bool isBalanced(TreeNode* root) {
        int deep = depth(root);
        return flag;
    }
};
```

#### 通过后序遍历判断是否为二叉搜索树

```c++
class Solution {
public:
    bool flag = true;
    bool check(int left,int  right,vector<int>& postorder){
        if(flag == false) return false;
        if(left >= right) return true;
        int root = right;
        int rightTreeStart;
         for(int i=left;i<right;i++){
             if(postorder[i] > postorder[root]){
                 rightTreeStart = i;
                 break;
             }
         }
         int p = left;
         for(p=left;p<root;p++){
             if(postorder[p] >= postorder[root]&&p<rightTreeStart) 
                break;
             if(postorder[p] <= postorder[root]&&p>=rightTreeStart)
                break;
         }     
         if(p == root && check(left,rightTreeStart-1,postorder) && check(rightTreeStart,root-1,postorder))
            return true;
        flag = false;
        return false;
    }
    bool verifyPostorder(vector<int>& postorder) {
        if(check(0,postorder.size()-1,postorder))
            return true;
        return false;
    }
};
```

#### 二叉树中和为某一值的路径

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
    vector<vector<int>> PathAns;
    void solve(int sum,vector<int> path,TreeNode * root,int target){
        path.push_back(root->val);
        sum += root->val;
        if(root->left==NULL && root->right==NULL){
            if(sum == target){
                PathAns.push_back(path);
                return;
            }
        }
        if(root->left) 
            solve(sum,path,root->left,target);
        if(root->right)
            solve(sum,path,root->right,target);
    }
    vector<vector<int>> pathSum(TreeNode* root, int target) {
        if(root == NULL){
            return PathAns;
        }
        vector<int> path;
        solve(0,path,root,target);
        return PathAns;
    }
};
```

#### 二叉搜索树于双向链表

```c++
/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* left;
    Node* right;

    Node() {}

    Node(int _val) {
        val = _val;
        left = NULL;
        right = NULL;
    }

    Node(int _val, Node* _left, Node* _right) {
        val = _val;
        left = _left;
        right = _right;
    }
};
*/
class Solution {
public:

    Node * head;Node * pre;
    void dfs(Node * cur){
        if(cur == NULL) return ;
        dfs(cur->left);
        if(pre != NULL){
            pre->right = cur;
        }else{
            head = cur;
        }
        cur->left = pre;
        pre = cur;
        dfs(cur->right);
    }
    Node* treeToDoublyList(Node* root) {
        if(root == NULL) return NULL;
         dfs(root);
         head->left = pre;
         pre->right = head;
         return head;
    }
};
```

#### 重建二叉树（RE代码）

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
    vector<int> hash;
    vector<int> cut(vector<int>& arr,int left,int right){
        vector<int>::const_iterator Fist = arr.begin() + left; // 找到第二个迭代器
        vector<int>::const_iterator Second = arr.begin() + right; // 找到第三个    迭代器
        vector<int> ans(left,right);
        return ans;
    }
    TreeNode * build(vector<int>& preorder,vector<int>& inorder){
        if(preorder.size() == 0&&inorder.size() == 0) return NULL;
        if(preorder.size() == 1 && inorder.size() == 1){
            TreeNode * root = new TreeNode(preorder[0]);
            return root;
        }
        int rootInOrder = hash[preorder[0]];
       TreeNode * root = new TreeNode(preorder[0]);
        vector<int> InorderLeft;
        vector<int> InorderRight;
        vector<int> preorderLeft;
        vector<int> preorderRight;
        InorderLeft = cut(inorder,0,rootInOrder-1);
        InorderRight = cut(inorder,rootInOrder+1,inorder.size()-1);
        preorderLeft = cut(preorder,0,rootInOrder);
        preorderRight = cut(preorder,rootInOrder+1,preorder.size()-1);
        root->left = build(preorderLeft,InorderLeft);
        root->right = build(preorderRight,InorderRight);
        return root;
    }
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        if(preorder.size() == 0 && inorder.size() == 0)
            return NULL;
        for(int i=0;i<preorder.size();i++){
            for(int j=0;j<inorder.size();j++)
                if(preorder[i] == inorder[j])
                    hash[i] = j;
        }
        TreeNode * root;   
        root = build(preorder,inorder);
        return root;
    }
};
```

#### 树的子结构

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
    bool DoesTree1HaveTree2(TreeNode * root1,TreeNode * root2)
    {
        if(root2 == NULL)
            return true;
        if(root1 == NULL)
            return false;
        if(!(root1->val == root2->val))
            return false;
        return DoesTree1HaveTree2(root1->left,root2->left) && DoesTree1HaveTree2(root1->right,root2->right);
    }
    bool HasSubTree(TreeNode * root1,TreeNode * root2)
    {
         bool result = false;
         if(root1 && root2)
         {
             if(root1->val == root2->val)
             {
                 result = DoesTree1HaveTree2(root1,root2);
             }
             if(!result)
                result = HasSubTree(root1->left,root2);//判断root2是不是root左子树的子结构
            if(!result)
                result = HasSubTree(root1->right,root2);//判断root2是不是root右子树的子结构
         }
         return result;
    }
    bool isSubStructure(TreeNode* A, TreeNode* B) {
        if(A == NULL || B == NULL)
            return false;
        return HasSubTree(A,B);
    }
};
```



### 动态规划

#### 连续子数组的最大和

```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int size = nums.size();
        int pre = 0;
        int maxAns = nums[0];
        for(int i=0;i<size;i++){
            pre = max(nums[i],pre+nums[i]);
            maxAns = max(maxAns,pre);
        }
        return maxAns;
    }
};
```

#### 礼物的最大价值

```c++
class Solution {
public:
    int maxValue(vector<vector<int>>& grid) {
        const int rows = grid.size();
        const int cols = grid[0].size();
        for(int j=1;j < cols;j++){
            grid[0][j] = grid[0][j-1] + grid[0][j];
        }
        for(int i=1;i < rows;i++){
            grid[i][0] = grid[i-1][0] + grid[i][0];
        }
        for(int i=1;i < rows;i++){
            for(int j=1;j< cols;j++){
                grid[i][j] = max(grid[i-1][j],grid[i][j-1]) + grid[i][j];
            }
        }
        return grid[rows-1][cols-1];
    }
};
```

#### 把数字翻译成字符串，滚动数组优化

```c++
public:
    int translateNum(int num) {
    string nums = to_string(num);
    int second = 0,first = 0,r = 1;
    for(int i=0;i<nums.size();i++){
        second = first;
        first = r;
        r = 0;
        r += first;
        if(i == 0) continue;
        int num1 = (int)(nums[i-1] - '0');
        int num2 = (int)(nums[i] - '0');
        if(num1*10 + num2 <= 25 && num1*10 + num2 >=10 ){
            r += second;
        }
    }
    return r;}
};
```

#### 剪绳子

```c++
class Solution {
public:
    int cuttingRope(int n) {
        if(n == 2) return 1;
        if(n == 3) return 2;
        int dp[n+1];
        memset(dp,0,sizeof(dp));
        dp[0] = 0;dp[1] = 1; dp[2] = 2;dp[3] = 3;
        int maxn = 0;
        for(int i=4;i<=n;i++){
            for(int j=1;j<=i/2;j++){
                dp[i] = max(dp[i],dp[j]*dp[i-j]);
            }
        }
        return dp[n];
    }
};
```

#### 股票的最大价值

```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int inf = 1e9;
        int minprice = inf, maxprofit = 0;
        for (int price: prices) {
            maxprofit = max(maxprofit, price - minprice);
            minprice = min(price, minprice);
        }
        return maxprofit;
    }
};
```

#### 第n个丑数

```c++
class Solution {
public:
    int nthUglyNumber(int n) {
        int a = 0,b = 0,c = 0;
        int dp[n];
        dp[0] = 1;
        for(int i=1;i<n;i++){
            int n2=dp[a]*2,n3=dp[b]*3,n5=dp[c]*5;
            dp[i] = min(min(n2,n3),n5);
            if(dp[i] == n2) a++;
            if(dp[i] == n3) b++;
            if(dp[i] == n5) c++; 
        }
        return dp[n-1];
    }
};
```

