

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

