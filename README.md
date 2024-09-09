# blind75

## Arrays

1. [Two Sum](https://leetcode.com/problems/two-sum/)

> Given an array of integers `nums` and an integer `target`, return _indices of the two numbers such that they add up to `target`_.
> 
> You may assume that each input would have **_exactly_ one solution**, and you may not use the _same_ element twice.
> 
> You can return the answer in any order.
> 
> **Example 1:**
> 
> **Input:** nums = [2,7,11,15], target = 9
> **Output:** [0,1]
> **Explanation:** Because nums[0] + nums[1] == 9, we return [0, 1].
> 
> **Example 2:**
> 
> **Input:** nums = [3,2,4], target = 6
> **Output:** [1,2]
> 
> **Example 3:**
> 
> **Input:** nums = [3,3], target = 6
> **Output:** [0,1]
> 
> **Constraints:**
> 
> - `2 <= nums.length <= 104`
> - `-109 <= nums[i] <= 109`
> - `-109 <= target <= 109`
> - **Only one valid answer exists.**
> 
> **Follow-up:** Can you come up with an algorithm that is less than `O(n2)` time complexity? 

Iterate over the list, add elements to a `{element: index}` map. During the iteration, if `target - num` is found in the map, we have found `element1 + element2 = target`, we'll return those indices.

```python
class Solution:
  def twoSum(self, nums: List[int], target: int) -> List[int]:
    """
    O(n), O(n)
    """
    
    m = {}
    for i, num in enumerate(nums):
      if target - num in m:
        return i, m[target - num]
      m[num] = i
```

2. [Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

> You are given an array `prices` where `prices[i]` is the price of a given stock on the `ith` day.
> 
> You want to maximize your profit by choosing a **single day** to buy one stock and choosing a **different day in the future** to sell that stock.
> 
> Return _the maximum profit you can achieve from this transaction_. If you cannot achieve any profit, return `0`.
> 
> **Example 1:**
> 
> **Input:** prices = [7,1,5,3,6,4]
> **Output:** 5
> **Explanation:** Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
> Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
> 
> **Example 2:**
> 
> **Input:** prices = [7,6,4,3,1]
> **Output:** 0
> **Explanation:** In this case, no transactions are done and the max profit = 0.
> 
> **Constraints:**
> 
> - `1 <= prices.length <= 105`
> - `0 <= prices[i] <= 104`

One pass, sliding window approach: keep track of minimum price and calculate the maximum profit for each `price - min_price`.

```python
class Solution:
  def maxProfit(self, prices: List[int]) -> int:
    """
    O(n), O(1)
    """

    min_price, max_profit = float("inf"), 0
    for price in prices:
      min_price = min(min_price, price)
      max_profit = max(max_profit, price - min_price)

  return max_profit
```

3. [Contains Duplicate](https://leetcode.com/problems/contains-duplicate/)

> Given an integer array `nums`, return `true` if any value appears **at least twice** in the array, and return `false` if every element is distinct.
> 
> **Example 1:**
> 
> **Input:** nums = [1,2,3,1]
> 
> **Output:** true
> 
> **Explanation:**
> 
> The element 1 occurs at the indices 0 and 3.
> 
> **Example 2:**
> 
> **Input:** nums = [1,2,3,4]
> 
> **Output:** false
> 
> **Explanation:**
> 
> All elements are distinct.
> 
> **Example 3:**
> 
> **Input:** nums = [1,1,1,3,3,4,3,2,4,2]
> 
> **Output:** true
> 
> **Constraints:**
> 
> - `1 <= nums.length <= 105`
> - `-109 <= nums[i] <= 109`

Check if any element exist in the set.

```python
class Solution:
  def containsDuplicate(self, nums: List[int]) -> bool:
    """
    O(n), O(n)
    """
    
    s = set()
    for num in nums:
      if num in s:
        return True
      s.add(num)
```

4. [Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/)

> Given an integer array `nums`, return _an array_ `answer` _such that_ `answer[i]` _is equal to the product of all the elements of_ `nums` _except_ `nums[i]`.
> 
> The product of any prefix or suffix of `nums` is **guaranteed** to fit in a **32-bit**integer.
> 
> You must write an algorithm that runs in `O(n)` time and without using the division operation.
> 
> **Example 1:**
> 
> **Input:** nums = [1,2,3,4]
> **Output:** [24,12,8,6]
> 
> **Example 2:**
> 
> **Input:** nums = [-1,1,0,-3,3]
> **Output:** [0,0,9,0,0]
> 
> **Constraints:**
> 
> - `2 <= nums.length <= 105`
> - `-30 <= nums[i] <= 30`
> - The product of any prefix or suffix of `nums` is **guaranteed** to fit in a **32-bit** integer.
> 
> **Follow up:** Can you solve the problem in `O(1)` extra space complexity? (The output array **does not** count as extra space for space complexity analysis.)

2 pointers, we create left and right product lists.

Instead of dividing the product of all by the number at the index, we can use the product of elements to the left and the product of elements to the right. Multiplying them will give us the desired result.

Instead of using two arrays to store the left and right products, we'll just use one and use 2 pointers to keep track of the products, and update the result array accordingly.

```python
class Solution:
  def productExceptSelf(self, nums: List[int]) -> List[int]:
    """
    O(n), O(1)
    """
    
    N = len(nums)
    res = [1] * N
    
    prefix = 1
    for i in range(N):
      res[i] = prefix
      prefix *= nums[i]
    
    postfix = 1
    for i in reversed(range(N)):
      res[i] *= postfix
      postfix *= nums[i]
    
    return res
```

5. [Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)

> Given an integer array `nums`, find the subarray with the largest sum, and return _its sum_.
> 
> **Example 1:**
> 
> **Input:** nums = [-2,1,-3,4,-1,2,1,-5,4]
> **Output:** 6
> **Explanation:** The subarray [4,-1,2,1] has the largest sum 6.
> 
> **Example 2:**
> 
> **Input:** nums = [1]
> **Output:** 1
> **Explanation:** The subarray [1] has the largest sum 1.
> 
> **Example 3:**
> 
> **Input:** nums = [5,4,-1,7,8]
> **Output:** 23
> **Explanation:** The subarray [5,4,-1,7,8] has the largest sum 23.
> 
> **Constraints:**
> 
> - `1 <= nums.length <= 105`
> - `-104 <= nums[i] <= 104`
> 
> **Follow up:** If you have figured out the `O(n)` solution, try coding another solution using the **divide and conquer** approach, which is more subtle.

Kadane's algorithm

Any subarray whose sum is positive is worth keeping. Whenever the sum of the array is negative, we know the entire array is not worth keeping, so we'll reset it back to an empty array.

However, we don't actually need to build the subarray, we can just keep a variable `curr_sum` and add the values of each element there. When it becomes negative, we reset it to 0 (an empty array).

```python
class Solution:
  def maxSubArray(self, nums: List[int]) -> int:
    """
    O(N), O(1)
    """
    
    curr_sum = max_sum = -inf # curr_sum = max_sum != 0 because for nums = [-1], we should return -1
    for num in nums:
      curr_sum = max(num, curr_sum + num)
      max_sum = max(max_sum, curr_sum)
    
    return max_sum
```

6. [Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/)

> Given an integer array `nums`, find a subarray that has the largest product, and return _the product_.
> 
> The test cases are generated so that the answer will fit in a **32-bit** integer.
> 
> **Example 1:**
> 
> **Input:** nums = [2,3,-2,4]
> **Output:** 6
> **Explanation:** [2,3] has the largest product 6.
> 
> **Example 2:**
> 
> **Input:** nums = [-2,0,-1]
> **Output:** 0
> **Explanation:** The result cannot be 2, because [-2,-1] is not a subarray.
> 
> **Constraints:**
> 
> - `1 <= nums.length <= 2 * 104`
> - `-10 <= nums[i] <= 10`
> - The product of any subarray of `nums` is **guaranteed** to fit in a **32-bit** integer.

This can be interpreted as a problem of getting the highest combo chain.

The simplest case is when the numbers are all positive numbers, where we only need to keep on multiplying the accumulated result to get a bigger and bigger combo chain.

However, two things can disrupt the combo chain: zeroes and negative numbers.
1. Zeroes reset the combo chain. We store the high score in a placeholder and restart the combo chain after zero. If we encounter another combo chain higher than the recorded high score, we need to update the result.
2. Negative numbers flip the result. If we have another negative number further, the 2 flips won't disrupt the combo chain.

We keep track of the `max_so_far`, the accumulated product of positive numbers and the `min_so_far`, to properly handle negative numbers.

`max_so_far` is updated by taking the maximum value among:
1. Current number.
  - This value will be picked if the accumulated product has been really bad (even compared to the current number). This can happen when the current number has a preceding zero (e.g. [0,4]) or is preceded by a single negative number (e.g. [-3,5]).
2. Product of last `max_so_far` and current number. 
  - This value will be picked if the accumulated product has been steadily increasing (all positive numbers).
3. Product of last `min_so_far` and current number. 
  - This value will be picked if the current number is a negative number and the combo chain has been disrupted by a single negative number before (In a sense, this value is like an antidote to an already poisoned combo chain).

`min_so_far` is updated in using the same three numbers except that we are taking minimum among the above three numbers.

```python
class Solution:
  def maxProduct(self, nums: List[int]) -> int:
    """
    O(N), O(1)
    """
    
    res = -inf
    max_so_far = min_so_far = 1
    
    for num in nums:
      candidates = [num, num * min_so_far, num * max_so_far]
      min_so_far, max_so_far = min(candidates), max(candidates)
      res = max(res, max_so_far)
      
    return res
```

7. [Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)
8. [Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)
9. [3 Sum](https://leetcode.com/problems/3sum/)
10. [Container With Most Water](https://leetcode.com/problems/container-with-most-water/)

---

## Binary

11. [Sum of Two Integers](https://leetcode.com/problems/sum-of-two-integers/)
12. [Number of 1 Bits](https://leetcode.com/problems/number-of-1-bits/)
13. [Counting Bits](https://leetcode.com/problems/counting-bits/)
14. [Missing Number](https://leetcode.com/problems/missing-number/)
15. [Reverse Bits](https://leetcode.com/problems/reverse-bits/)

---

## Dynamic Programming

16. [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)
17. [Coin Change](https://leetcode.com/problems/coin-change/)
18. [Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)
19. [Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)
20. [Word Break Problem](https://leetcode.com/problems/word-break/)
21. [Combination Sum](https://leetcode.com/problems/combination-sum-iv/)
22. [House Robber](https://leetcode.com/problems/house-robber/)
23. [House Robber II](https://leetcode.com/problems/house-robber-ii/)
24. [Decode Ways](https://leetcode.com/problems/decode-ways/)
25. [Unique Paths](https://leetcode.com/problems/unique-paths/)
26. [Jump Game](https://leetcode.com/problems/jump-game/)

---

## Graph

27. [Clone Graph](https://leetcode.com/problems/clone-graph/)
28. [Course Schedule](https://leetcode.com/problems/course-schedule/)
29. [Pacific Atlantic Water Flow](https://leetcode.com/problems/pacific-atlantic-water-flow/)
30. [Number of Islands](https://leetcode.com/problems/number-of-islands/)
31. [Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)
32. [Alien Dictionary](https://leetcode.com/problems/alien-dictionary/)
33. [Graph Valid Tree](https://leetcode.com/problems/graph-valid-tree/)
34. [Number of Connected Components in an Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/)

---

## Interval

35. [Insert Interval](https://leetcode.com/problems/insert-interval/)
36. [Merge Intervals](https://leetcode.com/problems/merge-intervals/)
37. [Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/)
38. [Meeting Rooms](https://leetcode.com/problems/meeting-rooms/)
39. [Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/)

---

## Linked List

40. [Reverse a Linked List](https://leetcode.com/problems/reverse-linked-list/)
41. [Detect Cycle in a Linked List](https://leetcode.com/problems/linked-list-cycle/)
42. [Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)
43. [Merge K Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)
44. [Remove Nth Node From End Of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)
45. [Reorder List](https://leetcode.com/problems/reorder-list/)

---

## Matrix

46. [Set Matrix Zeroes](https://leetcode.com/problems/set-matrix-zeroes/)
47. [Spiral Matrix](https://leetcode.com/problems/spiral-matrix/)
48. [Rotate Image](https://leetcode.com/problems/rotate-image/)
49. [Word Search](https://leetcode.com/problems/word-search/)

---

### String

50. [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
51. [Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/)
52. [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)
53. [Valid Anagram](https://leetcode.com/problems/valid-anagram/)
54. [Group Anagrams](https://leetcode.com/problems/group-anagrams/)
55. [Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)
56. [Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)
57. [Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)
58. [Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/)
59. [Encode and Decode Strings](https://leetcode.com/problems/encode-and-decode-strings/)

---

### Tree

60. [Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)
61. [Same Tree](https://leetcode.com/problems/same-tree/)
62. [Invert/Flip Binary Tree](https://leetcode.com/problems/invert-binary-tree/)
63. [Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)
64. [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
65. [Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)
66. [Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/)
67. [Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)
68. [Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)
69. [Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)
70. [Lowest Common Ancestor of BST](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)
71. [Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)
72. [Add and Search Word](https://leetcode.com/problems/add-and-search-word-data-structure-design/)
73. [Word Search II](https://leetcode.com/problems/word-search-ii/)

---

## Heap

74. [Merge K Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)
75. [Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)
76. [Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)

---
