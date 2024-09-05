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
4. [Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/)
5. [Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)
6. [Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/)
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
