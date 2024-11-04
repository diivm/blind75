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

> Suppose an array of length `n` sorted in ascending order is **rotated** between `1` and `n` times. For example, the array `nums = [0,1,2,4,5,6,7]` might become:
> 
> - `[4,5,6,7,0,1,2]` if it was rotated `4` times.
> - `[0,1,2,4,5,6,7]` if it was rotated `7` times.
> 
> Notice that **rotating** an array `[a[0], a[1], a[2], ..., a[n-1]]` 1 time results in the array `[a[n-1], a[0], a[1], a[2], ..., a[n-2]]`.
> 
> Given the sorted rotated array `nums` of **unique** elements, return _the minimum element of this array_.
> 
> You must write an algorithm that runs in `O(log n) time.`
> 
> **Example 1:**
> 
> **Input:** nums = [3,4,5,1,2]
> **Output:** 1
> **Explanation:** The original array was [1,2,3,4,5] rotated 3 times.
> 
> **Example 2:**
> 
> **Input:** nums = [4,5,6,7,0,1,2]
> **Output:** 0
> **Explanation:** The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.
> 
> **Example 3:**
> 
> **Input:** nums = [11,13,15,17]
> **Output:** 11
> **Explanation:** The original array was [11,13,15,17] and it was rotated 4 times. 
> 
> **Constraints:**
> 
> - `n == nums.length`
> - `1 <= n <= 5000`
> - `-5000 <= nums[i] <= 5000`
> - All the integers of `nums` are **unique**.
> - `nums` is sorted and rotated between `1` and `n` times.

In a rotated array we have an inflection point, which will be the minimum element:
- All the points to the left of the inflection point > first element of the array.
- All the points to the right of the inflection point < first element of the array.

The condition that decides the search direction will be different from a conventional binary search. In a sorted array, we have the property `first element < last element`. Here, we have the following conditions:
- If `nums[m] > nums[r]`, inflection point is to the right of mid. We also know that mid is greater than at least one number to the right, so we can use `l = m + 1` and never consider mid again.
```
[3,4,5,6,7,8,9,1,2]

nums[l] = 3
nums[m] = 7
nums[r] = 2
inflection = 1
```
- Else, `nums[m] <= nums[r]`, inflection point is either at or to the left of mid. It is possible for the mid to store a smaller value than at least one index at right, so we don't discard it by doing `r = m - 1`, it might still have the minimum value.
```
[8,9,1,2,3,4,5,6,7]

nums[l] = 8
nums[m] = 3
nums[r] = 7
inflection = 1
```

```python
class Solution:
  def findMin(self, nums: List[int]) -> int:
    """
    O(logN), O(1)
    """
    
    l, r = 0, len(nums) - 1
    
    # early exit: no rotation
    if nums[l] < nums[r]:
      return nums[l]
    
    while l < r:
      m = l + (r - l)//2
      if nums[m] > nums[r]:
        l = m + 1
      else:
        r = m
    
    return nums[l]
```

8. [Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)

> There is an integer array `nums` sorted in ascending order (with **distinct** values).
> 
> Prior to being passed to your function, `nums` is **possibly rotated** at an unknown pivot index `k` (`1 <= k < nums.length`) such that the resulting array is `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]` (**0-indexed**). For example, `[0,1,2,4,5,6,7]` might be rotated at pivot index `3` and become `[4,5,6,7,0,1,2]`.
> 
> Given the array `nums` **after** the possible rotation and an integer `target`, return _the index of_ `target` _if it is in_ `nums`_, or_ `-1` _if it is not in_ `nums`.
> 
> You must write an algorithm with `O(log n)` runtime complexity.
> 
> **Example 1:**
> 
> **Input:** nums = [4,5,6,7,0,1,2], target = 0
> **Output:** 4
> 
> **Example 2:**
> 
> **Input:** nums = [4,5,6,7,0,1,2], target = 3
> **Output:** -1
> 
> **Example 3:**
> 
> **Input:** nums = [1], target = 0
> **Output:** -1
> 
> **Constraints:**
> 
> - `1 <= nums.length <= 5000`
> - `-104 <= nums[i] <= 104`
> - All values of `nums` are **unique**.
> - `nums` is an ascending array that is possibly rotated.
> - `-104 <= target <= 104`

Revised binary search: we add some additional condition checks in the normal binary search in order to better narrow down the scope of the search.

If there's no rotation, both left and right subarrays are sorted. If there's rotation, we'll get at most one sorted subarray. 

We compare the target with the sorted subarray to figure out which subarray to retain for the next iteration. It is straightforward to do so, we can simply compare target with the two boundary values.

```python
class Solution:
  def search(self, nums: List[int], target: int) -> int:
    """
    O(logN), O(1)
    """
    
    l, r = 0, len(nums) - 1
    
    # while l < r: # incorrect
    while l <= r:
      m = l + (r - l)//2
      
      if nums[m] == target:
        return m
      # elif nums[m] > nums[l]: # incorrect
      elif nums[m] >= nums[l]: # left subarray is sorted
        if nums[l] <= target < nums[m]: # target in left subarray
          r = m - 1
        else: # target in right subarray
          l = m + 1
      else: # right subarray is sorted
        if nums[m] < target <= nums[r]: # target in right subarray
          l = m + 1
        else: # target in left subarray
          r = m - 1
    
    return -1
```

9. [3 Sum](https://leetcode.com/problems/3sum/)

> Given an integer array nums, return all the triplets `[nums[i], nums[j], nums[k]]` such that `i != j`, `i != k`, and `j != k`, and `nums[i] + nums[j] + nums[k] == 0`.
> 
> Notice that the solution set must not contain duplicate triplets.
> 
> **Example 1:**
> 
> **Input:** nums = [-1,0,1,2,-1,-4]
> **Output:** [[-1,-1,2],[-1,0,1]]
> **Explanation:** 
> nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
> nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
> nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0.
> The distinct triplets are [-1,0,1] and [-1,-1,2].
> Notice that the order of the output and the order of the triplets does not matter.
> 
> **Example 2:**
> 
> **Input:** nums = [0,1,1]
> **Output:** []
> **Explanation:** The only possible triplet does not sum up to 0.
> 
> **Example 3:**
> 
> **Input:** nums = [0,0,0]
> **Output:** [[0,0,0]]
> **Explanation:** The only possible triplet sums up to 0.
> 
> **Constraints:**
> 
> - `3 <= nums.length <= 3000`
> - `-105 <= nums[i] <= 105`

Sorting + 2 pointers approach to find all the applicable pairs. 

This is a minimum O(N^2) time problem, so the O(NlogN) additional sort doesn't add to the time complexity.

In the first loop, we keep track of the current element. We mark the left and right pointers to the next and last element respectively. Creating another loop for the left and right pointers, we try to find the triplets where the sum is zero, shifting left and right pointers accordingly to get a zero sum. 

For the current element, there might be multiple left-right pairs where the triplet sum is zero, so we need to account for all of those values.

```python
class Solution:
  def threeSum(self, nums: List[int]) -> List[List[int]]:
    """
    O(N^2), O(N)
    """
    
    nums.sort()
    
    N = len(nums)
    res = []
    
    for i in range(N):
      # optimization: skip duplicates for the current element
      if i > 0 and nums[i] == nums[i - 1]:
        continue
      
      l, r = i + 1, N - 1
      while l < r:
        three_sum = nums[i] + nums[l] + nums[r]
        
        if three_sum < 0: # move towards right to get a bigger sum
          l += 1
        elif three_sum > 0: # move towards left to get a smaller sum
          r -= 1
        else: # zero sum, capture the result
          res.append((nums[i], nums[l], nums[r]))
          
          # to check if we can have another left-right pair to get a zero sum, shift the pointers for the next iteration
          l += 1
          r -= 1
          
          # optimization: skip duplicates for the left element
          while l < r and nums[l] == nums[l - 1]:
            l += 1
          
          # optimization: skip duplicates for the right element
          while l < r and nums[r] == nums[r + 1]:
            r -= 1
    
    return res
```

No sort approach

We can put a combination of the 3 values in a set to avoid duplicates. Values in the combination should be ordered, otherwise we can have results with the same values in different positions.

The `seen` set is used to keep track of numbers we’ve encountered in the inner loop. This helps to efficiently check if the calculated `third` number exists.

```python
class Solution:
  def threeSum(self, nums: List[int]) -> List[List[int]]:
    """
    O(N^2), O(N)
    """
    
    res = set()
    dups = set() # optimization: avoid duplicates in outerloop
    
    for i, first in enumerate(nums):
      if first not in dups:
        dups.add(first)
        seen = set() # track of numbers in innerloop
        
        for second in nums[i+1:]:
          third = -(first + second) # zero sum
          if third in seen:
            # To form a valid triplet, third must be a number that has already been encountered while iterating through the nums array.
            res.add(tuple(sorted((first, second, third))))
            seen.add(second)
    
    return res
```

10. [Container With Most Water](https://leetcode.com/problems/container-with-most-water/)

> You are given an integer array `height` of length `n`. There are `n` vertical lines drawn such that the two endpoints of the `ith` line are `(i, 0)` and `(i, height[i])`.
> 
> Find two lines that together with the x-axis form a container, such that the container contains the most water.
> 
> Return _the maximum amount of water a container can store_.
> 
> **Notice** that you may not slant the container.
> 
> **Example 1:**
> 
> ![](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/07/17/question_11.jpg)
> 
> **Input:** height = [1,8,6,2,5,4,8,3,7]
> **Output:** 49
> **Explanation:** The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.
> 
> **Example 2:**
> 
> **Input:** height = [1,1]
> **Output:** 1
> 
> **Constraints:**
> 
> - `n == height.length`
> - `2 <= n <= 105`
> - `0 <= height[i] <= 104`

2 pointers

`area = width * height`, so we need to maximise both to get the largest area.

The area formed between the lines will always be limited by the height of the shorter line.

We start with the exterior most lines (max width), and gradually reduce to optimise for greatest width. By doing this, we won't gain any increase in area from the width, but we might overcome the reduction by the possible increase in height.

```python
class Solution:
  def maxArea(self, height: List[int]) -> int:
    """
    O(N), O(1)
    """
    
    l, r = 0, len(height) - 1
    max_area = 0
    
    while l < r:
      max_area = max(max_area, (r - l) * min(height[l], height[r]))
      
      if height[l] < height[r]:
        l += 1
      else:  
        r -= 1
    
    return res
```

11. [Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)

> Given an unsorted array of integers `nums`, return _the length of the longest consecutive elements sequence._
> 
> You must write an algorithm that runs in `O(n)` time.
> 
> **Example 1:**
> 
> **Input:** nums = [100,4,200,1,3,2]
> **Output:** 4
> **Explanation:** The longest consecutive elements sequence is `[1, 2, 3, 4]`. Therefore its length is 4.
> 
> **Example 2:**
> 
> **Input:** nums = [0,3,7,2,5,8,4,6,0,1]
> **Output:** 9
> 
> **Constraints:**
> 
> - `0 <= nums.length <= 105`
> - `-109 <= nums[i] <= 109`

```python
class Solution:
  def longestConsecutive(self, nums: List[int]) -> int:
    """
    O(N), O(N)
    """
    
    nums_set = set(nums) # convert to set for O(1) time lookups
    max_len = 0
    
    for num in nums_set:
      # if this num is the start of a new sequence (i.e. its predecessor is not in the list), initiate new count for this sequence
      if (num - 1) not in nums_set:
        count = 1
        
        # incrementally check for consecutive integers, increasing the count
        while (num + 1) in nums_set:
          num += 1
          count += 1
        
        max_len = max(max_len, count)
    
    return max_len
```

---

## Binary

12. [Sum of Two Integers](https://leetcode.com/problems/sum-of-two-integers/)

> Given two integers `a` and `b`, return _the sum of the two integers without using the operators_ `+` _and_ `-`.
> 
> **Example 1:**
> 
> **Input:** a = 1, b = 2
> **Output:** 3
> 
> **Example 2:**
> 
> **Input:** a = 2, b = 3
> **Output:** 5
> 
> **Constraints:**
> 
> - `-1000 <= a, b <= 1000`

Since Python integers can be arbitrarily large, the function simulates a 32-bit integer environment to handle overflow conditions.

`a ^ b` computes the sum of `a` and `b` without considering the carry.

`(a & b) << 1` computes the carry. `a & b` generates the carry, and the left shift moves the carry to the left (to the next higher bit).

Both results are masked with `& MASK` to ensure they fit within 32 bits.

If `a` is less than `MAX_INT`, it is within the valid range for a signed 32-bit integer and can be returned directly.

If `a` is greater than `MAX_INT`, it indicates an overflow (i.e., the result should be negative in 32-bit representation). The expression `~(a ^ MASK)` converts it back to the negative representation by flipping the bits (two's complement).

```python
class Solution:
    def getSum(self, a: int, b: int) -> int:
      """
      O()
      """
      
      MASK = 0xFFFFFFFF  # 32 1-bits, used to limit results to 32 bits
      MAX_INT = 0x7FFFFFFF  # 31 1-bits, represents the maximum positive value for a signed 32-bit integer

      while b:
        # Calculate the sum without carrying
        a, b = (a ^ b) & MASK, ((a & b) << 1) & MASK
        # a becomes the sum of a and b without considering carry
        # b becomes the carry that needs to be added in the next iteration
      
      # If a is greater than MAX_INT, it means we have a negative result in 32-bit representation
      return a if a < MAX_INT else ~(a ^ MASK)
```

13. [Number of 1 Bits](https://leetcode.com/problems/number-of-1-bits/)

> Write a function that takes the binary representation of a positive integer and returns the number of 
> 
> set bits
> 
>  it has (also known as the [Hamming weight](http://en.wikipedia.org/wiki/Hamming_weight)).
> 
> **Example 1:**
> 
> **Input:** n = 11
> 
> **Output:** 3
> 
> **Explanation:**
> 
> The input binary string **1011** has a total of three set bits.
> 
> **Example 2:**
> 
> **Input:** n = 128
> 
> **Output:** 1
> 
> **Explanation:**
> 
> The input binary string **10000000** has a total of one set bit.
> 
> **Example 3:**
> 
> **Input:** n = 2147483645
> 
> **Output:** 30
> 
> **Explanation:**
> 
> The input binary string **1111111111111111111111111111101** has a total of thirty set bits.
> 
> **Constraints:**
> 
> - `1 <= n <= 231 - 1`
> 
> **Follow up:** If this function is called many times, how would you optimize it?

Instead of checking every bit of the number, we repeatedly flip the least-significant 1-bit of the number to 0, and add 1 to the result. As soon as the number becomes 0, we know that it does not have any more 1-bits, and we return the sum.

For any number n, doing a bit-wise AND of n and n-1 flips the least-significant 1-bit in n to 0.

```python
class Solution:
  def hammingWeight(self, n: int) -> int:
    """
    O(1), O(1)
    """
    
    bits = 0
    while n:
      bits += 1
      n &= n - 1
    
    return bits
```

14. [Counting Bits](https://leetcode.com/problems/counting-bits/)

> Given an integer `n`, return _an array_ `ans` _of length_ `n + 1` _such that for each_ `i` (`0 <= i <= n`)_,_ `ans[i]` _is the **number of**_ `1`_**'s** in the binary representation of_ `i`.
> 
> **Example 1:**
> 
> **Input:** n = 2
> **Output:** [0,1,1]
> **Explanation:**
> 0 --> 0
> 1 --> 1
> 2 --> 10
> 
> **Example 2:**
> 
> **Input:** n = 5
> **Output:** [0,1,1,2,1,2]
> **Explanation:**
> 0 --> 0
> 1 --> 1
> 2 --> 10
> 3 --> 11
> 4 --> 100
> 5 --> 101
> 
> **Constraints:**
> 
> - `0 <= n <= 105`
> 
> **Follow up:**
> 
> - It is very easy to come up with a solution with a runtime of `O(n log n)`. Can you do it in linear time `O(n)` and possibly in a single pass?
> - Can you do it without using any built-in function (i.e., like `__builtin_popcount`in C++)?

DP + Last bit set (rightmost set bit)

We set the rightmost 1-bit to zero using the bit trick `x & (x-1)`, and add 1 (the count of the marked bit) to the the resultant integer's count from the previous iterations.

Transition function: `P(x) = P(x & (x−1)) + 1`

```
x          x & (x-1)    No. of bits

0 (0)      _            0 (base case)
1 (1)      0 (0)        bits[0] + 1 = 0 + 1 = 1
2 (10)     0 (0)        bits[0] + 1 = 0 + 1 = 1
3 (11)     2 (10)       bits[2] + 1 = 1 + 1 = 2
4 (100)    0 (0)        bits[0] + 1 = 0 + 1 = 1
5 (101)    4 (100)      bits[4] + 1 = 1 + 1 = 2
6 (110)    4 (100)      bits[4] + 1 = 1 + 1 = 2
7 (111)    6 (110)      bits[6] + 1 = 2 + 1 = 3
8 (1000)   0 (0)        bits[0] + 1 = 0 + 1 = 1
```

```python
class Solution:
  def countBits(self, n: int) -> List[int]:
    """
    O(N), O(1)
    """
    
    dp = [0] * (n + 1)
    for num in range(1, n + 1):
      dp[num] = dp[num & (num-1)] + 1
    
    return dp
```

Similar solutions can be made by manipulating other bits.

Ex. DP + Least Significant Bit

Shift right by 1, use the resultant integer's count, and add 1 if the least significant bit (lost in the shift) was 1.

`P(x) = P(x >> 1) + (x & 1)`

```python
class Solution:
  def countBits(self, n: int) -> List[int]:
    """
    O(N), O(1)
    """
    
    dp = [0] * (n + 1)
    for num in range(1, n + 1):
      dp[num] = dp[num >> 1] + (num & 1)
    
    return dp
```

15. [Missing Number](https://leetcode.com/problems/missing-number/)

> Given an array `nums` containing `n` distinct numbers in the range `[0, n]`, return _the only number in the range that is missing from the array._
> 
> **Example 1:**
> 
> **Input:** nums = [3,0,1]
> **Output:** 2
> **Explanation:** n = 3 since there are 3 numbers, so all numbers are in the range [0,3]. 2 is the missing number in the range since it does not appear in nums.
> 
> **Example 2:**
> 
> **Input:** nums = [0,1]
> **Output:** 2
> **Explanation:** n = 2 since there are 2 numbers, so all numbers are in the range [0,2]. 2 is the missing number in the range since it does not appear in nums.
> 
> **Example 3:**
> 
> **Input:** nums = [9,6,4,2,3,5,7,0,1]
> **Output:** 8
> **Explanation:** n = 9 since there are 9 numbers, so all numbers are in the range [0,9]. 8 is the missing number in the range since it does not appear in nums.
> 
> **Constraints:**
> 
> - `n == nums.length`
> - `1 <= n <= 104`
> - `0 <= nums[i] <= n`
> - All the numbers of `nums` are **unique**.
> 
> **Follow up:** Could you implement a solution using only `O(1)` extra space complexity and `O(n)` runtime complexity?

XOR operations

We can take XOR of the given numbers, and then XOR of the range to get our missing number. We'll be XORing each number by itself (result 0) except the missing one.

```
input = [3, 5, 2, 1, 0]

XOR(input) ^ XOR(range)

= (3 ^ 5 ^ 2 ^ 1 ^ 0) ^ (0 ^ 1 ^ 2 ^ 3 ^ 4 ^ 5)

= (0 ^ 0) ^ (1 ^ 1) ^ (2 ^ 2) ^ (3 ^ 3) ^ (4) ^ (5 ^ 5)

= 0 ^ 0 ^ 0 ^ 0 ^ 4 ^ 0

= 4
```

```python
class Solution:
  def missingNumber(self, nums: List[int]) -> int:
    """
    O(N), O(1)
    """
    
    n = len(nums)
    missing = 0
    
    for i in range(n + 1): # 0 .. n
      missing ^= i # XOR by range
      if i < n:
        missing ^= nums[i] # XOR by input
    
    return missing
```

Because we know that input contains n numbers and that it is missing exactly one number on the range [0..n-1], we know that n definitely replaces the missing number in input. Therefore, if we initialise an integer to n and XOR it with every index and value, we will be left with the missing number.

```python
class Solution:
  def missingNumber(self, nums: List[int]) -> int:
    """
    alternate solution
    
    O(N), O(1)
    """
    
    x = len(nums)
    for i, num in enumerate(nums):
      x ^= i ^ num
    
    return x
```

Side note: Expected sum calculated via Gauss formula `n * (n + 1) / 2`  subtracted by the actual sum of the range will also give a valid answer.

16. [Reverse Bits](https://leetcode.com/problems/reverse-bits/)

> Reverse bits of a given 32 bits unsigned integer.
> 
> **Note:**
> 
> - Note that in some languages, such as Java, there is no unsigned integer type. In this case, both input and output will be given as a signed integer type. They should not affect your implementation, as the integer's internal binary representation is the same, whether it is signed or unsigned.
> - In Java, the compiler represents the signed integers using [2's complement notation](https://en.wikipedia.org/wiki/Two%27s_complement). Therefore, in **Example 2** above, the input represents the signed integer `-3` and the output represents the signed integer `-1073741825`.
> 
> **Example 1:**
> 
> **Input:** n = 00000010100101000001111010011100
> **Output:**    964176192 (00111001011110000010100101000000)
> **Explanation:** The input binary string **00000010100101000001111010011100** represents the unsigned integer 43261596, so return 964176192 which its binary representation is **00111001011110000010100101000000**.
> 
> **Example 2:**
> 
> **Input:** n = 11111111111111111111111111111101
> **Output:**   3221225471 (10111111111111111111111111111111)
> **Explanation:** The input binary string **11111111111111111111111111111101** represents the unsigned integer 4294967293, so return 3221225471 which its binary representation is **10111111111111111111111111111111**.
> 
> **Constraints:**
> 
> - The input must be a **binary string** of length `32`
> 
> **Follow up:** If this function is called many times, how would you optimize it?

Mask and shift (divide and conquer)

A strategy of divide and conquer, we divide the original 32-bits into blocks with fewer bits via bit masking, then we reverse each block via bit shifting, and at the end we merge the result of each block to obtain the final result. 

1. First, we break the original 32-bit into 2 blocks of 16 bits, and switch them. 
2. We then break the 16-bits block into 2 blocks of 8 bits. Similarly, we switch the position of the 8-bits blocks 
3. We then continue to break the blocks into smaller blocks, until we reach the level with the block of 1 bit. 
4. At each of the above steps, we merge the intermediate results into a single integer which serves as the input for the next step.

```python
class Solution:
  def reverseBits(self, n: int) -> int:
    """
    O(1), O(1)
    """
    
    n = (n & 0b11111111111111110000000000000000) >> 16 | (n & 0b00000000000000001111111111111111) << 16
    n = (n & 0b11111111000000001111111100000000) >> 8 | (n & 0b00000000111111110000000011111111) << 8
    n = (n & 0b11110000111100001111000011110000) >> 4 | (n & 0b00001111000011110000111100001111) << 4
    n = (n & 0b11001100110011001100110011001100) >> 2 | (n & 0b00110011001100110011001100110011) << 2
    n = (n & 0b10101010101010101010101010101010) >> 1 | (n & 0b01010101010101010101010101010101) << 1
    
    return n
```

---

## Dynamic Programming

17. [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)

> You are climbing a staircase. It takes `n` steps to reach the top.
> 
> Each time you can either climb `1` or `2` steps. In how many distinct ways can you climb to the top?
> 
> **Example 1:**
> 
> **Input:** n = 2
> **Output:** 2
> **Explanation:** There are two ways to climb to the top.
> 1. 1 step + 1 step
> 2. 2 steps
> 
> **Example 2:**
> 
> **Input:** n = 3
> **Output:** 3
> **Explanation:** There are three ways to climb to the top.
> 1. 1 step + 1 step + 1 step
> 2. 1 step + 2 steps
> 3. 2 steps + 1 step
> 
> **Constraints:**
> 
> - `1 <= n <= 45`

We can reach i$^{th}$ step in one of two ways:
1. Taking a single step from (i - 1)$^{th}$ step.
2. Taking two steps from (i - 2)$^{th}$ step.

So, the total number of ways to reach step i$^{th}$ is the sum of the number of ways from (i - 1)$^{th}$ step and the (i - 2$^{th}$ step.

We get the transition function: `dp[i] = dp[i - 1] + dp[i - 2]`. This is the same function as fibonacci: `Fib(n) = Fib(n−1) + Fib(n−2)`

```python
class Solution:
  def climbStairs(self, n: int) -> int:
    """
    O(N), O(N)
    """
    
    if n == 1:
      return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    
    for i in range(3, n + 1):
      dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]
```

We don't actually need to track the entire array, we can just do it 2 integers.

```python
class Solution:
  def climbStairs(self, n: int) -> int:
    """
    O(N), O(1)
    """
    
    if n == 1:
      return n
    
    a, b = 1, 2
    
    for _ in range(3, n + 1):
      a, b = b, a + b

    return b
```

18. [Coin Change](https://leetcode.com/problems/coin-change/)

> You are given an integer array `coins` representing coins of different denominations and an integer `amount` representing a total amount of money.
> 
> Return _the fewest number of coins that you need to make up that amount_. If that amount of money cannot be made up by any combination of the coins, return `-1`.
> 
> You may assume that you have an infinite number of each kind of coin.
> 
> **Example 1:**
> 
> **Input:** coins = [1,2,5], amount = 11
> **Output:** 3
> **Explanation:** 11 = 5 + 5 + 1
> 
> **Example 2:**
> 
> **Input:** coins = [2], amount = 3
> **Output:** -1
> 
> **Example 3:**
> 
> **Input:** coins = [1], amount = 0
> **Output:** 0
> 
> **Constraints:**
> 
> - `1 <= coins.length <= 12`
> - `1 <= coins[i] <= 231 - 1`
> - `0 <= amount <= 104`

Bottom-up DP

```python
class Solution:
  def coinChange(self, coins: List[int], amount: int) -> int:
    """
    O(N*C), O(N)
    """
    
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0
    
    for i in range(amount + 1):
      for coin in coins:
        if coin <= i:
          dp[i] = min(dp[i], dp[i-coin] + 1)
    
    return dp[amount] if dp[amount] != float("inf") else -1
```

Top down DP

```python
class Solution:
  def coinChange(self, coins: List[int], amount: int) -> int:
    """
    O(N*C), O(N)
    """
  
    @lru_cache(None) # TLE without caching
    def dfs(rem):
      if rem == 0:
        return 0
      
      min_cost = math.inf
      for coin in coins:
        if coin <= rem:
          res = dfs(rem - coin)
        if res != -1:
          min_cost = min(min_cost, res + 1)
      
      return min_cost if min_cost != math.inf else -1
  
    return dfs(amount)
```

19. [Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)

> Given an integer array `nums`, return _the length of the longest **strictly increasing subsequence**_.
> 
> **Example 1:**
> 
> **Input:** nums = [10,9,2,5,3,7,101,18]
> **Output:** 4
> **Explanation:** The longest increasing subsequence is [2,3,7,101], therefore the length is 4.
> 
> **Example 2:**
> 
> **Input:** nums = [0,1,0,3,2,3]
> **Output:** 4
> 
> **Example 3:**
> 
> **Input:** nums = [7,7,7,7,7,7,7]
> **Output:** 1
> 
> **Constraints:**
> 
> - `1 <= nums.length <= 2500`
> - `-104 <= nums[i] <= 104`
> 
> **Follow up:** Can you come up with an algorithm that runs in `O(n log(n))` time complexity?

```
input: [1, 2, 4, 3]
idx:    0  1  2  3
  

                                      .

                              /0

                            [1]

                          /1             \2            \3

                       [1, 2]            [1, 4]         [1, 3]

                     /2       \3            _             _

                 [1, 2, 4]     [1, 2, 4]

                /3                _

              X
```

We can store the result, and use to skip computation for similar branches in other subtrees.
  
```
LIS[3] = 1 (default = len([nums[3]]))


For LIS[2], we check if nums[2] < nums[3] => 4 < 3, no, so we can't include that.
So, LIS[2] = 1


For LIS[1], we check if nums[1] < nums[2] => 2 < 4, yes, so we can include that. We also check if nums[1] < nums[3] => 2 < 3, yes, so that's in contention as well.
So, LIS[1] = 1 + max(LIS[2], LIS[3]) = 1 + max(1, 1) = 2


For LIS[0], nums[0] < nums[1] => 1 < 2; nums[0] < nums[2] => 1 < 4; nums[0] < nums[3] => 1 < 3
LIS[0] = 1 + max(LIS[1], LIS[2], LIS[3]) = 1 + max(2, 1, 1) = 3
```

(Realising a Dynamic Programming Problem)

This problem has two important attributes that let us know it should be solved by dynamic programming:
1. The question is asking for the maximum or minimum of something.
2. We have to make decisions that may depend on previously made decisions, which is very typical of a problem involving subsequences.

As we go through the input, each "decision" we must make is simple: is it worth it to consider this number? If we use a number, it may contribute towards an increasing subsequence, but it may also eliminate larger elements that came before it.

For example, let's say we have `nums = [5, 6, 7, 8, 1, 2, 3]`. It isn't worth using the 1, 2, or 3, since using any of them would eliminate 5, 6, 7, and 8, which form the longest increasing subsequence.

(A Framework to Solve Dynamic Programming Problems)

Typically, dynamic programming problems can be solved with three main components.

First, we need some function or array that represents the answer to the problem from a given state. For this problem, let's say that we have an array `dp`. 

Let's say that `dp[i]` represents the length of the longest increasing subsequence that ends with the i$^{th}$ element.

The "state" is one-dimensional since it can be represented with only one variable - the index i.

Second, we need a way to transition between states, such as `dp[5]` and `dp[7]`, i.e. a recurrence relation.

Let's say we know `dp[0]`, `dp[1]`, and `dp[2]`. We need to find `dp[3]` given this information.

Since `dp[2]` represents the length of the longest increasing subsequence that ends with `nums[2]`, if `nums[3] > nums[2]`, then we can simply take the subsequence ending at i = 2 and append `nums[3]` to it, increasing the length by 1.

The same can be said for `nums[0]` and `nums[1]` if `nums[3]` is larger. We need to maximise `dp[3]`, so we need to check all 3.

Formally, the recurrence relation is: `dp[i] = max(dp[j] + 1) for all j where nums[j] < nums[i] and j < i`.

Third, we need a base case. For this problem, we can initialise every element of `dp` to 1, since every element on its own is technically an increasing subsequence.

```python
class Solution:
  def lengthOfLIS(self, nums: List[int]) -> int:
    """
    recurrence relation: dp[i] = max(dp[j] + 1) for all j where nums[j] < nums[i] and j < i
    
    O(N^2), O(N)
    """
    
    N = len(nums)
    dp = [1] * N
    
    for i in range(N):
      for j in range(i):
        if nums[i] > nums[j]:
          dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)
```

Intelligently build a subsequence

Consider the example `nums = [8, 1, 6, 2, 3, 10]`.

Let's try to build an increasing subsequence starting with an empty one: `sub = []`.

1. At the first element 8, we might as well take it since it's better than nothing, so `sub = [8]`.
2. At the second element 1, we can't increase the length of the subsequence since 8 >= 1, so we have to choose only one element to keep. Let's take the 1 since there may be elements later on that are greater than 1 but less than 8, now we have `sub = [1]`.
3. At the third element 6, we can build on our subsequence since 6 > 1, now `sub = [1, 6]`.
4. At the fourth element 2, we can't build on our subsequence since 6 >= 2, but can we improve on it for the future? Similar to the decision we made at the second element, if we replace the 6 with 2, we will open the door to using elements that are greater than 2 but less than 6 in the future, so `sub = [1, 2]`.
5. At the fifth element 3, we can build on our subsequence since 3 > 2. Notice that this was only possible because of the swap we made in the previous step, so `sub = [1, 2, 3]`.
6. At the last element 10, we can build on our subsequence since 10 > 3, giving a final subsequence `sub = [1, 2, 3, 10]`. The length of sub is our answer.

The best way to build an increasing subsequence is: for each element num, if num is greater than the largest element in our subsequence, then add it to the subsequence.

Otherwise, perform a linear scan through the subsequence starting from the smallest element and replace the first element that is greater than or equal to num with num. This opens the door for elements that are greater than num but less than the element replaced to be included in the sequence.

This algorithm does not always generate a valid subsequence of the input, but the length of the subsequence will always equal the length of the longest increasing subsequence. For example, with the input [3, 4, 5, 1], at the end we will have sub = [1, 4, 5], which isn't a subsequence, but the length is still correct. The length remains correct because the length only changes when a new element is larger than any element in the subsequence. In that case, the element is appended to the subsequence instead of replacing an existing element.

```python
class Solution:
  def lengthOfLIS(self, nums: List[int]) -> int:
    """
    O(N^2)
    Consider an input where the first half is [1, 2, 3, 4, ..., 99998, 99999], then the second half is [99998, 99998, 99998, ..., 99998, 99998].
    We would need to iterate (N/2)^2 times for the second half because there are N/2 elements equal to 99998, and a linear scan for each one takes N/2 iterations.
    
    O(N)
    """
    
    sub = [nums[0]]
    
    for num in nums[1:]:
      if num > sub[-1]:
        sub.append(num)
      else:
        # put num in place of the element in sub which is greater than num
        i = 0
        while num > sub[i]:
          i += 1
        sub[i] = num # sub[i] was holding an element greater than or equal to num, replace with num
    
    return len(sub)
```

Binary search

Instead of a linear scan to find the first element in sub that is greater than or equal to num, we can do a binary search, as the sub is in sorted order.

```python
class Solution:
  def bisectLeft(sub, num):
    l, r = 0, len(sub) - 1
    
    while l < r:
      m = l + (r - l) // 2
      
      if num == m:
        return m
      elif num < sub[m]:
        r = m - 1
      else:
        l = m + 1

    return l

  def lengthOfLIS(self, nums: List[int]) -> int:
    """
    O(NlogN), O(N)
    """
    
    sub = []

    for num in nums:
      i = bisectLeft(sub, num)
      if i == len(sub): # num is greater than any element in sub
        sub.append(num)
      else: # replace the first element in sub greater than or equal to num
        sub[i] = num
    
    return len(sub)
```

20. [Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)

> Given two strings `text1` and `text2`, return _the length of their longest **common subsequence**._ If there is no **common subsequence**, return `0`.
> 
> A **subsequence** of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.
> 
> - For example, `"ace"` is a subsequence of `"abcde"`.
> 
> A **common subsequence** of two strings is a subsequence that is common to both strings.
> 
> **Example 1:**
> 
> **Input:** text1 = "abcde", text2 = "ace" 
> **Output:** 3  
> **Explanation:** The longest common subsequence is "ace" and its length is 3.
> 
> **Example 2:**
> 
> **Input:** text1 = "abc", text2 = "abc"
> **Output:** 3
> **Explanation:** The longest common subsequence is "abc" and its length is 3.
> 
> **Example 3:**
> 
> **Input:** text1 = "abc", text2 = "def"
> **Output:** 0
> **Explanation:** There is no such common subsequence, so the result is 0.
> 
> **Constraints:**
> 
> - `1 <= text1.length, text2.length <= 1000`
> - `text1` and `text2` consist of only lowercase English characters.

2D DP

`dp[i][j]` represents the length of the LCS of the first `i` characters of `text1` and the first `j` characters of `text2`.

For each character pair `(text1[i-1], text2[j-1])` (subtracting one since our DP array is 1-indexed):
- If the characters match: `dp[i][j]=dp[i−1][j−1]+1`. This means we can extend the length of the LCS by 1.
- If the characters do not match: `dp[i][j]=max⁡(dp[i−1][j],dp[i][j−1])`. This means the LCS length is the maximum length found by either ignoring the current character of `text1` or `text2`.

Let's take `text1 = "abcde"` and `text2 = "ace"`.

| i   | j   | text1[i-1] | text2[j-1] | Action   | dp[i][j] | DP Table State                                                                         |
| --- | --- | ---------- | ---------- | -------- | -------- | -------------------------------------------------------------------------------------- |
| 1   | 1   | 'a'        | 'a'        | Match    | 1        | `[[0, 0, 0, 0], [0, 1, 0, 0], ...]`                                                    |
| 1   | 2   | 'a'        | 'c'        | No Match | 1        | `[[0, 0, 0, 0], [0, 1, 1, 0], ...]`                                                    |
| 1   | 3   | 'a'        | 'e'        | No Match | 1        | `[[0, 0, 0, 0], [0, 1, 1, 1], ...]`                                                    |
| 2   | 1   | 'b'        | 'a'        | No Match | 1        | `[[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 0, 0], ...]`                                      |
| 2   | 2   | 'b'        | 'c'        | No Match | 1        | `[[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 0], ...]`                                      |
| 2   | 3   | 'b'        | 'e'        | No Match | 1        | `[[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 1], ...]`                                      |
| 3   | 1   | 'c'        | 'a'        | No Match | 1        | `[[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 0, 0], ...]`                        |
| 3   | 2   | 'c'        | 'c'        | Match    | 2        | `[[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 2, 0], ...]`                        |
| 3   | 3   | 'c'        | 'e'        | No Match | 2        | `[[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 2, 2], ...]`                        |
| 4   | 1   | 'd'        | 'a'        | No Match | 1        | `[[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 2, 2], [0, 1, 0, 0], ...]`          |
| 4   | 2   | 'd'        | 'c'        | No Match | 2        | `[[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 2, 2], [0, 1, 2, 0], ...]`          |
| 4   | 3   | 'd'        | 'e'        | No Match | 2        | `[[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 2, 2], [0, 1, 2, 2], ...]`          |
| 5   | 1   | 'e'        | 'a'        | No Match | 1        | `[[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 2, 2], [0, 1, 2, 2], [0, 1, 0, 0]]` |
| 5   | 2   | 'e'        | 'c'        | No Match | 2        | `[[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 2, 2], [0, 1, 2, 2], [0, 1, 2, 0]]` |
| 5   | 3   | 'e'        | 'e'        | Match    | 3        | `[[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 2, 2], [0, 1, 2, 2], [0, 1, 2, 3]]` |

```python
class Solution:
  def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    """
    O(MN), O(MN)
    """
    
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
      for j in range(1, n + 1):
        if text1[i - 1] == text2[j - 1]:
          dp[i][j] = dp[i - 1][j - 1] + 1
        else:
          dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]
```

DP space optimised

The outer loop iterates through `text1`, and the inner loop iterates through `text2`. If characters match, we increment the value from `prev[j - 1]` (the diagonal). Else, we take the maximum from either ignoring the current character in `text1` or `text2`.

```python
class Solution:
  def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    """
    O(MN), O(min(M, N))
    """
    
    m, n = len(text1), len(text2)
    
    # Ensure we use the smaller array for optimization
    if m < n:
      text1, text2, m, n = text2, text1, n, m
    
    # Initialize previous and current arrays
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    
    # Fill the DP table row by row
    for i in range(1, m + 1):
      for j in range(1, n + 1):
        if text1[i - 1] == text2[j - 1]:
          curr[j] = prev[j - 1] + 1
        else:
          curr[j] = max(prev[j], curr[j - 1])
    
      # Swap references for the next iteration
      prev, curr = curr, prev
    
    return prev[n] # The result is in prev[n]
```

21. [Word Break Problem](https://leetcode.com/problems/word-break/)

> Given a string `s` and a dictionary of strings `wordDict`, return `true` if `s` can be segmented into a space-separated sequence of one or more dictionary words.
> 
> **Note** that the same word in the dictionary may be reused multiple times in the segmentation.
> 
> **Example 1:**
> 
> **Input:** s = "leetcode", wordDict = ["leet","code"]
> **Output:** true
> **Explanation:** Return true because "leetcode" can be segmented as "leet code".
> 
> **Example 2:**
> 
> **Input:** s = "applepenapple", wordDict = ["apple","pen"]
> **Output:** true
> **Explanation:** Return true because "applepenapple" can be segmented as "apple pen apple".
> Note that you are allowed to reuse a dictionary word.
> 
> **Example 3:**
> 
> **Input:** s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
> **Output:** false
> 
> **Constraints:**
> 
> - `1 <= s.length <= 300`
> - `1 <= wordDict.length <= 1000`
> - `1 <= wordDict[i].length <= 20`
> - `s` and `wordDict[i]` consist of only lowercase English letters.
> - All the strings of `wordDict` are **unique**.

DP

For each position `i` in the string, check all previous positions `j` (from 0 to `i-1`).

If `dp[j]` is `true` (meaning the substring `s[0:j]` can be segmented) and the substring `s[j:i]` is in the dictionary, then set `dp[i]` to `true`.

The value of `dp[len(s)]` will indicate if the whole string can be segmented.

```python
class Solution:
  def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    """
    O(N^2), O(N)
    """
    
    word_set = set(wordDict)  # Convert list to set for O(1) lookups
    dp = [False] * (len(s) + 1)
    dp[0] = True  # Base case: empty string
    
    for i in range(1, len(s) + 1):
      for j in range(i):
        # if string till j is breakable (d[j]) and the next substring till i (s[i:j]) is in wordDict, we can mark string till i (dp[i]) to be breakable
        if d[j] and s[j:i] in word_set: 
          dp[i] = True
          break
    
    return dp[N]
```

22. [Combination Sum](https://leetcode.com/problems/combination-sum-iv/)

> Given an array of **distinct** integers `candidates` and a target integer `target`, return _a list of all **unique combinations** of_ `candidates` _where the chosen numbers sum to_ `target`_._ You may return the combinations in **any order**.
> 
> The **same** number may be chosen from `candidates` an **unlimited number of times**. Two combinations are unique if the 
> 
> frequency
> 
>  of at least one of the chosen numbers is different.
> 
> The test cases are generated such that the number of unique combinations that sum up to `target` is less than `150` combinations for the given input.
> 
> **Example 1:**
> 
> **Input:** candidates = [2,3,6,7], target = 7
> **Output:** [[2,2,3],[7]]
> **Explanation:**
> 2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
> 7 is a candidate, and 7 = 7.
> These are the only two combinations.
> 
> **Example 2:**
> 
> **Input:** candidates = [2,3,5], target = 8
> **Output:** [[2,2,2,2],[2,3,3],[3,5]]
> 
> **Example 3:**
> 
> **Input:** candidates = [2], target = 1
> **Output:** []
> 
> **Constraints:**
> 
> - `1 <= candidates.length <= 30`
> - `2 <= candidates[i] <= 40`
> - All elements of `candidates` are **distinct**.
> - `1 <= target <= 40`

Backtracking

We use a recursive backtracking approach to explore all potential combinations.

Start from an empty combination and build it up by adding candidates. If the sum exceeds the target, backtrack. If the sum equals the target, store the combination.

DFS v/s Backtracking

DFS is primarily a traversal algorithm used to explore all nodes in a graph or tree structure. It can be used to find a path, search for a specific node, or explore all connected components.

Backtracking is a refinement of DFS used to solve constraint satisfaction problems, where we want to find all or some solutions to a problem by exploring potential candidates and abandoning them if they don't satisfy the constraints.

```python
class Solution:
  def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
    """
    N be the number of candidates, T be the target value, M be the target value
    
    O(N^(T/min(candidates))), O(T/M)
    """
    
    def backtrack(start, path, remaining):
      if remaining < 0:
        return
        
      if remaining == 0:
        result.append(path)
        return
      
      for i in range(start, len(candidates)):
        backtrack(i, path + [candidates[i]], remaining - candidates[i])
    
    result = []
    backtrack(0, [], target)
    
    return result
```

23. [House Robber](https://leetcode.com/problems/house-robber/)

> You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and **it will automatically contact the police if two adjacent houses were broken into on the same night**.
> 
> Given an integer array `nums` representing the amount of money of each house, return _the maximum amount of money you can rob tonight **without alerting the police**_.
> 
> **Example 1:**
> 
> **Input:** nums = [1,2,3,1]
> **Output:** 4
> **Explanation:** Rob house 1 (money = 1) and then rob house 3 (money = 3).
> Total amount you can rob = 1 + 3 = 4.
> 
> **Example 2:**
> 
> **Input:** nums = [2,7,9,3,1]
> **Output:** 12
> **Explanation:** Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
> Total amount you can rob = 2 + 9 + 1 = 12.
> 
> **Constraints:**
> 
> - `1 <= nums.length <= 100`
> - `0 <= nums[i] <= 400`

DP

State Transition

For each house `i`, the robber has two options:
1. Rob the current house: Add the money from the current house (`nums[i]`) to the maximum amount from `i-2` houses (`dp[i-2]`).
  2. Skip the current house: Take the maximum amount from the previous house (`dp[i-1]`).

Thus, the recurrence relation is: `dp[i]=max⁡(dp[i−1],nums[i]+dp[i−2])`

```python
class Solution:
  def rob(self, nums: List[int]) -> int:
    """
    O(N), O(N)
    """
    
    N = len(nums)
    if N == 1:
      return nums[0]
    
    dp = [0] * N
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    
    for i in range(2, N):
      dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    
    return dp[N-1]
```

Space optimisation

Since we only need the last two values at any time, we can replace the array with 2 variables.

```python
class Solution:
  def rob(self, nums: List[int]) -> int:
    """
    O(N), O(1)
    """
    
    N = len(nums)
    if N == 1:
      return nums[0]
    
    a, b = nums[0], max(nums[0], nums[1])
    
    for i in range(2, N):
      a, b = b, max(b, a + nums[i])
    
    return b
```

Recursive

```python
class Solution:
  def rob(self, nums: List[int]) -> int:
    """
    O(N), O(N)
    """
    
    @lru_cache(None)
    def robFrom(i):
      if i == 0:
        return nums[0]
      if i == 1:
        return max(nums[0], nums[1])
    
      return max(robFrom(i-1), robFrom(i-2) + nums[i])
    
    return robFrom(len(nums) - 1)
```

24. [House Robber II](https://leetcode.com/problems/house-robber-ii/)

> You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are **arranged in a circle.**That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have a security system connected, and **it will automatically contact the police if two adjacent houses were broken into on the same night**.
> 
> Given an integer array `nums` representing the amount of money of each house, return _the maximum amount of money you can rob tonight **without alerting the police**_.
> 
> **Example 1:**
> 
> **Input:** nums = [2,3,2]
> **Output:** 3
> **Explanation:** You cannot rob house 1 (money = 2) and then rob house 3 (money = 2), because they are adjacent houses.
> 
> **Example 2:**
> 
> **Input:** nums = [1,2,3,1]
> **Output:** 4
> **Explanation:** Rob house 1 (money = 1) and then rob house 3 (money = 3).
> Total amount you can rob = 1 + 3 = 4.
> 
> **Example 3:**
> 
> **Input:** nums = [1,2,3]
> **Output:** 3
> 
> **Constraints:**
> 
> - `1 <= nums.length <= 100`
> - `0 <= nums[i] <= 1000`

DP (built over the solution of House Robber I)

Since the houses are arranged in a circle, we can either:
1. Rob houses from index `0` to `n-2` (excluding the last house).
2. Rob houses from index `1` to `n-1` (excluding the first house).

The result will be the maximum of these two scenarios.

```python
class Solution:
  def rob_I(self, nums: List[int]) -> int:
    N = len(nums)
    if N == 1:
      return nums[0]
    
    a, b = nums[0], max(nums[0], nums[1])
    
    for i in range(2, N):
      a, b = b, max(b, a + nums[i])
    
    return b

  def rob(self, nums: List[int]) -> int:
    """
    O(N), O(1)
    """
    
    N = len(nums)
    if N == 1:
      return nums[0]
    
    return max(self.rob_I(nums[:-1]), self.rob_I(nums[1:]))
```

25. [Decode Ways](https://leetcode.com/problems/decode-ways/)

> You have intercepted a secret message encoded as a string of numbers. The message is **decoded** via the following mapping:
> 
> `"1" -> 'A'   "2" -> 'B'   ...   "25" -> 'Y'   "26" -> 'Z'`
> 
> However, while decoding the message, you realize that there are many different ways you can decode the message because some codes are contained in other codes (`"2"` and `"5"` vs `"25"`).
> 
> For example, `"11106"` can be decoded into:
> 
> - `"AAJF"` with the grouping `(1, 1, 10, 6)`
> - `"KJF"` with the grouping `(11, 10, 6)`
> - The grouping `(1, 11, 06)` is invalid because `"06"` is not a valid code (only `"6"`is valid).
> 
> Note: there may be strings that are impossible to decode.  
>   
> Given a string s containing only digits, return the **number of ways** to **decode** it. If the entire string cannot be decoded in any valid way, return `0`.
> 
> The test cases are generated so that the answer fits in a **32-bit** integer.
> 
> **Example 1:**
> 
> **Input:** s = "12"
> 
> **Output:** 2
> 
> **Explanation:**
> 
> "12" could be decoded as "AB" (1 2) or "L" (12).
> 
> **Example 2:**
> 
> **Input:** s = "226"
> 
> **Output:** 3
> 
> **Explanation:**
> 
> "226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).
> 
> **Example 3:**
> 
> **Input:** s = "06"
> 
> **Output:** 0
> 
> **Explanation:**
> 
> "06" cannot be mapped to "F" because of the leading zero ("6" is different from "06"). In this case, the string is not a valid encoding, so return 0.
> 
> **Constraints:**
> 
> - `1 <= s.length <= 100`
> - `s` contains only digits and may contain leading zero(s).

DP

State Transition

For each digit from index 2 to `n`, calculate `dp[i]` based on the following conditions:
1. If `s[i-1]` (current digit) is between '1' and '9', add `dp[i-1]` (ways to decode up to the previous digit).
2. If the two digits `s[i-2:i]` (previous and current digits) form a valid number between '10' and '26', add `dp[i-2]`(ways to decode up to two digits before).

```python
class Solution:
  def numDecodings(self, s: str) -> int:
    """
    O(N), O(N)
    """
    
    N = len(s)
    dp = [0] * (N + 1)
    dp[0] = 1 # empty string
    dp[1] = 1 if s[0] != '0' else 0 # valid if first character is not '0'
    
    for i in range(2, N + 1):
      # check single digit decode
      if s[i - 1] != '0':
        dp[i] += dp[i - 1]
      
      # check double digit decode
      if 10 <= int(s[i-2:i]) <= 26:
        dp[i] += dp[i - 2]
    
    return dp[N]
```

Why consider empty string case as `dp[0]`?

This design choice simplifies the implementation and reduces the need for additional checks in the loop.

Base Case

In many dynamic programming problems, especially those that build on previous results, it’s useful to define the number of ways to decode an empty substring as `1`. This means that if there's nothing to decode, there's exactly one way to interpret it: do nothing.

Uniform Indexing

- Defining `dp[0]` allows for a uniform way to handle the indices, where `dp[i]` directly corresponds to decoding the substring `s[0:i]`.
- Without `dp[0]`, we would need to handle the cases for the first character separately, which could complicate the logic.

Space optimisation for DP: we can use 2 variable approach

```python
class Solution:
  def numDecodings(self, s: str) -> int:
    """
    O(N), O(1)
    """
    
    N = len(s)
    a = 1
    b = 1 if s[0] != '0' else 0
    
    for i in range(2, N + 1):
      curr = 0
      
      # check single digit decode
      if s[i - 1] != '0':
        curr += b
      
      # check double digit decode
      if 10 <= int(s[i-2:i]) <= 26:
        curr += a
      
      a, b = b, curr
    
    return b
```

26. [Unique Paths](https://leetcode.com/problems/unique-paths/)

> There is a robot on an `m x n` grid. The robot is initially located at the **top-left corner**(i.e., `grid[0][0]`). The robot tries to move to the **bottom-right corner** (i.e., `grid[m - 1][n - 1]`). The robot can only move either down or right at any point in time.
> 
> Given the two integers `m` and `n`, return _the number of possible unique paths that the robot can take to reach the bottom-right corner_.
> 
> The test cases are generated so that the answer will be less than or equal to `2 * 109`.
> 
> **Example 1:**
> 
> ![](https://assets.leetcode.com/uploads/2018/10/22/robot_maze.png)
> 
> **Input:** m = 3, n = 7
> **Output:** 28
> 
> **Example 2:**
> 
> **Input:** m = 3, n = 2
> **Output:** 3
> **Explanation:** From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
> 1. Right -> Down -> Down
> 2. Down -> Down -> Right
> 3. Down -> Right -> Down
> 
> **Constraints:**
> 
> - `1 <= m, n <= 100`

Backtracking

```python
class Solution:
  def uniquePaths(self, m: int, n: int) -> int:
    """
    O(mn), O(mn)
    """
    
    @lru_cache(None)
    def backtrack(x, y):
      if x == m - 1 and y == n - 1:
        return 1
      
      if x >= m or y >= n:
        return 0
      
      return backtrack(x + 1, y) + backtrack(x, y + 1)
    
    return backtrack(0, 0)
```

DP

Base case

The number of ways to reach any cell in the first row or the first column is `1` since there's only one way to get there: either move right (for the first row) or move down (for the first column).

State transition

For each cell `(i, j)`, the number of unique paths to reach that cell is the sum of unique paths to reach the cell directly above it `(i-1, j)` and the cell directly to the left `(i, j-1)`:  `dp[i][j]=dp[i−1][j]+dp[i][j−1]`

```python
class Solution:
  def uniquePaths(self, m: int, n: int) -> int:
    """
    O(mn), O(mn)
    """
    
    dp = [[0] * n for _ in range(m)]
    
    for i in range(m):
      dp[i][0] = 1 # Only one way to reach any cell in the first column
    
    for j in range(n):
      dp[0][j] = 1 # Only one way to reach any cell in the first row
    
    for i in range(1, m):
      for j in range(1, n):
        dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    
    return dp[m - 1][n - 1]
```

Space optimisation

To optimise the space complexity, we can reduce the 2D DP array to just one 1D array since each row only depends on the current and previous rows.

```python
class Solution:
  def uniquePaths(self, m: int, n: int) -> int:
    """
    O(mn), O(n)
    """
    
    dp = [1] * n  # first row
    
    for i in range(1, m):
      for j in range(1, n):
        dp[j] += dp[j - 1]
    
    return dp[n - 1]
```

Combination formula: `C(k,r) = k! / (r!(k−r)!)​`

To travel from the top-left corner to the bottom-right corner of an `m x n` grid, we must make a total of `(m - 1)`downward moves and `(n - 1)` rightward moves.

The total number of moves is thus: `(m − 1) + (n − 1) = m + n − 2`

Out of these total moves, we need to choose `(m - 1)` moves to go down (or equivalently, `(n - 1)` moves to go right).

Thus, the number of unique paths will be `C(m + n − 2, m − 1) = (m + n - 2)! / ((m - 1)!(n - 1)!)`.

```python
from math import factorial

class Solution:
  def uniquePaths(self, m: int, n: int) -> int:
    """
    In python, k! can be computed in O(k(logkloglogk)^2), let's say that's k' complexity
    
    O(m' + n'), O(1)
    """
    
    return factorial(m - 1 + n - 1) // factorial(m - 1) // factorial(n - 1)
```

27. [Jump Game](https://leetcode.com/problems/jump-game/)

> You are given an integer array `nums`. You are initially positioned at the array's **first index**, and each element in the array represents your maximum jump length at that position.
> 
> Return `true` _if you can reach the last index, or_ `false` _otherwise_.
> 
> **Example 1:**
> 
> **Input:** nums = [2,3,1,1,4]
> **Output:** true
> **Explanation:** Jump 1 step from index 0 to 1, then 3 steps to the last index.
> 
> **Example 2:**
> 
> **Input:** nums = [3,2,1,0,4]
> **Output:** false
> **Explanation:** You will always arrive at index 3 no matter what. Its maximum jump length is 0, which makes it impossible to reach the last index.
> 
> **Constraints:**
> 
> - `1 <= nums.length <= 104`
> - `0 <= nums[i] <= 105`

DP

Iterate through the array, and for each index `i`, if `dp[i]` is `True`, check how far we can jump from that index. Update the subsequent indices that can be reached from `i` as `True`.

```python
class Solution:
  def canJump(self, nums: List[int]) -> bool:
    """
    O(N^2), O(N)
    """
    
    N = len(nums)
    dp = [False] * N
    dp[0] = True # Starting position
    
    for i in range(N):
      if dp[i]: # If the current position is reachable
        furthest_jump = min(N, nums[i] + 1)
        for j in range(1, furthest_jump):
          if i + j < N:
            # Mark reachable positions
            dp[i + j] = True 
            
            # Early exit if we can reach the last index
            if dp[-1]: 
              return True
    
    return dp[-1]
```

Greedy

The idea is to keep track of the farthest index we can reach while iterating through the array. If at any point our current index exceeds the farthest reachable index, it means we cannot proceed further.

```python
class Solution:
  def canJump(self, nums: List[int]) -> bool:
    """
    O(N), O(1)
    """
    
    N = len(nums)
    max_reachable = 0
    
    for i in range(N):
      if max_reachable < i:
        return False
      
      max_reachable = max(max_reachable, i + nums[i])
      
      if max_reachable >= N - 1:
        return True
```

---

## Graph

28. [Clone Graph](https://leetcode.com/problems/clone-graph/)

> Given a reference of a node in a **[connected](https://en.wikipedia.org/wiki/Connectivity_(graph_theory)#Connected_graph)** undirected graph.
> 
> Return a [**deep copy**](https://en.wikipedia.org/wiki/Object_copying#Deep_copy) (clone) of the graph.
> 
> Each node in the graph contains a value (`int`) and a list (`List[Node]`) of its neighbors.
> 
> **Test case format:**
> 
> For simplicity, each node's value is the same as the node's index (1-indexed). For example, the first node with `val == 1`, the second node with `val == 2`, and so on. The graph is represented in the test case using an adjacency list.
> 
> **An adjacency list** is a collection of unordered **lists** used to represent a finite graph. Each list describes the set of neighbors of a node in the graph.
> 
> The given node will always be the first node with `val = 1`. You must return the **copy of the given node** as a reference to the cloned graph.
> 
> **Example 1:**
> 
> ![](https://assets.leetcode.com/uploads/2019/11/04/133_clone_graph_question.png)
> 
> **Input:** adjList = [[2,4],[1,3],[2,4],[1,3]]
> **Output:** [[2,4],[1,3],[2,4],[1,3]]
> **Explanation:** There are 4 nodes in the graph.
> 1st node (val = 1)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
> 2nd node (val = 2)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
> 3rd node (val = 3)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
> 4th node (val = 4)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
> 
> **Example 2:**
> 
> ![](https://assets.leetcode.com/uploads/2020/01/07/graph.png)
> 
> **Input:** adjList = [[]]
> **Output:** [[]]
> **Explanation:** Note that the input contains one empty list. The graph consists of only one node with val = 1 and it does not have any neighbors.
> 
> **Example 3:**
> 
> **Input:** adjList = []
> **Output:** []
> **Explanation:** This an empty graph, it does not have any nodes.
> 
> **Constraints:**
> 
> - The number of nodes in the graph is in the range `[0, 100]`.
> - `1 <= Node.val <= 100`
> - `Node.val` is unique for each node.
> - There are no repeated edges and no self-loops in the graph.
> - The Graph is connected and all nodes can be visited starting from the given node.

We can use either Depth-First Search (DFS) or Breadth-First Search (BFS) to traverse the graph and clone each node and its neighbours.

A hash map will help to keep track of already cloned nodes to avoid infinite loops and duplicate copies.

```python
class Solution:
  visited = {}

  def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
    """
    O(V + E), O(V)
    """
    
    if not node:
      return None
    
    if node in self.visited:
      return self.visited[node]
    
    clone = Node(node.val)
    self.visited[node] = clone
    clone.neighbors = [self.cloneGraph(neighbor) for neighbor in node.neighbors]
    
    """
    shouldn't do:
    
    ```
    clone = Node(node.val, [self.cloneGraph(neighbor) for neighbor in node.neighbors])
    self.visited[node] = clone
    ```
    
    we need to add the cloned node to visited dict first before we go on to the neighbors, otherwise we'll reach max recursion depth upon encountering this node during the neighbor's cloning
    """
    
    return clone
```

29. [Course Schedule](https://leetcode.com/problems/course-schedule/)

> There are a total of `numCourses` courses you have to take, labeled from `0` to `numCourses - 1`. You are given an array `prerequisites` where `prerequisites[i] = [ai, bi]` indicates that you **must** take course `bi` first if you want to take course `ai`.
> 
> - For example, the pair `[0, 1]`, indicates that to take course `0` you have to first take course `1`.
> 
> Return `true` if you can finish all courses. Otherwise, return `false`.
> 
> **Example 1:**
> 
> **Input:** numCourses = 2, prerequisites = [[1,0]]
> **Output:** true
> **Explanation:** There are a total of 2 courses to take. 
> To take course 1 you should have finished course 0. So it is possible.
> 
> **Example 2:**
> 
> **Input:** numCourses = 2, prerequisites = [[1,0],[0,1]]
> **Output:** false
> **Explanation:** There are a total of 2 courses to take. 
> To take course 1 you should have finished course 0, and to take course 0 you should also have finished course 1. So it is impossible.
> 
> **Constraints:**
> 
> - `1 <= numCourses <= 2000`
> - `0 <= prerequisites.length <= 5000`
> - `prerequisites[i].length == 2`
> - `0 <= ai, bi < numCourses`
> - All the pairs prerequisites[i] are **unique**.

Graph Construction: Each prerequisite pair `[a, b]` indicates a directed edge from course `b` to course `a` (i.e., we must complete course `b`before course `a`).

Kahn’s Algorithm

This algorithm is used to perform a topological sort of the directed graph. It helps detect cycles in the graph. If a cycle is present, it means it's impossible to finish all courses.

Use a queue to perform a breadth-first search (BFS):
- Dequeue a course, add it to the count of processed courses, and reduce the in-degrees of its dependent courses.
- If a dependent course's in-degree drops to zero, enqueue it.

```python
class Solution:
  def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    O(V + E), O(V + E)
    """
    
    indegree = [0] * numCourses
    graph = defaultdict(set)
    
    for course, prereq in prerequisites:
      graph[prereq].add(course)  # prereq -> course
      indegree[course] += 1
    
    # Initialize the queue with courses having no prerequisites
    queue = []
    for course in range(numCourses):
      if indegree[course] == 0:
        queue.append(course)
    
    processed_nodes = 0
    while queue:
      course = queue.pop(0)
      processed_nodes += 1
      
      for next_course in graph[course]:
        indegree[next_course] -= 1  # Remove the prerequisite
        if indegree[next_course] == 0:
          queue.append(next_course)  # Add to queue if no more prerequisites
    
    return processed_nodes == numCourses
```

30. [Pacific Atlantic Water Flow](https://leetcode.com/problems/pacific-atlantic-water-flow/)

> There is an `m x n` rectangular island that borders both the **Pacific Ocean** and **Atlantic Ocean**. The **Pacific Ocean** touches the island's left and top edges, and the **Atlantic Ocean** touches the island's right and bottom edges.
> 
> The island is partitioned into a grid of square cells. You are given an `m x n` integer matrix `heights` where `heights[r][c]` represents the **height above sea level** of the cell at coordinate `(r, c)`.
> 
> The island receives a lot of rain, and the rain water can flow to neighboring cells directly north, south, east, and west if the neighboring cell's height is **less than or equal to** the current cell's height. Water can flow from any cell adjacent to an ocean into the ocean.
> 
> Return _a **2D list** of grid coordinates_ `result` _where_ `result[i] = [ri, ci]` _denotes that rain water can flow from cell_ `(ri, ci)` _to **both** the Pacific and Atlantic oceans_.
> 
> **Example 1:**
> 
> ![](https://assets.leetcode.com/uploads/2021/06/08/waterflow-grid.jpg)
> 
> **Input:** heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
> **Output:** [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
> **Explanation:** The following cells can flow to the Pacific and Atlantic oceans, as shown below:
```
[0,4]: [0,4] -> Pacific Ocean 
       [0,4] -> Atlantic Ocean
[1,3]: [1,3] -> [0,3] -> Pacific Ocean 
       [1,3] -> [1,4] -> Atlantic Ocean
[1,4]: [1,4] -> [1,3] -> [0,3] -> Pacific Ocean 
       [1,4] -> Atlantic Ocean
[2,2]: [2,2] -> [1,2] -> [0,2] -> Pacific Ocean 
       [2,2] -> [2,3] -> [2,4] -> Atlantic Ocean
[3,0]: [3,0] -> Pacific Ocean 
       [3,0] -> [4,0] -> Atlantic Ocean
[3,1]: [3,1] -> [3,0] -> Pacific Ocean 
       [3,1] -> [4,1] -> Atlantic Ocean
[4,0]: [4,0] -> Pacific Ocean 
       [4,0] -> Atlantic Ocean
```
> Note that there are other possible paths for these cells to flow to the Pacific and Atlantic oceans.
> 
> **Example 2:**
> 
> **Input:** heights = [[1]]
> **Output:** [[0,0]]
> **Explanation:** The water can flow from the only cell to the Pacific and Atlantic oceans.
> 
> **Constraints:**
> 
> - `m == heights.length`
> - `n == heights[r].length`
> - `1 <= m, n <= 200`
> - `0 <= heights[r][c] <= 105`

DFS from Both Oceans

The naive approach would be to check every cell - that is, iterate through every cell, and at each one, start a traversal that follows the problem's conditions. That is, find every cell that manages to reach both oceans.

This approach, however, is extremely slow, as it repeats a ton of computation. Instead of looking for every path from cell to ocean, let's start at the oceans and try to work our way to the cells. This will be much faster because when we start a traversal at a cell, whatever result we end up with can be applied to only that cell. However, when we start from the ocean and work backwards, we already know that every cell we visit must be connected to the ocean.

If we start traversing from the ocean and flip the condition (check for higher height instead of lower height), then we know that every cell we visit during the traversal can flow into that ocean. 

Let's start a traversal from every cell that is immediately beside the Pacific ocean, and figure out what cells can flow into the Pacific. Then, let's do the exact same thing with the Atlantic ocean. At the end, the cells that end up connected to both oceans will be our answer.

```python
class Solution:
  def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
    """
    O(mn), O(mn)
    """
    
    if not heights:
        return []
    
    m, n = len(heights), len(heights[0])
    
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    
    pacific_reachable, atlantic_reachable = set(), set()
    
    def dfs(x, y, reachable):
      reachable.add((x, y))
      for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < m and 0 <= ny < n and (nx, ny) not in reachable and heights[nx][ny] >= heights[x][y]:
          dfs(nx, ny, reachable)
    
    # Perform DFS from Pacific Ocean (top and left edges)
    for i in range(m):
      dfs(i, 0, pacific_reachable)  # Left edge
    for j in range(n):
      dfs(0, j, pacific_reachable)  # Top edge
    
    # Perform DFS from Atlantic Ocean (bottom and right edges)
    for i in range(m):
      dfs(i, n - 1, atlantic_reachable)  # Right edge
    for j in range(n):
      dfs(m - 1, j, atlantic_reachable)  # Bottom edge
    
    return list(pacific_reachable.intersection(atlantic_reachable))
```

31. [Number of Islands](https://leetcode.com/problems/number-of-islands/)

> Given an `m x n` 2D binary grid `grid` which represents a map of `'1'`s (land) and `'0'`s (water), return _the number of islands_.
> 
> An **island** is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.
> 
> **Example 1:**
> 
> **Input:** grid = [
>   ["1","1","1","1","0"],
>   ["1","1","0","1","0"],
>   ["1","1","0","0","0"],
>   ["0","0","0","0","0"]
> ]
> **Output:** 1
> 
> **Example 2:**
> 
> **Input:** grid = [
>   ["1","1","0","0","0"],
>   ["1","1","0","0","0"],
>   ["0","0","1","0","0"],
>   ["0","0","0","1","1"]
> ]
> **Output:** 3
> 
> **Constraints:**
> 
> - `m == grid.length`
> - `n == grid[i].length`
> - `1 <= m, n <= 300`
> - `grid[i][j]` is `'0'` or `'1'`. 

Go through each cell in the grid. When we encounter an unvisited `1`, we increment the island count and trigger a DFS/BFS to explore the entire island, marking all connected land (`1`s) as visited (changing them to `0` or by maintaining a visited set).

```python
class Solution:
  def numIslands(self, grid: List[List[str]]) -> int:
    """
    O(mn), O(mn)
    """
    
    m, n = len(grid), len(grid[0])
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    visited = set()
    
    def dfs(x, y):
      visited.add((x, y))
      for a, b in directions:
        nx, ny = x + a, y + b
        if 0 <= nx < m and 0 <= ny < n and (nx, ny) not in visited and grid[nx][ny] == "1":
            dfs(nx, ny)
    
    islands = 0
    for x in range(m):
      for y in range(n):
        if (x, y) not in visited and grid[x][y] == "1":
          dfs(x, y)
          islands += 1
    
    return islands
```

32. [Alien Dictionary](https://leetcode.com/problems/alien-dictionary/)

> There is a new alien language that uses the English alphabet. However, the order among the letters is unknown to you.
> 
> You are given a list of strings words from the alien language's dictionary, where the strings in words are sorted lexicographically by the rules of this new language.
> 
> Return a string of the unique letters in the new alien language sorted in lexicographically increasing order by the new language's rules. If there is no solution, return "". If there are multiple solutions, return any of them.
> 
> Example 1:
> 
> Input: words = ["wrt", "wrf","er","ett","rftt"]
> Output: "wertf"
> 
> Example 2:
> 
> Input: words = ["z", "x"]
> Output: "zx"
> 
> Example 3:
> 
> Input: words = ["z", "x", "z"]
> Output: ""
> 
> Explanation: The order is invalid, so return "".
> 
> Constraints:
> 
> - ﻿﻿`1 < words.length <= 100`
> - `﻿﻿1 < words[i].length <= 100`
> - ﻿﻿`words[i]` consists of only lowercase English letters

Graph Representation
- Represent the characters and their order as a directed graph where each character is a node.
- Directed edges between characters will indicate which character comes before another.

Building the Graph
- Compare each pair of adjacent words in the list to determine the order of characters.
- Create edges based on the first differing characters between words.
`
```
input = wxqkj whag cckgh cdxg cdxdt cdht ktgxt ktgch ktdw ktdc jqw jmc jmg

word1:      wxqkj whag  cckgh cdxg  cdxdt cdht  ktgxt ktgch ktdw ktdc jqw  jmc
word2:      whag  cckgh cdxg  cdxdt cdht  ktgxt ktgch ktdw  ktdc jqw  jmc  jmg
dependency: x->h  w->c  c->d  c->g  x->h  c->k  x->c  g->d  w->c k->j q->m c->g
```
 
Topological Sort
- Use Kahn’s algorithm (BFS) or Depth-First Search (DFS) to perform a topological sort on the graph. This will help detect cycles and find a valid ordering of characters.

```python
from collections import defaultdict, deque

class Solution:
    def alienOrder(self, words: List[str]) -> str:
        """
        O(V + E), O(V + E)
        """
        
        adj = defaultdict(set)
        indegree = {char: 0 for word in words for char in word} 
        
        # indegree = defaultdict(int) won't work, as we need to account for all characters
        # during queue construction, when we check indegree[x] == 0, x needs to be present in the indegree to be accounted for
        
        for word1, word2 in zip(words, words[1:]):
          # prefix edge case needs to be handled explicitly
            # if second word is prefix of the first, ex. [abcd, abc] -> return
            if word1.startswith(word2) and len(word1) > len(word2):
                return ""
                
            for char1, char2 in zip(word1, word2):
                if char1 != char2:
                    if char2 not in adj[char1]: # if not checked, we'll keep increasing the indegree for the same char again and again
                        adj[char1].add(char2)
                        indegree[char2] += 1
                    break
            
            # checking prefix post-loop will result in TLE
            # else:
            #     if len(word1) > len(word2):
            #         return ""
        
        
        queue = deque(char for char in indegree if indegree[char] == 0)
        
        order = []
        while queue:
            char = queue.popleft()
            order.append(char)
            
            for neighbour in adj[char]:
                indegree[neighbour] -= 1
                if indegree[neighbour] == 0:
                    queue.append(neighbour)
                    
        # no valid order, cycle
        if len(order) < len(indegree):
            return ""
        
        return "".join(order)
```

33. [Graph Valid Tree](https://leetcode.com/problems/graph-valid-tree/)

> You have a graph of n nodes labeled from `0` to `n - 1`. You are given an integer `n` and a list of edges where `edges[i] = [ai, bi]` indicates that there is an undirected edge between nodes `ai` and `bi` in the graph.
> 
> Return `true` if the edges of the given graph make up a valid tree, and false otherwise.
> 
> Example 1:
> 
> Input: n = 5, edges = [ [0,1], [0,2], [0,3], [1,4]]
> Output: true
> 
> Example 2:
> 
> Input: n = 5, edges = [[0,1], [1,2], [2,3], [1,3], [1,4]]
> Output: false
> 
> Constraints:
> 
> - ﻿﻿`1 < n <= 2000`
> - `﻿﻿0 < edges. length <= 5000`
> - ﻿﻿`edges[i]. length = 2`
> - ﻿﻿`0 <= ai, bi < n`
> - ﻿﻿`ai != bi`
> - ﻿﻿There are no self-loops or repeated edges.

G is a tree iff:
1. G is fully connected, i.e., for every pair of nodes in G, there is a path between them.
2. G contains no cycles, i.e., there is exactly one path between each pair of nodes in G.

For a graph to be a valid tree, it must have exactly n-1 edges. Any less, it can't be fully connected. Any more, it has to contain cycles. Additionally, if it is fully connected and contains exactly n-1 edges, it can't possibly contain a cycle, and therefore must be a tree.

Thus, the algorithm:
1. Check whether or not there are n-1 edges. If not, then return False.
2. Check whether or not the graph is fully connected. Return True if it is, false if otherwise.

```python
from collections import defaultdict

class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        """
        O(N), O(N)
        """
        
        if len(edges) != n-1:
            return False
        
        adj = defaultdict(set)
        for a, b in edges:
            adj[a].add(b)
            adj[b].add(a)
        
        
        visited = set()
        def dfs(node):
            visited.add(node)
            for neighbour in adj[node]:
                if neighbour not in visited:
                    dfs(neighbour)
        
        dfs(0)
        
        return len(visited) == n
```

Union Find

Same as before:
1. Fully connected: exactly n-1 edges.
2. No cycles: Each time there is no merge in our disjoint set DS, it's because we're adding an edge between 2 nodes that's already connected via path, indicating a cycle.

```
Ex 1: n = 6, edges = [(0, 5), (4, 0), (1, 2), (4, 5), (3, 2)]

{0} {1} {2} {3} {4} {5}

edge (0, 5) -> {0,5} {1} {2} {3} {4}

edge (4, 0) -> {0,5,4} {1} {2} {3}

edge (1, 2) -> {0,5,4} {1,2} {3}

edge (4, 5) -> {0,5,4} {1,2} {3}

edge (3, 2) -> {0,5,4} {1,2,3}

Edges are not in a single connected component, there must be a cycle.


Ex 2: n = 6, edges = [(0, 2), (4, 1), (2, 3), (3, 5), (1, 3)]  

{0} {1} {2} {3} {4} {5}

edge (0, 2) -> {0,2} {1} {3} {4} {5}

edge (4, 1) -> {0,2} {1,4} {3} {5}

edge (1, 2) -> {0,1,2,4} {3} {5}

edge (4, 5) -> {0,1,2,4,5} {3}

edge (3, 2) -> {0,1,2,3,4,5}

Edges are not in a single connected component, no cycle.
```

Complexity analysis ignores constants, sometimes they are still having a big impact on the run-time in practise.

The previous approach has a lot of overhead in needing to create an adjacent list with the edges before it could even begin the DFS. This is all treated as a constant, as it ultimately has the same time complexity as DFS.

This approach doesn't need to change the input format, it can just get straight to determining whether or not there is a cycle. Additionally, the bit that stops it being constant, α(N), will never have a value larger than 4. So in practise, it behaves as a constant too - and a far smaller one at that.

```python
class UnionFind:
    def __init__(self, size):
        self.root = [i for i in range(size)]
        self.rank = [1] * size
    
    def find(self, x):
        if self.root[x] != x:
            self.root[x] = self.find(self.root[x]) # Path compression
        
        return self.root[x]
    
    def union(self, x, y):
        rootX, rootY = self.find(x), self.find(y)
        if rootX == rootY:
            return False
        
        # Union by rank
        if self.rank[rootX] == self.rank[rootY]:
            self.root[rootY] = self.root[rootX]
            self.rank[rootX] += 1
        elif self.rank[rootX] > self.rank[rootY]:
            self.root[rootY] = self.root[rootX]
        else:
            self.root[rootX] = self.root[rootY]
            
        return True
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)


class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        """
        O(E⋅α(N)) = O(N⋅α(N))
        O(N)
        """
        
        if len(edges) != n-1:
            return False
        
        u = UnionFind(n)
        for a, b in edges:
            if not u.union(a, b):
                return False # Cycle detected
        
        return True
```

34. [Number of Connected Components in an Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/)

> You have a graph of n nodes. You are given an integer n and an array edges where `edges[i] = [ai, bi]` indicates that there is an edge between `ai` and `bi` in the graph.
> 
> Return the number of connected components in the graph.
> 
> Example 1:
> 
> Input: n = 5, edges = [ [0,1], [1,2], [3,4]]
> Output: 2
> 
> Example 2:
> 
> Input: n = 5, edges = [[0,1], [1,2], [2,3], [3,4]]
> Output: 1
> 
> Constraints:
> 
> - ﻿﻿`1 <= n <= 2000`
> - ﻿﻿`1 < edges. length <= 5000`
> - `﻿edges[i]. length = 2`
> - `0 < ai = bi <n`
> - `ai != bi`
> - ﻿﻿There are no repeated edges.

Union Find

Initially we'll have `n` components (disjoint nodes). For each successful union, we reduce that count by 1.

```python
class UnionFind:
    def __init__(self, size):
        self.root = [i for i in range(size)]
        self.rank = [1] * size
    
    def find(self, x):
        if self.root[x] != x:
            self.root[x] = self.find(self.root[x]) # Path compression
        
        return self.root[x]
    
    def union(self, x, y):
        rootX, rootY = self.find(x), self.find(y)
        if rootX == rootY:
            return False
        
        # Union by rank
        if self.rank[rootX] == self.rank[rootY]:
            self.root[rootY] = self.root[rootX]
            self.rank[rootX] += 1
        elif self.rank[rootX] > self.rank[rootY]:
            self.root[rootY] = self.root[rootX]
        else:
            self.root[rootX] = self.root[rootY]
            
        return True
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)


class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        """
        O(N + E.α(n)): O(N) to initialise the DSU arrays + iterating over edges, inverse Ackermann function O(α(n)) for each operation

        O(N)
        """
        
        u = UnionFind(n)
        for A, B in edges:
            if u.union(A, B):
                n -= 1
        
        return n
```

DFS

For each unvisited node, initiate a DFS. Each DFS initiation corresponds to discovering a new connected component.

```python
class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        """
        O(N+E), O(N+E)
        """
        
        adj_list = [[] for _ in range(n)]
        for A, B in edges:
            adj_list[A].append(B)
            adj_list[B].append(A)
        
        visited = set()
        def dfs(node):
            visited.add(node)
            for neighbour in adj_list[node]:
                if neighbour not in visited:
                    dfs(neighbour)
        
        count = 0
        for i in range(n):
            if i not in visited:
                dfs(i)
                count += 1
        
        return count
```

---

## Interval

35. [Insert Interval](https://leetcode.com/problems/insert-interval/)

> You are given an array of non-overlapping intervals `intervals` where `intervals[i] = [starti, endi]` represent the start and the end of the `ith` interval and `intervals` is sorted in ascending order by `starti`. You are also given an interval `newInterval = [start, end]` that represents the start and end of another interval.
> 
> Insert `newInterval` into `intervals` such that `intervals` is still sorted in ascending order by `starti` and `intervals` still does not have any overlapping intervals (merge overlapping intervals if necessary).
> 
> Return `intervals` _after the insertion_.
> 
> **Note** that you don't need to modify `intervals` in-place. You can make a new array and return it.
> 
> **Example 1:**
> 
> **Input:** intervals = [[1,3],[6,9]], newInterval = [2,5]
> **Output:** [[1,5],[6,9]]
> 
> **Example 2:**
> 
> **Input:** intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
> **Output:** [[1,2],[3,10],[12,16]]
> **Explanation:** Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].
> 
> **Constraints:**
> 
> - `0 <= intervals.length <= 104`
> - `intervals[i].length == 2`
> - `0 <= starti <= endi <= 105`
> - `intervals` is sorted by `starti` in **ascending** order.
> - `newInterval.length == 2`
> - `0 <= start <= end <= 105`

Linear Search

Non-overlapping Cases
- New interval starts after the current interval ends: add the current interval to `newIntervals`.
- Current interval starts after the new interval ends: add the new interval to the list, update `newInterval` to the current interval.

Overlapping cases: merge by taking min and max of the 2 ranges.

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        """
        O(N), O(N)
        """
        
        newIntervals = []
        for interval in intervals:
            a, b = interval
            c, d = newInterval
            if c > b:  # New interval is after the current interval
                newIntervals.append(interval)
            elif a > d:  # Current interval is after the new interval
                newIntervals.append(newInterval)
                newInterval = interval
            else:  # Overlapping intervals
                newInterval = [min(a, c), max(b, d)]

        newIntervals.append(newInterval)  # Append the last merged interval
        
        return newIntervals
```

Binary Search

The use of binary search can potentially reduce the number of comparisons.

1. Find Insertion Point: Use binary search to determine the position where the new interval can be inserted in the sorted list of existing intervals.
2. Prepare Result List: Create a new list to hold the result. Include all intervals that come before the insertion point.
3. Merge New Interval: Check if the last interval in the result list overlaps with the new interval:
    - If it overlaps, merge them by updating the end of the last interval.
    - If it does not overlap, simply add the new interval to the result list.
4. Process Remaining Intervals: Iterate through the remaining intervals (those after the insertion point):
    - If the last interval in the result list overlaps with the current interval, merge them.
    - If there is no overlap, add the current interval to the result list.

```python
import bisect

class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        """
        O(N), O(N)
        """
        
        # Step 1: Find the index to insert the new interval
        idx = bisect.bisect_left(intervals, newInterval)
        
        # Step 2: Prepare the result list with intervals before the new interval
        newIntervals = intervals[:idx]
        
        # Step 3: Add and merge the new interval
        if newIntervals and newIntervals[-1][1] >= newInterval[0]:
            # Merge with the last interval in newIntervals if it overlaps
            newIntervals[-1][1] = max(newIntervals[-1][1], newInterval[1])
        else:
            newIntervals.append(newInterval)
        
        # Step 4: Add all remaining intervals
        for i in range(idx, len(intervals)):
            if newIntervals[-1][1] >= intervals[i][0]:  # Overlapping case
                newIntervals[-1][1] = max(newIntervals[-1][1], intervals[i][1])
            else:
                newIntervals.append(intervals[i])
        
        return newIntervals
```

36. [Merge Intervals](https://leetcode.com/problems/merge-intervals/)

> Given an array of `intervals` where `intervals[i] = [starti, endi]`, merge all overlapping intervals, and return _an array of the non-overlapping intervals that cover all the intervals in the input_.
> 
> **Example 1:**
> 
> **Input:** intervals = [[1,3],[2,6],[8,10],[15,18]]
> **Output:** [[1,6],[8,10],[15,18]]
> **Explanation:** Since intervals [1,3] and [2,6] overlap, merge them into [1,6].
> 
> **Example 2:**
> 
> **Input:** intervals = [[1,4],[4,5]]
> **Output:** [[1,5]]
> **Explanation:** Intervals [1,4] and [4,5] are considered overlapping.
> 
> **Constraints:**
> 
> - `1 <= intervals.length <= 104`
> - `intervals[i].length == 2`
> - `0 <= starti <= endi <= 104`

Sorting + Merging

```python
from typing import List

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        O(NlogN)
        O(N): sort O(N), merged O(N)
        """
        
        # Sort intervals by their starting points
        intervals.sort(key=lambda x: x[0])
        
        merged = []
        for interval in intervals:
            if not merged: # First interval, just add it and skip to next
                merged.append(interval)
                continue
            
            prev_left, prev_right = merged[-1]
            curr_left, curr_right = interval
            
            # Check for overlap
            if prev_right >= curr_left:
                # Merge by updating the last interval
                merged[-1][1] = max(prev_right, curr_right)
            else:
                merged.append(interval)  # No overlap, add current interval
        
        return merged
```

37. [Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/)

> Given an array of intervals `intervals` where `intervals[i] = [starti, endi]`, return _the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping_.
> 
> **Note** that intervals which only touch at a point are **non-overlapping**. For example, `[1, 2]`and `[2, 3]` are non-overlapping.
> 
> **Example 1:**
> 
> **Input:** intervals = [[1,2],[2,3],[3,4],[1,3]]
> **Output:** 1
> **Explanation:** [1,3] can be removed and the rest of the intervals are non-overlapping.
> 
> **Example 2:**
> 
> **Input:** intervals = [[1,2],[1,2],[1,2]]
> **Output:** 2
> **Explanation:** You need to remove two [1,2] to make the rest of the intervals non-overlapping.
> 
> **Example 3:**
> 
> **Input:** intervals = [[1,2],[2,3]]
> **Output:** 0
> **Explanation:** You don't need to remove any of the intervals since they're already non-overlapping.
> 
> **Constraints:**
> 
> - `1 <= intervals.length <= 105`
> - `intervals[i].length == 2`
> - `-5 * 104 <= starti < endi <= 5 * 104`

Sorting based on starting times + Greedy selection

In case of an overlap, we keep the one with the earlier end time and discard the other, as it'll lead to more space to accommodate more intervals later on.

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        """
        O(NlogN)
        O(N): sort O(N)
        """
        
        # Sort the intervals by their end times
        intervals.sort(key=lambda x: x[0])
        
        count = 0
        prev_end = float('-inf')
        
        for interval in intervals:
            if prev_end > interval[0]:
                count += 1  # Increment count for overlapping interval
                # Update prev_end to the minimum end time of the overlapping intervals
                prev_end = min(prev_end, interval[1])
            else:
                prev_end = interval[1]  # Update the prev end time to the current interval's end
        
        return count
```

Sorting based on end times + Greedy selection

This is a more common greedy strategy, as it helps in selecting the interval that finishes the earliest, maximising the chance to accommodate subsequent intervals.

If an overlap occurs, we always drop the current interval, as that decision will lead to more space to accommodate more intervals.

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        """
        O(NlogN)
        O(N): sort O(N)
        """
        
        # Sort the intervals by their end times
        intervals.sort(key=lambda x: x[1])
        
        count = 0
        prev_end = float('-inf')
        
        for interval in intervals:
            if prev_end > interval[0]:
                count += 1  # Increment count for overlapping interval
            else:
                prev_end = interval[1]  # Update the prev end time to the current interval's end
        
        return count
```

38. [Meeting Rooms](https://leetcode.com/problems/meeting-rooms/)

> Given an array of meeting time intervals where intervals[i] = [starti, endi], determine if a person could attend all meetings.
> 
> Example 1:
> 
> Input: intervals = [ [0,30], [5,10], [15,20]]
> Output: false
> 
> Example 2:
> 
> Input: intervals = [[7,10], [2,4]]
> Output: true
> 
> Constraints:
> 
> - ﻿﻿0 <= intervals. length <= 10^4
> - ﻿﻿intervals [i].length = 2
> - ﻿﻿0 < starti < endi == 10^6

Sorting + checking for overlaps

```python
class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        """
        O(NlogN), O(N)
        """
        
        intervals.sort()
        
        for prev, curr in zip(intervals, intervals[1:]):
            if prev[1] > curr[0]: # if prev ends before the next one starts
                return False
        
        return True
```

39. [Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/)

> Given an array of meeting time intervals intervals where intervals[i] = [starti, endi], return the minimum number of conference rooms required.
> 
> Example 1:
> 
> Input: intervals = [[0,30], [5,10], [15,20]]
> Output: 2
> 
> Example 2:
> 
> Input: intervals = [ [7,10], [2,4]]
> Output: 1
> 
> Constraints:
> 
> - ﻿﻿1 < intervals. length <= 10^4
> - ﻿﻿0 < starti < endi <= 10^6

A meeting is defined by its start and end times. However, for this specific algorithm, we need to treat the start and end times individually. When we encounter an ending event, that means that some meeting that started earlier has ended now. We are not really concerned with which meeting has ended, all we need is that some meeting ended thus making a room available.

When a meeting starts before the last one ends (`start < ends[end_pointer]`), it indicates that a new room is needed. Thus, we increment `room_count`.

Else, it means that the current meeting can reuse a room that is now free. There’s no need to adjust (decrease) `room_count` because we are not currently holding onto that room; we are just checking for overlapping.

```python
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        """
        O(NlogN), O(N)
        """
        
        # Separate start and end times
        starts = sorted([i[0] for i in intervals])  # Start times
        ends = sorted([i[1] for i in intervals])    # End times
        
        room_count = 0
        end_pointer = 0
        
        for start in starts:
            # If a meeting starts before the last one ends, we need a new room
            if start < ends[end_pointer]:
                room_count += 1
            else:
                # Reuse the room: move the end pointer to the next meeting
                end_pointer += 1
        
        return room_count  # Return the minimum number of rooms required
```

---

## Linked List

40. [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)

> Given the `head` of a singly linked list, reverse the list, and return _the reversed list_.
> 
> **Example 1:**
> 
> ![](https://assets.leetcode.com/uploads/2021/02/19/rev1ex1.jpg)
> 
> **Input:** head = [1,2,3,4,5]
> **Output:** [5,4,3,2,1]
> 
> **Example 2:**
> 
> ![](https://assets.leetcode.com/uploads/2021/02/19/rev1ex2.jpg)
> 
> **Input:** head = [1,2]
> **Output:** [2,1]
> 
> **Example 3:**
> 
> **Input:** head = []
> **Output:** []
> 
> **Constraints:**
> 
> - The number of nodes in the list is the range `[0, 5000]`.
> - `-5000 <= Node.val <= 5000`
> 
> **Follow up:** A linked list can be reversed either iteratively or recursively. Could you implement both?

Iterative

We can reverse a linked list by iteratively changing the direction of the pointers. For each node, reverse the pointer to point to the previous node, move to the next node and repeat until all nodes are processed.

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        """
        O(N), O(1)
        """
        
        prev = None  # Initialize the previous pointer
        
        # Iterate through the linked list
        while head:
            next_node = head.next  # Store the next node
            head.next = prev  # Reverse the link
            prev, head = head, next_node  # Move prev to current, current to next node
            
        return prev  # Return the new head of the reversed list
```

Recursive

Recursive Call: The function calls itself with the next node (`head.next`) to process the rest of the list.

Rearranging Pointers: After the recursive call, the `next` pointer of the next node is updated to point back to the current node (`head`), and the current node's `next` pointer is set to `None`.

```python
head.next.next = head  # Make the next node point to the current 
head head.next = None  # Set current head's next to None
```

`head` refers to the current node in the recursion. `head.next` refers to the next node in the original order of the list.

- `head.next.next = head`:
    - This line is executed after the recursive call has returned.
    - The recursive call has already reversed the rest of the list starting from `head.next`.
    - By accessing `head.next.next`, we are effectively reaching the node that was originally right after `head`.
    - This means we are telling that node (the one after `head`) to point back to `head`, reversing the link between these two nodes.
    - For example, if the list was originally `1 -> 2 -> 3`, and we are currently at node `2`, after the recursive call, `head.next` (which is `3`) will have its `next` pointer set to point back to `2`.
- `head.next = None`:
    - This line is crucial for preventing cycles in the linked list.
    - Since `head.next` originally pointed to the next node in the list, setting it to `None` indicates that `head` is now the last node in the reversed list.
    - In our example, when we're at node `2`, after pointing `3` to `2`, we set `2.next` to `None` to indicate that `2` is the last node in the new reversed list.

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        """
        O(N)
        O(N): stack
        """
        
        # Base case: if head is None or only one node
        if not head or not head.next:
            return head
        
        # Recursive case: reverse the rest of the list
        new_head = self.reverseList(head.next)
        
        # Rearranging pointers
        head.next.next = head  # Make the next node point to the current head
        head.next = None  # Set current head's next to None
        
        return new_head  # Return the new head of the reversed list
```

41. [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)

> Given `head`, the head of a linked list, determine if the linked list has a cycle in it.
> 
> There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the `next` pointer. Internally, `pos` is used to denote the index of the node that tail's `next` pointer is connected to. **Note that `pos` is not passed as a parameter**.
> 
> Return `true` _if there is a cycle in the linked list_. Otherwise, return `false`.
> 
> **Example 1:**
> 
> ![](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist.png)
> 
> **Input:** head = [3,2,0,-4], pos = 1
> **Output:** true
> **Explanation:** There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).
> 
> **Example 2:**
> 
> ![](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist_test2.png)
> 
> **Input:** head = [1,2], pos = 0
> **Output:** true
> **Explanation:** There is a cycle in the linked list, where the tail connects to the 0th node.
> 
> **Example 3:**
> 
> ![](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist_test3.png)
> 
> **Input:** head = [1], pos = -1
> **Output:** false
> **Explanation:** There is no cycle in the linked list.
> 
> **Constraints:**
> 
> - The number of the nodes in the list is in the range `[0, 104]`.
> - `-105 <= Node.val <= 105`
> - `pos` is `-1` or a **valid index** in the linked-list.
> 
> **Follow up:** Can you solve it using `O(1)` (i.e. constant) memory?

The most efficient way to detect a cycle in a linked list is to use the Floyd's Tortoise and Hare algorithm, which involves using two pointers that move at different speeds.

If there's no cycle in the list, the fast pointer will eventually reach the end and return false.

```python
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        """
        O(N), O(1)
        """
        
        slow = fast = head
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
            if slow == fast:
                return True
        
        return False
```

42. [Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)

> You are given the heads of two sorted linked lists `list1` and `list2`.
> 
> Merge the two lists into one **sorted** list. The list should be made by splicing together the nodes of the first two lists.
> 
> Return _the head of the merged linked list_.
> 
> **Example 1:**
> 
> ![](https://assets.leetcode.com/uploads/2020/10/03/merge_ex1.jpg)
> 
> **Input:** list1 = [1,2,4], list2 = [1,3,4]
> **Output:** [1,1,2,3,4,4]
> 
> **Example 2:**
> 
> **Input:** list1 = [], list2 = []
> **Output:** []
> 
> **Example 3:**
> 
> **Input:** list1 = [], list2 = [0]
> **Output:** [0]
> 
> **Constraints:**
> 
> - The number of nodes in both lists is in the range `[0, 50]`.
> - `-100 <= Node.val <= 100`
> - Both `list1` and `list2` are sorted in **non-decreasing** order.

We merge the two lists by comparing the values of the nodes one by one and building a new sorted linked list.

```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        """
        O(N + M), O(1)
        """
        
        # Early exits
        if not list1:
            return list2
        if not list2:
            return list1
        
        dummy = head = ListNode()  # sentinel head
        
        while list1 and list2:
            if list1.val < list2.val:
                head.next = list1
                list1 = list1.next
            else:
                head.next = list2
                list2 = list2.next
            head = head.next
        
        # Attach any remaining nodes
        head.next = list1 if list1 else list2
        
        # Return the merged list, starting from the node after dummy
        return dummy.next
```

43. [Merge K Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)

> You are given an array of `k` linked-lists `lists`, each linked-list is sorted in ascending order.
> 
> _Merge all the linked-lists into one sorted linked-list and return it._
> 
> **Example 1:**
> 
> **Input:** lists = [[1,4,5],[1,3,4],[2,6]]
> **Output:** [1,1,2,3,4,4,5,6]
> **Explanation:** The linked-lists are:
> [
>   1->4->5,
>   1->3->4,
>   2->6
> ]
> merging them into one sorted list:
> 1->1->2->3->4->4->5->6
> 
> **Example 2:**
> 
> **Input:** lists = []
> **Output:** []
> 
> **Example 3:**
> 
> **Input:** lists = [[]]
> **Output:** []
> 
> **Constraints:**
> 
> - `k == lists.length`
> - `0 <= k <= 104`
> - `0 <= lists[i].length <= 500`
> - `-104 <= lists[i][j] <= 104`
> - `lists[i]` is sorted in **ascending order**.
> - The sum of `lists[i].length` will not exceed `104`.

Merge with Divide and Conquer

```
[ L1, L2, L3, L4, L5, L6, L7, L8 ]

[ L1, L2, L3, L4 ]   [ L5, L6, L7, L8 ]

[ L1, L2 ]   [ L3, L4 ]   [ L5, L6 ]   [ L7, L8 ]

[ L1 ]   [ L2 ]   [ L3 ]   [ L4 ]   [ L5 ]   [ L6 ]   [ L7 ]   [ L8 ]
```

```
merge(L1, L2) -> M1
merge(L3, L4) -> M2
merge(L5, L6) -> M3
merge(L7, L8) -> M4

merge(M1, M2) -> M5
merge(M3, M4) -> M6

merge(M5, M6) -> Final Merged List
```

```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        # Early exits
        if not list1:
            return list2
        if not list2:
            return list1
        
        dummy = head = ListNode()  # sentinel head
        
        while list1 and list2:
            if list1.val < list2.val:
                head.next = list1
                list1 = list1.next
            else:
                head.next = list2
                list2 = list2.next
            head = head.next
        
        # Attach any remaining nodes
        head.next = list1 if list1 else list2
        
        # Return the merge
        return dummy.next
    
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        k: number of linked lists
        N: total number of nodes
        
        O(NlogK): Merge two sorted lists in O(N) time, logk times
        O(logK): recursion stack
        """
        
        N = len(lists)
        
        if N == 0:
            return None  # No lists to merge
        if N == 1:
            return lists[0]  # Only one list to return
        
        # Divide and conquer: merge the lists in halves
        mid = N // 2
        left_merged = self.mergeKLists(lists[:mid])
        right_merged = self.mergeKLists(lists[mid:])
        
        return self.mergeTwoLists(left_merged, right_merged)
```

Min heap

We use a min-heap (or priority queue) to keep track of the smallest current nodes from each list. 

Continuously extract the smallest node from the heap and add it to the merged list. After extracting a node, push the next node from the same list into the heap.

This approach also runs in O(N log k) time, but it may be faster in practice for larger inputs due to fewer overall comparisons. It can efficiently manage cases where the number of lists varies significantly.

It requires O(k) space for the heap, which may be higher than the recursive stack space in the divide-and-conquer approach.

```python
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        k: number of linked lists
        N: total number of nodes
        
        O(Nlogk): comparison cost is logk for every pop and insertion, N nodes
        O(k): atmost k nodes in the heap
        """
        
        min_heap = []
        
        # Push the head of each list into the heap
        for i, node in enumerate(lists):
            if node:  # Need to check for None node
                heapq.heappush(min_heap, (node.val, i, node))
        
        dummy = head = ListNode()
        
        # Process the heap until it's empty
        while min_heap:
            val, index, node = heapq.heappop(min_heap)  # Get the smallest node
            current.next = node  # Link it to the merged list
            current = current.next  # Move the current pointer
            
            if node.next:  # If there's a next node, push it into the heap
                heapq.heappush(min_heap, (node.next.val, index, node.next))
        
        # Return the merged list, starting from the node after dummy
        return dummy.next
```

44. [Remove Nth Node From End Of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)

> Given the `head` of a linked list, remove the `nth` node from the end of the list and return its head.
> 
> **Example 1:**
> 
> ![](https://assets.leetcode.com/uploads/2020/10/03/remove_ex1.jpg)
> 
> **Input:** head = [1,2,3,4,5], n = 2
> **Output:** [1,2,3,5]
> 
> **Example 2:**
> 
> **Input:** head = [1], n = 1
> **Output:** []
> 
> **Example 3:**
> 
> **Input:** head = [1,2], n = 1
> **Output:** [1]
> 
> **Constraints:**
> 
> - The number of nodes in the list is `sz`.
> - `1 <= sz <= 30`
> - `0 <= Node.val <= 100`
> - `1 <= n <= sz`
> 
> **Follow up:** Could you do this in one pass?

We can use a two-pointer technique. The idea is to place one pointer `n` nodes ahead of the other. When the first pointer reaches the end of the list, the second pointer will be at the node just before the one we want to remove.

```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """
        O(N), O(1)
        """
        
        # A dummy node is created and points to the head. This simplifies edge case handling (e.g., removing the head).
        dummy = first = second = ListNode(0, head)
        
        # Advance the first pointer n+1 steps ahead
        for _ in range(n + 1):
            first = first.next
        
        # Move both pointers until the first pointer reaches the end
        while first:
            first = first.next
            second = second.next
        
        # The second pointer will now point to the node before the one we need to remove. Adjust the pointers to remove the n-th node.
        second.next = second.next.next
        
        return dummy.next
```

45. [Reorder List](https://leetcode.com/problems/reorder-list/)

> You are given the head of a singly linked-list. The list can be represented as:
> 
> L0 → L1 → … → Ln - 1 → Ln
> 
> _Reorder the list to be on the following form:_
> 
> L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
> 
> You may not modify the values in the list's nodes. Only nodes themselves may be changed.
> 
> **Example 1:**
> 
> ![](https://assets.leetcode.com/uploads/2021/03/04/reorder1linked-list.jpg)
> 
> **Input:** head = [1,2,3,4]
> **Output:** [1,4,2,3]
> 
> **Example 2:**
> 
> ![](https://assets.leetcode.com/uploads/2021/03/09/reorder2-linked-list.jpg)
> 
> **Input:** head = [1,2,3,4,5]
> **Output:** [1,5,2,4,3]
> 
> **Constraints:**
> 
> - The number of nodes in the list is in the range `[1, 5 * 104]`.
> - `1 <= Node.val <= 1000`

To get the desired pattern:
1. Find the Middle of the List: Use the fast and slow pointer technique to find the middle node of the linked list.
2. Reverse the Second Half: Reverse the second half of the linked list starting from the middle node.
3. Merge Two Halves: Merge the first half and the reversed second half together.

```python
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        O(N), O(1)
        """
        
        if not head or not head.next:
            return
        
        # Step 1: Find the middle of the list
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        # Step 2: Reverse the second half of the list
        prev, curr = None, slow
        while curr:
            next_node = curr.next
            curr.next = prev
            prev, curr = curr, next_node
        
        # Step 3: Merge the two halves
        # Merge all nodes from the reversed second half until it is fully integrated into the first half, maintaining the required order without leaving any nodes unmerged.
        first, second = head, prev  # second is the head of the reversed list
        while second.next:  
            tmp1, tmp2 = first.next, second.next
            first.next = second
            second.next = tmp1
            first, second = tmp1, tmp2
```

---

## Matrix

46. [Set Matrix Zeroes](https://leetcode.com/problems/set-matrix-zeroes/)

> Given an `m x n` integer matrix `matrix`, if an element is `0`, set its entire row and column to `0`'s.
> 
> You must do it [in place](https://en.wikipedia.org/wiki/In-place_algorithm).
> 
> **Example 1:**
> 
> ![](https://assets.leetcode.com/uploads/2020/08/17/mat1.jpg)
> 
> **Input:** matrix = [[1,1,1],[1,0,1],[1,1,1]]
> **Output:** [[1,0,1],[0,0,0],[1,0,1]]
> 
> **Example 2:**
> 
> ![](https://assets.leetcode.com/uploads/2020/08/17/mat2.jpg)
> 
> **Input:** matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
> **Output:** [[0,0,0,0],[0,4,5,0],[0,3,1,0]]
> 
> **Constraints:**
> 
> - `m == matrix.length`
> - `n == matrix[0].length`
> - `1 <= m, n <= 200`
> - `-231 <= matrix[i][j] <= 231 - 1`
> 
> **Follow up:**
> 
> - A straightforward solution using `O(mn)` space is probably a bad idea.
> - A simple improvement uses `O(m + n)` space, but still not the best solution.
> - Could you devise a constant space solution?

1. Identify Rows and Columns to Zero:
    - Use the first row and first column to keep track of which rows and columns should be zeroed.
    - Iterate through the matrix and mark the first cell of the respective row and column if a zero is found.
2. Zero Out Rows and Columns:
    - Based on the markings in the first row and column, update the respective rows and columns to zero.
3. Handle the First Row and First Column Separately:
    - Finally, zero out the first row and/or first column if necessary.

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        O(M*N), O(1)
        """
        
        if not matrix:
            return
        
        rows, cols = len(matrix), len(matrix[0])
        zero_first_row = False
        zero_first_col = False
        
        # Determine if the first row and first column should be zeroed
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == 0:
                    if i == 0:
                        zero_first_row = True
                    if j == 0:
                        zero_first_col = True
                    matrix[i][0] = 0
                    matrix[0][j] = 0
        
        # Zero out the cells based on the markings
        for i in range(1, rows):
            for j in range(1, cols):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        
        # Zero out the first row and first column if needed
        if zero_first_row:
            for j in range(cols):
                matrix[0][j] = 0
        
        if zero_first_col:
            for i in range(rows):
                matrix[i][0] = 0
```

47. [Spiral Matrix](https://leetcode.com/problems/spiral-matrix/)

> Given an `m x n` `matrix`, return _all elements of the_ `matrix` _in spiral order_.
> 
> **Example 1:**
> 
> ![](https://assets.leetcode.com/uploads/2020/11/13/spiral1.jpg)
> 
> **Input:** matrix = [[1,2,3],[4,5,6],[7,8,9]]
> **Output:** [1,2,3,6,9,8,7,4,5]
> 
> **Example 2:**
> 
> ![](https://assets.leetcode.com/uploads/2020/11/13/spiral.jpg)
> 
> **Input:** matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
> **Output:** [1,2,3,4,8,12,11,10,9,5,6,7]
> 
> **Constraints:**
> 
> - `m == matrix.length`
> - `n == matrix[i].length`
> - `1 <= m, n <= 10`
> - `-100 <= matrix[i][j] <= 100`

The outer while loop runs as long as there are rows and columns left to traverse.

The for loops handle the traversal in the respective directions, updating the result list and the boundaries after each complete traversal.

Conditions to Avoid Redundant Traversal: Before traversing the bottom row and the left column, checks ensure that the traversal is valid.

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        """
        O(M*N), O(1)/O(M*N)
        """
        
        # Check if the matrix is empty
        if not matrix or not matrix[0]:
            return []
        
        result = []
        top, bottom = 0, len(matrix) - 1  # Initialize top and bottom boundaries
        left, right = 0, len(matrix[0]) - 1  # Initialize left and right boundaries

        # Continue until all boundaries have been processed
        while top <= bottom and left <= right:
            # Traverse from left to right along the top row
            for j in range(left, right + 1):
                result.append(matrix[top][j])
            top += 1  # Move the top boundary down
            
            # Traverse from top to bottom along the right column
            for i in range(top, bottom + 1):
                result.append(matrix[i][right])
            right -= 1  # Move the right boundary left
            
            if top <= bottom:  # Check if there's still a bottom row
                # Traverse from right to left along the bottom row
                for j in reversed(range(left, right + 1)):
                    result.append(matrix[bottom][j])
                bottom -= 1  # Move the bottom boundary up
            
            if left <= right:  # Check if there's still a left column
                # Traverse from bottom to top along the left column
                for i in reversed(range(top, bottom + 1)):
                    result.append(matrix[i][left])
                left += 1  # Move the left boundary right
        
        return result
```

Alternative implementation

The `direction` variable alternates between `1` (moving right or down) and `-1` (moving left or up). The direction is flipped after each complete traversal to change from horizontal to vertical movement.

We first move along the current row (`cols` iterations) and then along the current column (`rows` iterations). After traversing a full row or column, we decreases the respective count of rows or columns remaining.

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        """
        O(M*N), O(1)/O(M*N)
        """
        
        # Check if the matrix is empty
        if not matrix or not matrix[0]:
            return []
        
        rows, cols = len(matrix), len(matrix[0])
        
        direction = 1  # Start by moving right
        i, j = 0, -1  # Start at the first row and just before the first column
        
        result = []
        while rows > 0 and cols > 0:
            # Traverse the current row: iterate over cols
            for _ in range(cols):
                j += direction  # Move right (1) or left (-1)
                result.append(matrix[i][j])
            rows -= 1
            
            # Traverse the current column: iterate over rows
            for _ in range(rows):
                i += direction  # Move down (1) or up (-1)
                result.append(matrix[i][j])
            cols -= 1
            
            direction *= -1  # Switch direction for the next row/column traversal
        
        return result
```

48. [Rotate Image](https://leetcode.com/problems/rotate-image/)

> You are given an `n x n` 2D `matrix` representing an image, rotate the image by **90**degrees (clockwise).
> 
> You have to rotate the image [**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm), which means you have to modify the input 2D matrix directly. **DO NOT** allocate another 2D matrix and do the rotation.
> 
> **Example 1:**
> 
> ![](https://assets.leetcode.com/uploads/2020/08/28/mat1.jpg)
> 
> **Input:** matrix = [[1,2,3],[4,5,6],[7,8,9]]
> **Output:** [[7,4,1],[8,5,2],[9,6,3]]
> 
> **Example 2:**
> 
> ![](https://assets.leetcode.com/uploads/2020/08/28/mat2.jpg)
> 
> **Input:** matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
> **Output:** [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
> 
> **Constraints:**
> 
> - `n == matrix.length == matrix[i].length`
> - `1 <= n <= 20`
> - `-1000 <= matrix[i][j] <= 1000`

Matrix Algebra
1. Transpose: Convert rows to columns and columns to rows. This is the first step in rotating the matrix.
2. Reflect: After transposing, reverse each row to achieve the final rotated result.

```
1 2 3      1 4 7      7 4 1
4 5 6  ->  2 5 8  ->  8 5 2
7 8 9      3 6 9      9 6 3
```

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        
        O(N^2), O(1)
        """
        
        n = len(matrix)
        
        # Step 1: Transpose the matrix
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        
        # Step 2: Reverse each row
        for i in range(n):
            matrix[i].reverse()
```

49. [Word Search](https://leetcode.com/problems/word-search/)

> Given an `m x n` grid of characters `board` and a string `word`, return `true` _if_ `word` _exists in the grid_.
> 
> The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.
> 
> **Example 1:**
> 
> ![](https://assets.leetcode.com/uploads/2020/11/04/word2.jpg)
> 
> **Input:** board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
> **Output:** true
> 
> **Example 2:**
> 
> ![](https://assets.leetcode.com/uploads/2020/11/04/word-1.jpg)
> 
> **Input:** board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
> **Output:** true
> 
> **Example 3:**
> 
> ![](https://assets.leetcode.com/uploads/2020/10/15/word3.jpg)
> 
> **Input:** board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
> **Output:** false
> 
> **Constraints:**
> 
> - `m == board.length`
> - `n = board[i].length`
> - `1 <= m, n <= 6`
> - `1 <= word.length <= 15`
> - `board` and `word` consists of only lowercase and uppercase English letters.
> 
> **Follow up:** Could you use search pruning to make your solution faster with a larger `board`?

Why the need for visit?

1. Preventing Reuse of Cells

In the problem, we are allowed to move to adjacent cells to form the word, but we cannot use the same cell more than once in a single search path. If we don't mark a cell as visited, the algorithm could mistakenly revisit it, leading to incorrect matches.

2. Correct Backtracking

When exploring potential paths (especially with recursive backtracking), marking a cell as visited allows us to maintain a clear state of which cells have been explored in the current path. After all possible paths from a cell have been explored, restoring its original value ensures that other search paths can still use that cell if needed.

Time complexity

For the backtracking function, initially we could have at most 4 directions to explore, but further the choices are reduced into 3 (since we won't go back to where we come from).

As a result, the execution trace after the first step could be visualised as a 3-nary tree, each of the branches represent a potential exploration in the corresponding direction. Therefore, in the worst case, the total number of invocation would be the number of nodes in a full 3-nary tree, which is about 3^L.

We iterate through the board for backtracking, i.e. there could be number of nodes times invocation for the backtracking function in the worst case.

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        """
        O(M*N*3^L): M*N times 3^L invocations
        O(L): recursion stack
        """
        
        directions = ((0, 1), (0, -1), (1, 0), (-1, 0))
        m, n = len(board), len(board[0])
        l = len(word)
        
        def backtrack(i, j, k):
            # If we have matched all characters of the word
            if k == l:
                return True
            
            # Check if current position is out of bounds or if the character does not match
            # We need this boundary check post word index check, else we'll not get accurate results
            # Ex: board = [["a"]], word = "a", if we do a boundary check before, it'll never reach the function call to mark k = 1
            if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != word[k]:
                return False
            
            # Mark the cell as visited
            temp = board[i][j]
            board[i][j] = '#'  # Temporary marker
            
            # Explore all four directions
            for x, y in directions:
                new_i, new_j = i + x, j + y
                if backtrack(new_i, new_j, k + 1):
                    return True
            
            # Restore the cell value after exploring
            board[i][j] = temp
            
            return False
        
        # Start backtracking from each cell
        for i in range(m):
            for j in range(n):
                if backtrack(i, j, 0):  # Start backtracking with k = 0
                    return True
        
        return False  # If no match is found
```

---

### String

50. [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

> Given a string `s`, find the length of the **longest substring** without repeating characters.
> 
> **Example 1:**
> 
> **Input:** s = "abcabcbb"
> **Output:** 3
> **Explanation:** The answer is "abc", with the length of 3.
> 
> **Example 2:**
> 
> **Input:** s = "bbbbb"
> **Output:** 1
> **Explanation:** The answer is "b", with the length of 1.
> 
> **Example 3:**
> 
> **Input:** s = "pwwkew"
> **Output:** 3
> **Explanation:** The answer is "wke", with the length of 3.
> Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
> 
> **Constraints:**
> 
> - `0 <= s.length <= 5 * 104`
> - `s` consists of English letters, digits, symbols and spaces.

Sliding window with 2 pointers

As we expand the `right` pointer, we check if the character is already in the `char_set`. If a repeating character is found, we slide the `left` pointer to the right until the character can be added to the set. We calculate the max length after each iteration.

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        """
        O(n), O(min(n, c))
        """
        
        char_set = set()
        left = 0
        max_length = 0
        
        for right in range(len(s)):
            # If the character is already in the set, slide the window from the left
            while s[right] in char_set:
                char_set.remove(s[left])
                left += 1
            char_set.add(s[right])  # Add the current character to the set
            max_length = max(max_length, right - left + 1)  # Update the max length
        
        return max_length
```

Optimisation

The above solution uses at most 2n steps (outer + inner loop). We can reduce it to require at most n steps by using a dictionary by mapping the index as well, then skip the characters immediately when we find a repeated character.

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        """
        O(n), O(min(n, c))
        """
        
        l = 0
        curr_chars = {}
        max_len = 0
        
        for r in range(len(s)):
            if s[r] in curr_chars:
                next_idx = curr_chars[s[r]] + 1
                l = max(l, next_idx)
            curr_chars[s[r]] = r
            max_len = max(max_len, r - l + 1)
        
        return max_len
```

51. [Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/)

> You are given a string `s` and an integer `k`. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most `k` times.
> 
> Return _the length of the longest substring containing the same letter you can get after performing the above operations_.
> 
> **Example 1:**
> 
> **Input:** s = "ABAB", k = 2
> **Output:** 4
> **Explanation:** Replace the two 'A's with two 'B's or vice versa.
> 
> **Example 2:**
> 
> **Input:** s = "AABABBA", k = 1
> **Output:** 4
> **Explanation:** Replace the one 'A' in the middle with 'B' and form "AABBBBA".
> The substring "BBBB" has the longest repeating letters, which is 4.
> There may exists other ways to achieve this answer too.
> 
> **Constraints:**
> 
> - `1 <= s.length <= 105`
> - `s` consists of only uppercase English letters.
> - `0 <= k <= s.length`

Sliding Window

Keep track of the frequency of characters in the current window and maintain the count of the most frequent character.

Window Adjustment: If the condition `(right - left + 1) - max_count > k` holds true, it means we cannot form a valid substring with at most `k` replacements, so we shrink the window from the left.

```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        """
        O(N), O(c) = O(1)
        """
        
        left = 0
        count = defaultdict(int)
        max_count = 0
        max_length = 0
        window_length = lambda: right - left + 1
        
        for right in range(len(s)):
            count[s[right]] += 1
            max_count = max(max_count, count[s[right]])
            
            # If the current window size minus the max count is greater than k,
            # it means we need to shrink the window
            while window_length() - max_count > k:
                count[s[left]] -= 1  # Remove the leftmost character from the count
                left += 1  # Slide the window from the left
            
            max_length = max(max_length, window_length())
        
        return max_length
```

52. [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)

> Given two strings `s` and `t` of lengths `m` and `n` respectively, return _the **minimum window substring_** of `s` _such that every character in_ `t` _(**including duplicates**) is included in the window_. If there is no such substring, return _the empty string_ `""`.
> 
> The testcases will be generated such that the answer is **unique**.
> 
> **Example 1:**
> 
> **Input:** s = "ADOBECODEBANC", t = "ABC"
> **Output:** "BANC"
> **Explanation:** The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.
> 
> **Example 2:**
> 
> **Input:** s = "a", t = "a"
> **Output:** "a"
> **Explanation:** The entire string s is the minimum window.
> 
> **Example 3:**
> 
> **Input:** s = "a", t = "aa"
> **Output:** ""
> **Explanation:** Both 'a's from t must be included in the window.
> Since the largest window of s only has one 'a', return empty string.
> 
> **Constraints:**
> 
> - `m == s.length`
> - `n == t.length`
> - `1 <= m, n <= 105`
> - `s` and `t` consist of uppercase and lowercase English letters.
> 
> **Follow up:** Could you find an algorithm that runs in `O(m + n)` time?

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        """
        O(|S| + |T|): worst case we might end up visiting every element of S twice.
        O(|S| + |T|): |S| when window size is equal to the entire string S. |T| when T has all unique characters.
        """
        
        if not t or not s:
            return ""
        
        t_counter = Counter(t)  # Frequency of characters in t
        counter = defaultdict(int)  # Frequency of characters in the current window
        
        match_count = 0  # Number of characters that match t
        required_length = len(t_counter)  # Unique characters in t
        
        min_length = float('inf')
        min_substring = ""
        
        # Lambda to calculate the current window length
        window_length = lambda: right - left + 1
        
        left = 0
        for right in range(len(s)):
            counter[s[right]] += 1
            
            # Check if the character is part of t (explicitly not required in counters) and if we have enough of it
            if counter[s[right]] == t_counter[s[right]]:
                match_count += 1
            
            # Try to contract the window until it ceases to be valid
            while left <= right and match_count == required_length:
                # Update the minimum window if the current window is smaller
                if window_length() < min_length:
                    min_length = window_length()
                    min_substring = s[left:right + 1]
            
                # Remove the leftmost character from the window, check if that changes the number of matches
                counter[s[left]] -= 1
                if counter[s[left]] < t_counter[s[left]]:
                    match_count -= 1
                
                left += 1  # Update pointer
        
        return min_substring
```

A small improvement to the approach can reduce the time complexity of the algorithm to 0(2 * |filtered S| +IS| + |T|), where `filtered_S` is the string formed from S by removing all the elements not present in T.

This complexity reduction is evident when `|filtered_S| <<< |S|`. 

This kind of scenario might happen when length of string T is way too small than the length of string S and string S consists of numerous characters which are not present in T.

```
S = "ABCDDDDDDEEAFFBC" T = "ABC"

filtered_s = [(O, 'A'), (1, 'B'), (2, 'C'), (11, 'A'), (14, 'B'), (15, 'C')]
```

53. [Valid Anagram](https://leetcode.com/problems/valid-anagram/)

> Given two strings `s` and `t`, return `true` if `t` is an anagram of `s`, and `false` otherwise.
> 
> **Example 1:**
> 
> **Input:** s = "anagram", t = "nagaram"
> 
> **Output:** true
> 
> **Example 2:**
> 
> **Input:** s = "rat", t = "car"
> 
> **Output:** false
> 
> **Constraints:**
> 
> - `1 <= s.length, t.length <= 5 * 104`
> - `s` and `t` consist of lowercase English letters.
> 
> **Follow up:** What if the inputs contain Unicode characters? How would you adapt your solution to such a case?

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        """
        O(|S| + |T|), O(c) = O(1)
        """
        
        # Anagrams must be of the same length
        if len(s) != len(t):
            return False
            
        return Counter(s) == Counter(t)  # Compare the two count dictionaries 
```

54. [Group Anagrams](https://leetcode.com/problems/group-anagrams/)

> Given an array of strings `strs`, group the anagrams together. You can return the answer in **any order**.
> 
> **Example 1:**
> 
> **Input:** strs = ["eat","tea","tan","ate","nat","bat"]
> 
> **Output:** [["bat"],["nat","tan"],["ate","eat","tea"]]
> 
> **Explanation:**
> 
> - There is no string in strs that can be rearranged to form `"bat"`.
> - The strings `"nat"` and `"tan"` are anagrams as they can be rearranged to form each other.
> - The strings `"ate"`, `"eat"`, and `"tea"` are anagrams as they can be rearranged to form each other.
> 
> **Example 2:**
> 
> **Input:** strs = [""]
> 
> **Output:** [[""]]
> 
> **Example 3:**
> 
> **Input:** strs = ["a"]
> 
> **Output:** [["a"]]
> 
> **Constraints:**
> 
> - `1 <= strs.length <= 104`
> - `0 <= strs[i].length <= 100`
> - `strs[i]` consists of lowercase English letters.

Use a dictionary to group the strings based on their character count.

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        """
        N is the number of strings, K is the maximum length of string
        
        O(N * K), O(N * K)
        """
        
        anagrams = defaultdict(list)
        
        for s in strs:
            # Create a character count (for 26 lowercase letters)
            count = [0] * 26  # Since we assume the input contains only lowercase letters
            for char in s:
                count[ord(char) - ord('a')] += 1
            
            # Convert the count list to a tuple to use as a key
            key = tuple(count)
            anagrams[key].append(s)  # Group the original string with its anagram
        
        return list(anagrams.values())
```

Product of primes

Assign a unique prime number to each letter and multiply them to create a unique product for each word. Since the product of primes is unique for the combination of letters, this can be used to identify character counts.

```
multiples_dict in func generate_primes()

{}
{4: [2]}
{4: [2], 9: [3]}
{9: [3], 6: [2]}
{9: [3], 6: [2], 25: [5]}
{9: [3], 25: [5], 8: [2]}
{9: [3], 25: [5], 8: [2], 49: [7]}
{9: [3], 25: [5], 49: [7], 10: [2]}
```

```python
class Solution:
    @staticmethod
    def generate_primes():
        multiples_dict = defaultdict(list)  # Dictionary to hold multiples of primes
        curr_num = 2  # Start checking for primes from 2
        
        while True:
            if curr_num not in multiples_dict:
                yield curr_num  # Yield the current prime
                multiples_dict[curr_num * curr_num].append(curr_num)  # Mark the square of the prime
            else:
                # Mark the multiples of the current prime
                for prime in multiples_dict[curr_num]:
                    multiples_dict[prime + curr_num].append(prime)
                del multiples_dict[curr_num]  # Remove the entry for current_number as it has been processed
            
            curr_num += 1  # Move to the next integer
    
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        """
        N is the number of strings, K is the maximum length of string
        
        O(N * K), O(N)
        """
        
        anagrams = defaultdict(list)
        
        primes = self.generate_primes()  # Prime number generator
        prime_map = {chr(i + ord('a')): next(primes) for i in range(26)}  # Map each letter to a prime
        
        for s in strs:
        # Calculate product of primes for the string
            product_of_primes = 1
            for char in s:
                product_of_primes *= prime_map[char]
            
            anagrams[product_of_primes].append(s)  # Group strings by their product of primes
        
        return list(anagrams.values())
```

55. [Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)

> Given a string `s` containing just the characters `'('`, `')'`, `'{'`, `'}'`, `'['` and `']'`, determine if the input string is valid.
> 
> An input string is valid if:
> 
> 1. Open brackets must be closed by the same type of brackets.
> 2. Open brackets must be closed in the correct order.
> 3. Every close bracket has a corresponding open bracket of the same type.
> 
> **Example 1:**
> 
> **Input:** s = "()"
> 
> **Output:** true
> 
> **Example 2:**
> 
> **Input:** s = "()[]{}"
> 
> **Output:** true
> 
> **Example 3:**
> 
> **Input:** s = "(]"
> 
> **Output:** false
> 
> **Example 4:**
> 
> **Input:** s = "([])"
> 
> **Output:** true
> 
> **Constraints:**
> 
> - `1 <= s.length <= 104`
> - `s` consists of parentheses only `'()[]{}'`.

We use a dictionary to map closing brackets to their corresponding opening brackets.

We use a stack to keep track of opening brackets. When we encounter a closing bracket, we check if it matches the top of the stack. 

In the end, we need to check if all brackets are matched (i.e., the stack is empty).

```python
class Solution:
    def isValid(self, s: str) -> bool:
        """
        O(N), O(N)
        """
        
        bracket_map = {')': '(', '}': '{', ']': '['}
        stack = []

        for char in s:
            if char in bracket_map:  # If it's a closing bracket
                top_element = stack.pop() if stack else '#'
                if bracket_map[char] != top_element:
                    return False
            else:  # If it's an opening bracket
                stack.append(char)
        
        return not stack  # Return True if stack is empty (all brackets matched)
```

56. [Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)

> A phrase is a **palindrome** if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.
> 
> Given a string `s`, return `true` _if it is a **palindrome**, or_ `false` _otherwise_.
> 
> **Example 1:**
> 
> **Input:** s = "A man, a plan, a canal: Panama"
> **Output:** true
> **Explanation:** "amanaplanacanalpanama" is a palindrome.
> 
> **Example 2:**
> 
> **Input:** s = "race a car"
> **Output:** false
> **Explanation:** "raceacar" is not a palindrome.
> 
> **Example 3:**
> 
> **Input:** s = " "
> **Output:** true
> **Explanation:** s is an empty string "" after removing non-alphanumeric characters.
> Since an empty string reads the same forward and backward, it is a palindrome.
> 
> **Constraints:**
> 
> - `1 <= s.length <= 2 * 105`
> - `s` consists only of printable ASCII characters.

Two pointers

Compare characters from both ends, skipping non-alphanumeric characters and ignoring case sensitivity, moving inward until the pointers meet.

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        """
        O(N), O(1)
        """
        
        l, r = 0, len(s) - 1
        
        while l < r:
            # Move left pointer to the right if the character is not alphanumeric
            while l < r and not s[l].isalnum():
                l += 1
            
            # Move right pointer to the left if the character is not alphanumeric
            while l < r and not s[r].isalnum():
                r -= 1

            # Compare characters at both pointers, ignoring case
            if s[l].lower() != s[r].lower():
                return False
            
            # Move both pointers inward
            l += 1
            r -= 1
        
        return True  # If all characters matched, it is a palindrome
```

57. [Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)

> Given a string `s`, return _the longest palindromic substring_ in `s`.
> 
> **Example 1:**
> 
> **Input:** s = "babad"
> **Output:** "bab"
> **Explanation:** "aba" is also a valid answer.
> 
> **Example 2:**
> 
> **Input:** s = "cbbd"
> **Output:** "bb"
> 
> **Constraints:**
> 
> - `1 <= s.length <= 1000`
> - `s` consist of only digits and English letters.

Expand Around Center

For each character (and each pair of consecutive characters), expand outward to check for palindromes. This accounts for both odd and even length palindromes.

Maintain variables to track the start and end indices of the longest palindrome found during the process.

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        """
        O(N^2)
        n centers for odd-length + (n - 1) centers for even-length
        = 2n - 1 centers = O(n)
        
        For each center, expand costs O(n), although practically it'll be much less as most centers will not produce long palindromes.
        
        O(1)
        """
        
        start, end = 0, 0
        
        for i in range(len(s)):
            # Check for odd-length palindromes
            left1, right1 = self.expandAroundCenter(s, i, i)
            # Check for even-length palindromes
            left2, right2 = self.expandAroundCenter(s, i, i + 1)
            
            # Update start and end for the longer palindrome found
            if right1 - left1 > end - start:
                start, end = left1, right1
            if right2 - left2 > end - start:
                start, end = left2, right2
        
        return s[start:end + 1]
    
    def expandAroundCenter(self, s: str, left: int, right: int) -> tuple:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        
        return left + 1, right - 1  # left is decremented and right is incremented one extra time before exiting the loop
```

58. [Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/)

> Given a string `s`, return _the number of **palindromic substrings** in it_.
> 
> A string is a **palindrome** when it reads the same backward as forward.
> 
> A **substring** is a contiguous sequence of characters within the string.
> 
> **Example 1:**
> 
> **Input:** s = "abc"
> **Output:** 3
> **Explanation:** Three palindromic strings: "a", "b", "c".
> 
> **Example 2:**
> 
> **Input:** s = "aaa"
> **Output:** 6
> **Explanation:** Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
> 
> **Constraints:**
> 
> - `1 <= s.length <= 1000`
> - `s` consists of lowercase English letters.

Expand Around Center

For each character (for odd-length palindromes) and each pair of consecutive characters (for even-length palindromes), expand outward to count palindromic substrings. Increment a count each time a valid palindrome is found.

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        """
        O(N^2), O(1)
        """
        
        count = 0
        
        for i in range(len(s)):
            # Count odd-length palindromes
            count += self.expandAroundCenter(s, i, i)
            # Count even-length palindromes
            count += self.expandAroundCenter(s, i, i + 1)
        
        return count

    def expandAroundCenter(self, s: str, left: int, right: int) -> int:
        count = 0
        while left >= 0 and right < len(s) and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
        
        return count
```

59. [Encode and Decode Strings](https://leetcode.com/problems/encode-and-decode-strings/)

> Design an algorithm to encode a list of strings to a string. The encoded string is then sent over the network and is decoded back to the original list of strings.
> 
> Machine 1 (sender) has the function:
> 
> string encode(vector\<string> strs) {
>     // ... your code
>     return encoded_string;
> }
> 
> Machine 2 (receiver) has the function:
> 
> vector\<string> decode(string s) {
>     //... your code
>     return strs;
> }
> 
> So Machine 1 does:
> 
> string encoded_string = encode(strs);
> 
> and Machine 2 does:
> 
> vector\<string> strs2 = decode (encoded_string);
> 
> `strs2` in Machine 2 should be the same as `strs` in Machine 1.
> 
> Implement the `encode` and `decode` methods.
> 
> You are not allowed to solve the problem using any serialize methods (such as `eval`).
> 
> Example 1:
> 
> Input: dummy_input = ["Hello", "World" ]
> Output: ["Hello", "World" ]
> Explanation:
> 
> Machine 1:
> Codec encoder = new Codec() ;
> String msg = encoder.encode(strs) ;
> Machine 1 --msg--> Machine 2
> 
> Machine 2:
> Codec decoder = new Codec() ;
> String[] strs = decoder. decode(msg) ;
> 
> Example 2:
> 
> Input: dummy_input = [''']
> Output: ['''']
> 
> Constraints:
> - ﻿﻿1< strs. length <= 200
> - ﻿﻿0 <= strs[i]. length <= 200
> - ﻿﻿strs [i] contains any possible characters out of 256 valid ASCII characters.
> 
> Follow up: Could you write a generalized algorithm to work on any possible set of characters?

Chunked Transfer Encoding

Prefix each string with its length as a fixed-width string (e.g., 4 characters), which allows for easy parsing without using delimiters.

```python
class Codec:
    def encode(self, strs: List[str]) -> str:
        """
        O(N), O(N)
        """
        
        encoded_chunks = []
        for s in strs:
            # Convert length to a fixed-width string (e.g., 4 characters)
            length_prefix = f"{len(s):04d}"
            # Encode each string with its length and a delimiter
            encoded_chunks.append(f"{length_prefix}{s}")
        
        return ''.join(encoded_chunks)

    def decode(self, s: str) -> List[str]:
        """
        O(N), O(N)
        """
        
        result = []
        i = 0
        
        while i < len(s):
            # Read the fixed-width length prefix (4 characters)
            length_prefix = s[i:i + 4]
            length = int(length_prefix)
            i += 4  # Move past the length prefix
            
            # Extract the string of the specified length
            result.append(s[i:i + length])
            i += length  # Move to the next chunk
        
        return result
```

---

### Tree

60. [Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)

> Given the `root` of a binary tree, return _its maximum depth_.
> 
> A binary tree's **maximum depth** is the number of nodes along the longest path from the root node down to the farthest leaf node.
> 
> **Example 1:**
> 
> ![](https://assets.leetcode.com/uploads/2020/11/26/tmp-tree.jpg)
> 
> **Input:** root = [3,9,20,null,null,15,7]
> **Output:** 3
> 
> **Example 2:**
> 
> **Input:** root = [1,null,2]
> **Output:** 2
> 
> **Constraints:**
> 
> - The number of nodes in the tree is in the range `[0, 104]`.
> - `-100 <= Node.val <= 100`

DFS

```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        """
        O(N), O(H)
        """
        
        if not root:
            return 0  # Base case: empty tree has depth 0
        
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
```

BFS

```python
from collections import deque

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        """
        O(N), O(W)
        """
        
        if not root:
            return 0  # Base case: empty tree has depth 0
        
        queue = deque([root])
        depth = 0
        
        while queue:
            depth += 1  # Increase depth for each level
            for _ in range(len(queue)):  # Process all nodes at the current level
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        
        return depth
```

61. [Same Tree](https://leetcode.com/problems/same-tree/)

> Given the roots of two binary trees `p` and `q`, write a function to check if they are the same or not.
> 
> Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.
> 
> **Example 1:**
> 
> ![](https://assets.leetcode.com/uploads/2020/12/20/ex1.jpg)
> 
> **Input:** p = [1,2,3], q = [1,2,3]
> **Output:** true
> 
> **Example 2:**
> 
> ![](https://assets.leetcode.com/uploads/2020/12/20/ex2.jpg)
> 
> **Input:** p = [1,2], q = [1,null,2]
> **Output:** false
> 
> **Example 3:**
> 
> ![](https://assets.leetcode.com/uploads/2020/12/20/ex3.jpg)
> 
> **Input:** p = [1,2,1], q = [1,1,2]
> **Output:** false
> 
> **Constraints:**
> 
> - The number of nodes in both trees is in the range `[0, 100]`.
> - `-104 <= Node.val <= 104`

DFS

```python
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        """
        O(N), O(H)
        """
        
        if not p and not q:
            return True
        
        if not p or not q or (p.val != q.val):
            return False
        
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
```

BFS

```python
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        """
        O(N), O(W)
        """
        
        queue = deque([[p, q]])
        while queue:
            a, b = queue.popleft()
            
            if not a and not b:
                continue
            
            if not a or not b or a.val != b.val:
                return False
            
            queue.append([a.left, b.left])
            queue.append([a.right, b.right])
        
        return True
```

62. [Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)

> Given the `root` of a binary tree, invert the tree, and return _its root_.
> 
> **Example 1:**
> 
> ![](https://assets.leetcode.com/uploads/2021/03/14/invert1-tree.jpg)
> 
> **Input:** root = [4,2,7,1,3,6,9]
> **Output:** [4,7,2,9,6,3,1]
> 
> **Example 2:**
> 
> ![](https://assets.leetcode.com/uploads/2021/03/14/invert2-tree.jpg)
> 
> **Input:** root = [2,1,3]
> **Output:** [2,3,1]
> 
> **Example 3:**
> 
> **Input:** root = []
> **Output:** []
> 
> **Constraints:**
> 
> - The number of nodes in the tree is in the range `[0, 100]`.
> - `-100 <= Node.val <= 100`

DFS

```python
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        O(N), O(H)
        """
        
        if not root:
            return
        
        # Swap the left and right children
        root.left, root.right = root.right, root.left
        
        self.invertTree(root.left)
        self.invertTree(root.right)
        
        return root
```

BFS

```python
from collections import deque

class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        """
        O(N), O(W)
        """
        
        if not root:
            return None
        
        queue = deque([root])
        
        while queue:
            node = queue.popleft()
            # Swap the left and right children
            node.left, node.right = node.right, node.left
            
            # Add children to the queue
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        return root
```

63. [Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)

> A **path** in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence **at most once**. Note that the path does not need to pass through the root.
> 
> The **path sum** of a path is the sum of the node's values in the path.
> 
> Given the `root` of a binary tree, return _the maximum **path sum** of any **non-empty**path_.
> 
> **Example 1:**
> 
> ![](https://assets.leetcode.com/uploads/2020/10/13/exx1.jpg)
> 
> **Input:** root = [1,2,3]
> **Output:** 6
> **Explanation:** The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.
> 
> **Example 2:**
> 
> ![](https://assets.leetcode.com/uploads/2020/10/13/exx2.jpg)
> 
> **Input:** root = [-10,9,20,null,null,15,7]
> **Output:** 42
> **Explanation:** The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 = 42.
> 
> **Constraints:**
> 
> - The number of nodes in the tree is in the range `[1, 3 * 104]`.
> - `-1000 <= Node.val <= 1000`

Post order DFS

1. Path starts at root and goes down through the root's left child.
2. Path starts at root and goes down through the root's right child.
3. Path involves both the left and the right child.
4. Path doesn't involve any child. The root itself is the only element of the path with maximum sum.

Since every path must include the root, we begin by assuming that the initial path sum is equal to the root node's value. To find the overall maximum path sum, we first determine the possible path sums from both the left and right subtrees.

The contributions from these subtrees can be either negative or positive. It is essential to consider a subtree's contribution only if it is positive; if the contribution is negative, it would decrease the overall path sum, so we ignore it.

Therefore, we need to calculate the maximum gain in path sum from both the left and right subtrees. This means we must process the child nodes before the parent node, which is why a post-order traversal is ideal: it ensures that we fully explore the children before evaluating the parent node's contribution.

While we calculate the maximum path sums that include the root, we also need to consider paths that do not pass through the root. To accommodate this, our recursive function not only returns the maximum path sum contribution from a subtree but also tracks the overall maximum path sum encountered during the traversal. We update this maximum whenever we find a new higher sum, regardless of whether the path includes the root or not.

```python
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        """
        O(N), O(H)
        """
        
        self.max_sum = float('-inf')
        
        def max_gain(node):
            if not node:
                return 0  # Base case: no contribution from null nodes
            
            # Recursively get the maximum contribution from left and right subtrees, only consider positive gains
            left_gain = max(max_gain(node.left), 0)
            right_gain = max(max_gain(node.right), 0)
            
            # Current path sum including the node
            current_path_sum = node.val + left_gain + right_gain
            
            # Update the global maximum path sum
            self.max_sum = max(self.max_sum, current_path_sum)
            
            # Return the maximum gain for the current node to its parent
            return node.val + max(left_gain, right_gain)
        
        max_gain(root)
        return self.max_sum
```

64. [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)

> Given the `root` of a binary tree, return _the level order traversal of its nodes' values_. (i.e., from left to right, level by level).
> 
> **Example 1:**
> 
> ![](https://assets.leetcode.com/uploads/2021/02/19/tree1.jpg)
> 
> **Input:** root = [3,9,20,null,null,15,7]
> **Output:** [[3],[9,20],[15,7]]
> 
> **Example 2:**
> 
> **Input:** root = [1]
> **Output:** [[1]]
> 
> **Example 3:**
> 
> **Input:** root = []
> **Output:** []
> 
> **Constraints:**
> 
> - The number of nodes in the tree is in the range `[0, 2000]`.
> - `-1000 <= Node.val <= 1000`

BFS

```python
from collections import deque

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        """
        O(N), O(W)
        """
        
        if not root:
            return []
        
        result = []
        queue = deque([root])
        
        while queue:
            level_size = len(queue)  # Number of nodes at the current level
            current_level = []
            
            for _ in range(level_size):
                node = queue.popleft()  # Dequeue the front node
                current_level.append(node.val)  # Add the node's value to the current level
                
                # Enqueue left and right children
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(current_level)  # Add the current level to the result
        
        return result
```

65. [Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)

> Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.
> 
> Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.
> 
> **Clarification:** The input/output format is the same as [how LeetCode serializes a binary tree](https://support.leetcode.com/hc/en-us/articles/32442719377939-How-to-create-test-cases-on-LeetCode#h_01J5EGREAW3NAEJ14XC07GRW1A). You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.
> 
> **Example 1:**
> 
> ![](https://assets.leetcode.com/uploads/2020/09/15/serdeser.jpg)
> 
> **Input:** root = [1,2,3,null,null,4,5]
> **Output:** [1,2,3,null,null,4,5]
> 
> **Example 2:**
> 
> **Input:** root = []
> **Output:** []
> 
> **Constraints:**
> 
> - The number of nodes in the tree is in the range `[0, 104]`.
> - `-1000 <= Node.val <= 1000`

We can use a pre-order traversal for serialisation, which captures the structure of the tree as we traverse. For deserialisation, we can split the string and reconstruct the tree using the same traversal order.

```
      1
     / \
    2   3
       / \
      4   5

Serialized String: `"1,2,#,#,3,4,#,#,5,#,#"` (# represents null)
```

```python
class Codec:
    def serialize(self, root: TreeNode) -> str:
        """Encodes a tree to a single string.
        
        O(N), O(N)
        """
        def preorder(node):
            if not node:
                return ["#"]  # Use "#" to represent null nodes
            return [str(node.val)] + preorder(node.left) + preorder(node.right)
        
        return ",".join(preorder(root))
    
    def deserialize(self, data: str) -> TreeNode:
        """Decodes your encoded data to tree.
        
        O(N), O(N)
        """
        values = iter(data.split(","))  # Create an iterator from the split string
        
        def build_tree():
            val = next(values)
            if val == "#":
                return None  # If the value is "#", return None
            node = TreeNode(int(val))
            node.left = build_tree()  # Recursively build the left subtree
            node.right = build_tree()  # Recursively build the right subtree
            return node
        
        return build_tree()
```

66. [Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/)

> Given the roots of two binary trees `root` and `subRoot`, return `true` if there is a subtree of `root` with the same structure and node values of `subRoot` and `false`otherwise.
> 
> A subtree of a binary tree `tree` is a tree that consists of a node in `tree` and all of this node's descendants. The tree `tree` could also be considered as a subtree of itself.
> 
> **Example 1:**
> 
> ![](https://assets.leetcode.com/uploads/2021/04/28/subtree1-tree.jpg)
> 
> **Input:** root = [3,4,5,1,2], subRoot = [4,1,2]
> **Output:** true
> 
> **Example 2:**
> 
> ![](https://assets.leetcode.com/uploads/2021/04/28/subtree2-tree.jpg)
> 
> **Input:** root = [3,4,5,1,2,null,null,null,null,0], subRoot = [4,1,2]
> **Output:** false
> 
> **Constraints:**
> 
> - The number of nodes in the `root` tree is in the range `[1, 2000]`.
> - The number of nodes in the `subRoot` tree is in the range `[1, 1000]`.
> - `-104 <= root.val <= 104`
> - `-104 <= subRoot.val <= 104`

DFS

For each node in `root`, check if the subtree rooted at that node matches `subRoot`. Use a helper function to compare two trees.

```python
class Solution:
    def isIdentical(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        # Both trees are empty
        if not root1 and not root2:
            return True
        
        # One tree is empty or values don't match
        if not root1 or not root2 or root1.val != root2.val:
            return False
        
        # Recursively check left and right children
        return self.isIdentical(root1.left, root2.left) and self.isIdentical(root1.right, root2.right)

    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        """
        N: nodes in root, M: nodes in subroot
        
        O(N*M), O(H)
        """
        
        # If root is None, subRoot cannot be a subtree
        if not root:
            return False
        
        # Check if the current tree matches subRoot or recurse on left/right children
        return (self.isIdentical(root, subRoot) or 
                self.isSubtree(root.left, subRoot) or 
                self.isSubtree(root.right, subRoot))
```

String matching

Serialise the trees into a string format, use substring search to check if the serialised subtree exists in the serialised main tree.

On top the standard serialisation method, we have one limitation.

```
tree = [12], subtree = [2]

Serialisation:

root: "12,#,#"
subtree: "2,#,#"

Subtree check:

"2,#,#" is a substring of "12,#,#"
```

We can add a character (like `^`) before a node's value to overcome this.

```
tree = [12], subtree = [2]

Serialisation:

root: "^12,#,#"
subtree: "^2,#,#"

Subtree check:

"^2,#,#" is not a substring of "^12,#,#"
```
\
```python
class Solution:
    def serialize(self, root: TreeNode) -> str:
        if not root:
            return '#'  # Use "#" to represent null nodes
        
        return f"^{root.val},{self.serialize(root.left)},{self.serialize(root.right)}"
    
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        """
        O(N + M), O(N + M)
        """
        
        return self.serialize(subRoot) in self.serialize(root)
```

67. [Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

> Given two integer arrays `preorder` and `inorder` where `preorder` is the preorder traversal of a binary tree and `inorder` is the inorder traversal of the same tree, construct and return _the binary tree_.
> 
> **Example 1:**
> 
> ![](https://assets.leetcode.com/uploads/2021/02/19/tree.jpg)
> 
> **Input:** preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
> **Output:** [3,9,20,null,null,15,7]
> 
> **Example 2:**
> 
> **Input:** preorder = [-1], inorder = [-1]
> **Output:** [-1]
> 
> **Constraints:**
> 
> - `1 <= preorder.length <= 3000`
> - `inorder.length == preorder.length`
> - `-3000 <= preorder[i], inorder[i] <= 3000`
> - `preorder` and `inorder` consist of **unique** values.
> - Each value of `inorder` also appears in `preorder`.
> - `preorder` is **guaranteed** to be the preorder traversal of the tree.
> - `inorder` is **guaranteed** to be the inorder traversal of the tree.

The two key observations are:
1. ﻿﻿﻿Preorder traversal follows `root -> left -> right`, given the `preorder` array, the `root` is the first element `preorder[0]`.
2. ﻿﻿﻿Inorder traversal follows `left -> root -> right`, if we know the position of `root`, we can recursively split the entire array Into two subtrees.

We use the first element of the preorder as the root. We use that to find the index of the root in the inorder array, which in turn gives us the left and right subtrees.

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        """
        O(N^2) due to the index lookup in the inorder array
        O(N)
        """
        
        if not preorder or not inorder:
            return None

        # The first element in preorder is the root
        root_val = preorder[0]
        root = TreeNode(root_val)

        # Find the index of the root in inorder
        root_index = inorder.index(root_val)

        # Split the preorder and inorder lists for left and right subtrees
        # Preorder for left subtree: preorder[1:root_index + 1]
        # Preorder for right subtree: preorder[root_index + 1:]
        # Inorder for left subtree: inorder[:root_index]
        # Inorder for right subtree: inorder[root_index + 1:]

        root.left = self.buildTree(preorder[1:root_index + 1], inorder[:root_index])
        root.right = self.buildTree(preorder[root_index + 1:], inorder[root_index + 1:])

```

68. [Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)

> Given the `root` of a binary tree, _determine if it is a valid binary search tree (BST)_.
> 
> A **valid BST** is defined as follows:
> 
> - The left 
>     
>     subtree
>     
>      of a node contains only nodes with keys **less than** the node's key.
> - The right subtree of a node contains only nodes with keys **greater than** the node's key.
> - Both the left and right subtrees must also be binary search trees.
> 
> **Example 1:**
> 
> ![](https://assets.leetcode.com/uploads/2020/12/01/tree1.jpg)
> 
> **Input:** root = [2,1,3]
> **Output:** true
> 
> **Example 2:**
> 
> ![](https://assets.leetcode.com/uploads/2020/12/01/tree2.jpg)
> 
> **Input:** root = [5,1,4,null,null,3,6]
> **Output:** false
> **Explanation:** The root node's value is 5 but its right child's value is 4.
> 
> **Constraints:**
> 
> - The number of nodes in the tree is in the range `[1, 104]`.
> - `-231 <= Node.val <= 231 - 1`

Keep track of the allowable value range for each node using min and max parameters.

Check if the current node's value is within the specified range (greater than min and less than max). Recursively check the left subtree (with updated max) and the right subtree (with updated min).

```python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        """
        O(N), O(H)
        """
        
        def validate(node: TreeNode, low: float, high: float) -> bool:
            if not node:
                return True
            if not (low < node.val < high):
                return False
            
            return validate(node.left, low, node.val) and validate(node.right, node.val, high)
        
        return validate(root, float('-inf'), float('inf'))
```

Inorder Traversal

Perform an inorder traversal of the tree and collect the node values, verify that the collected values are in strictly increasing order.

```python
class Solution:
    def __init__(self):
        self.previous_value = float('-inf')  # To track the last visited node value

    def isValidBST(self, node: Optional[TreeNode]) -> bool:
        """
        O(N), O(H)
        """
        
        if not node:
            return True  # An empty tree is a valid BST
        
        # Validate the left subtree
        if not self.isValidBST(node.left):
            return False

        # Check if the current node's value is greater than the previous value
        if not (node.val > self.previous_value):
            return False  # Violates the BST property
        
        # Update the previous_value to the current node's value
        self.previous_value = node.val
        
        # Validate the right subtree
        return self.isValidBST(node.right)
```

69. [Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)

> Given the `root` of a binary search tree, and an integer `k`, return _the_ `kth` _smallest value (**1-indexed**) of all the values of the nodes in the tree_.
> 
> **Example 1:**
> 
> ![](https://assets.leetcode.com/uploads/2021/01/28/kthtree1.jpg)
> 
> **Input:** root = [3,1,4,null,2], k = 1
> **Output:** 1
> 
> **Example 2:**
> 
> ![](https://assets.leetcode.com/uploads/2021/01/28/kthtree2.jpg)
> 
> **Input:** root = [5,3,6,2,4,null,null,1], k = 3
> **Output:** 3
> 
> **Constraints:**
> 
> - The number of nodes in the tree is `n`.
> - `1 <= k <= n <= 104`
> - `0 <= Node.val <= 104`
> 
> **Follow up:** If the BST is modified often (i.e., we can do insert and delete operations) and you need to find the kth smallest frequently, how would you optimize?

Iterative DFS

```python
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        """
        O(H + k), O(H)
        """
        
        stack = []
        while True:
            while root:
                stack.append(root)
                root = root.left
            
            root = stack.pop()
            k -= 1
            if k == 0:
                return root.val
            
            root = root.right
```

70. [Lowest Common Ancestor of BST](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)

> Given a binary search tree (BST), find the lowest common ancestor (LCA) node of two given nodes in the BST.
> 
> According to the [definition of LCA on Wikipedia](https://en.wikipedia.org/wiki/Lowest_common_ancestor): “The lowest common ancestor is defined between two nodes `p` and `q` as the lowest node in `T` that has both `p` and `q`as descendants (where we allow **a node to be a descendant of itself**).”
> 
> **Example 1:**
> 
> ![](https://assets.leetcode.com/uploads/2018/12/14/binarysearchtree_improved.png)
> 
> **Input:** root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
> **Output:** 6
> **Explanation:** The LCA of nodes 2 and 8 is 6.
> 
> **Example 2:**
> 
> ![](https://assets.leetcode.com/uploads/2018/12/14/binarysearchtree_improved.png)
> 
> **Input:** root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
> **Output:** 2
> **Explanation:** The LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.
> 
> **Example 3:**
> 
> **Input:** root = [2,1], p = 2, q = 1
> **Output:** 2
> 
> **Constraints:**
> 
> - The number of nodes in the tree is in the range `[2, 105]`.
> - `-109 <= Node.val <= 109`
> - All `Node.val` are **unique**.
> - `p != q`
> - `p` and `q` will exist in the BST.

Tree traversal:
1. If both target nodes (let's call them `p` and `q`) are less than the current node, move to the left child.
2. If both are greater, move to the right child.
3. If one is on one side and the other is on the other side (or if one of them is equal to the current node), then the current node is the LCA.

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """
        O(H), O(H)
        """
        
        # If both p and q are less than root, LCA is in the left subtree
        if p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        # If both p and q are greater than root, LCA is in the right subtree
        elif p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        else:
            # We have found the split point; this is the LCA
            return root
```

Iterative traversal: saves recursion stack space

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """
        O(H), O(1)
        """
        
        # Start from the root and traverse the tree
        while root:
            # If both p and q are less than root, LCA is in the left subtree
            if p.val < root.val and q.val < root.val:
                root = root.left
            # If both p and q are greater than root, LCA is in the right subtree
            elif p.val > root.val and q.val > root.val:
                root = root.right
            else:
                # We have found the split point; this is the LCA
                return root
```

71. [Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)

> A [**trie**](https://en.wikipedia.org/wiki/Trie) (pronounced as "try") or **prefix tree** is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker.
> 
> Implement the Trie class:
> 
> - `Trie()` Initializes the trie object.
> - `void insert(String word)` Inserts the string `word` into the trie.
> - `boolean search(String word)` Returns `true` if the string `word` is in the trie (i.e., was inserted before), and `false` otherwise.
> - `boolean startsWith(String prefix)` Returns `true` if there is a previously inserted string `word` that has the prefix `prefix`, and `false` otherwise.
> 
> **Example 1:**
> 
> **Input**
> ["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
> [[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
> **Output**
> [null, null, true, false, true, null, true]
> 
> **Explanation**
> Trie trie = new Trie();
> trie.insert("apple");
> trie.search("apple");   // return True
> trie.search("app");     // return False
> trie.startsWith("app"); // return True
> trie.insert("app");
> trie.search("app");     // return True
> 
> **Constraints:**
> 
> - `1 <= word.length, prefix.length <= 2000`
> - `word` and `prefix` consist only of lowercase English letters.
> - At most `3 * 104` calls **in total** will be made to `insert`, `search`, and `startsWith`.

```python
class TrieNode:
    def __init__(self):
        self.children = {}  # Dictionary to hold children nodes
        self.is_end_of_word = False  # Flag to mark the end of a word

class Trie:
    def __init__(self):
        self.root = TrieNode()  # Initialize the root of the Trie

    def insert(self, word: str) -> None:
        """
        n: key length
        
        O(n), O(n)
        """
        
        node = self.root
        for char in word:
            # If the character is not already a child, create a new TrieNode
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True  # Mark the end of the word

    def search(self, word: str) -> bool:
        """
        O(n), O(1)
        """
        
        node = self.root
        for char in word:
            if char not in node.children:
                return False  # If character not found, the word doesn't exist
            node = node.children[char]
        return node.is_end_of_word  # Return True if it's a complete word

    def startsWith(self, prefix: str) -> bool:
        """
        O(n), O(1)
        """
        
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False  # If character not found, no words with that prefix
            node = node.children[char]
        return True  # Prefix found
```

72. [Design Add and Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure)

> Design a data structure that supports adding new words and finding if a string matches any previously added string.
> 
> Implement the `WordDictionary` class:
> 
> - `WordDictionary()` Initializes the object.
> - `void addWord(word)` Adds `word` to the data structure, it can be matched later.
> - `bool search(word)` Returns `true` if there is any string in the data structure that matches `word` or `false` otherwise. `word` may contain dots `'.'` where dots can be matched with any letter.
> 
> **Example:**
> 
> **Input**
> ["WordDictionary","addWord","addWord","addWord","search","search","search","search"]
> [[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]
> **Output**
> [null,null,null,null,false,true,true,true]
> 
> **Explanation**
> WordDictionary wordDictionary = new WordDictionary();
> wordDictionary.addWord("bad");
> wordDictionary.addWord("dad");
> wordDictionary.addWord("mad");
> wordDictionary.search("pad"); // return False
> wordDictionary.search("bad"); // return True
> wordDictionary.search(".ad"); // return True
> wordDictionary.search("b.."); // return True
> 
> **Constraints:**
> 
> - `1 <= word.length <= 25`
> - `word` in `addWord` consists of lowercase English letters.
> - `word` in `search` consist of `'.'` or lowercase English letters.
> - There will be at most `2` dots in `word` for `search` queries.
> - At most `104` calls will be made to `addWord` and `search`.

Trie + Wildcard search

Search with Wildcards:
- If a `.` is encountered, check all possible children nodes recursively to see if any complete the word.
- Else, if a specific character is found, continue to the next character in the word (standard search).

```python
class TrieNode:
    def __init__(self):
        self.children = {}  # Dictionary to hold children nodes
        self.is_end_of_word = False  # Flag to mark the end of a word

class WordDictionary:
    def __init__(self):
        self.root = TrieNode()  # Initialize the root of the Trie

    def addWord(self, word: str) -> None:
        """
        N: len of word
        
        O(N), O(N)
        """
        
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()  # Create new TrieNode for new character
            node = node.children[char]
        node.is_end_of_word = True  # Mark the end of the word

    def search(self, word: str) -> bool:
        """
        N: len of word, K: len of wildcards
        
        O(N) for well-defined words, O(N.26^K) worst case for undefined words
        O(1) for well-defined words, O(N) worst case for undefined words
        """
        
        return self._search_in_node(word, self.root)

    def _search_in_node(self, word: str, node: TrieNode) -> bool:
        for i, char in enumerate(word):
            if char == '.':
                # If current character is a wildcard, check all children nodes
                for child in node.children.values():
                    if self._search_in_node(word[i + 1:], child):
                        return True
                return False  # If no child matched, return False
            else:
                if char not in node.children:
                    return False  # If character not found, return False
                node = node.children[char]
        return node.is_end_of_word  # Return True if it's the end of a valid word
```

73. [Word Search II](https://leetcode.com/problems/word-search-ii/)

> Given an `m x n` `board` of characters and a list of strings `words`, return _all words on the board_.
> 
> Each word must be constructed from letters of sequentially adjacent cells, where **adjacent cells** are horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.
> 
> **Example 1:**
> 
> ![](https://assets.leetcode.com/uploads/2020/11/07/search1.jpg)
> 
> **Input:** board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]
> **Output:** ["eat","oath"]
> 
> **Example 2:**
> 
> ![](https://assets.leetcode.com/uploads/2020/11/07/search2.jpg)
> 
> **Input:** board = [["a","b"],["c","d"]], words = ["abcb"]
> **Output:** []
> 
> **Constraints:**
> 
> - `m == board.length`
> - `n == board[i].length`
> - `1 <= m, n <= 12`
> - `board[i][j]` is a lowercase English letter.
> - `1 <= words.length <= 3 * 104`
> - `1 <= words[i].length <= 10`
> - `words[i]` consists of lowercase English letters.
> - All the strings of `words` are unique.

Trie + Backtracking

When initiating the search, we insert all the words into the Trie for efficient prefix searching. The backtracking starts from each cell on the board, exploring all possible paths while simultaneously traversing the Trie. By marking cells as visited during the search, we prevent cycles and ensure that each character is used only once per word formation.

Optimisation

An essential optimisation is pruning the Trie during the backtracking process: once a word is found, its corresponding node can be marked to avoid re-checking. This reduces the search space and improves efficiency, particularly in cases with many overlapping words.

```python
node.is_end_of_word = False  # Avoid duplicate entries
```

When a complete word is found (i.e., when `node.is_end_of_word` is `True`), the word is added to the `result` set. Immediately after that, the algorithm sets `node.is_end_of_word` to `False`. This marks the node as no longer representing the end of a word, effectively preventing the algorithm from adding the same word again during subsequent searches.

This change means that if the DFS reaches this node again in future iterations, it will not consider it a valid endpoint for a word, thereby pruning unnecessary paths in the Trie and optimising the search process.

Additionally, while the initial implementation doesn't explicitly remove child nodes from the Trie, you could further enhance pruning by checking if the node has no children after finding a word and removing it to save space, which can be done as follows:

```python
# After restoring the cell's original value
if not node.children and not node.is_end_of_word:
    node.children.pop(char, None)
```

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        """
        N and M are the dimensions of the board
        L is the length of the longest word
        T is the total number of letters in the dictionary
        
        O(N * M * (4 * 3^(L-1))): no. of cells * backtracking steps
        number of cells: N * M
        maximum steps in backtracking for each starting cell: 4 * 3^(L-1)
        
        O(T): Trie storage
        """
        
        result = set()
        m, n = len(board), len(board[0])

        def backtrack(node, x, y, path):
            # If the current TrieNode marks the end of a word, add the word to the result
            if node.is_end_of_word:
                result.add(path)
                node.is_end_of_word = False  # Avoid duplicate entries
            
            # Boundary and visited cell check
            if x < 0 or x >= m or y < 0 or y >= n or board[x][y] == '#':
                return
            
            char = board[x][y]
            # If the character is not a valid child in the Trie, exit
            if char not in node.children:
                return
            
            # Mark the cell as visited
            board[x][y] = '#'
            
            # Explore the neighboring cells
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                backtrack(node.children[char], x + dx, y + dy, path + char)
            
            # Restore the cell's original value after exploring (unvisit)
            board[x][y] = char

        # Create a Trie and insert all words
        trie = Trie()
        for word in words:
            trie.insert(word)
        
        # Start backtracking from each cell in the board
        for i in range(m):
            for j in range(n):
                backtrack(trie.root, i, j, "")
        
        return list(result)
```

---

## Heap

74. [Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)

> Given an integer array `nums` and an integer `k`, return _the_ `k` _most frequent elements_. You may return the answer in **any order**.
> 
> **Example 1:**
> 
> **Input:** nums = [1,1,1,2,2,3], k = 2
> **Output:** [1,2]
> 
> **Example 2:**
> 
> **Input:** nums = [1], k = 1
> **Output:** [1]
> 
> **Constraints:**
> 
> - `1 <= nums.length <= 105`
> - `-104 <= nums[i] <= 104`
> - `k` is in the range `[1, the number of unique elements in the array]`.
> - It is **guaranteed** that the answer is **unique**.
> 
> **Follow up:** Your algorithm's time complexity must be better than `O(n log n)`, where n is the array's size.

Heap

Use a dictionary to count the frequency of each element in the input list. Create a max-heap to keep track of the top k elements based on their frequency.

```python
from collections import Counter
import heapq

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        """
        U: unique elements, U <= N
        
        O(N + U log U + k log U) = O(NlogN)
        O(U) = O(N)
        """
        
        # Count the frequency of each number in the list: O(N)
        frequency = Counter(nums)
        
        # Use a max-heap to store elements based on their frequency: O(UlogU)
        max_heap = []
        for num, freq in frequency.items():
            # Push the negative frequency and the number into the max-heap
            heapq.heappush(max_heap, (-freq, num))
        
        # Extract the top k elements from the max-heap: O(klogU)
        return [heapq.heappop(max_heap)[1] for _ in range(k)]
```

Bucket sort

Create a counter, organise elements into buckets based on their frequencies and collect the top k elements by iterating through the buckets in reverse order.

```
nums = [1,1,1,2,2,2,3,3,4]

frequency = {1: 3, 2: 3, 3: 2, 4: 1}

bucket = [[], [4], [3], [1, 2], [], [], [], [], [], []]
```

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        """
        O(N), O(N)
        """
        
        n = len(nums)
        
        # Count frequencies
        frequency = Counter(nums)
        
        # Create buckets
        bucket = [[] for _ in range(n + 1)]
        for num, freq in frequency.items():
            bucket[freq].append(num)
        
        # Collect results
        flat_list = []
        for i in reversed(range(n + 1)):
            for num in bucket[i]:
                flat_list.append(num)
                if len(flat_list) == k:
                    return flat_list
```

75. [Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)

> The **median** is the middle value in an ordered integer list. If the size of the list is even, there is no middle value, and the median is the mean of the two middle values.
> 
> - For example, for `arr = [2,3,4]`, the median is `3`.
> - For example, for `arr = [2,3]`, the median is `(2 + 3) / 2 = 2.5`.
> 
> Implement the MedianFinder class:
> 
> - `MedianFinder()` initializes the `MedianFinder` object.
> - `void addNum(int num)` adds the integer `num` from the data stream to the data structure.
> - `double findMedian()` returns the median of all elements so far. Answers within `10-5`of the actual answer will be accepted.
> 
> **Example 1:**
> 
> **Input**
> ["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
> [[], [1], [2], [], [3], []]
> **Output**
> [null, null, null, 1.5, null, 2.0]
> 
> **Explanation**
> MedianFinder medianFinder = new MedianFinder();
> medianFinder.addNum(1);    // arr = [1]
> medianFinder.addNum(2);    // arr = [1, 2]
> medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
> medianFinder.addNum(3);    // arr[1, 2, 3]
> medianFinder.findMedian(); // return 2.0
> 
> **Constraints:**
> 
> - `-105 <= num <= 105`
> - There will be at least one element in the data structure before calling `findMedian`.
> - At most `5 * 104` calls will be made to `addNum` and `findMedian`.
> 
> **Follow up:**
> 
> - If all integer numbers from the stream are in the range `[0, 100]`, how would you optimize your solution?
> - If `99%` of all integer numbers from the stream are in the range `[0, 100]`, how would you optimize your solution?

2 Heaps
- A max-heap for the lower half of the numbers.
- A min-heap for the upper half of the numbers.

| Number | Max-Heap    | Min-Heap   | Median |
| ------ | ----------- | ---------- | ------ |
| **5**  | `[-5]`      | `[]`       | `5.0`  |
| **15** | `[-5]`      | `[15]`     | `10.0` |
| **10** | `[-10, -5]` | `[15]`     | `10.0` |
| **20** | `[-10, -5]` | `[15, 20]` | `12.5` |
| **30** | `[-15, -5]` | `[20, 30]` | `15.0` 

Adding a number
1. The number is added to the `max_heap` (inverted).
2. The largest element from `max_heap` is moved to `min_heap` to maintain the order.
3. If `min_heap` becomes larger than `max_heap`, the smallest element from `min_heap` is moved back to `max_heap`.

```python
import heapq

class MedianFinder:
    def __init__(self):
        # Max-heap for the lower half of the numbers
        self.max_heap = []
        # Min-heap for the upper half of the numbers
        self.min_heap = []

    def addNum(self, num: int) -> None:
        """
        O(logN), O(N)
        """
        
        # Add the number to the max-heap (invert the number for max-heap behavior)
        heapq.heappush(self.max_heap, -num)

        # Move the largest number from max-heap to min-heap to maintain order
        heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))

        # Ensure max-heap can have at most one more element than min-heap
        if len(self.max_heap) < len(self.min_heap):
            # Move the smallest number from min-heap back to max-heap
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def findMedian(self) -> float:
        """
        O(1), O(1)
        """
        
        # If max-heap has more elements, the median is the root of max-heap
        if len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        # If both heaps are of equal size, the median is the average of the roots
        return (self.min_heap[0] + -self.max_heap[0]) / 2

# Example usage:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()

```

SortedList

```python
from sortedcontainers import SortedList

class MedianFinder:
    def __init__(self):
        # SortedList to maintain the numbers in sorted order
        self.numbers = SortedList()

    def addNum(self, num: int) -> None:
        """
        O(logN): uses binary search
        """
        
        # Add the number to the multiset (SortedList)
        self.numbers.add(num)

    def findMedian(self) -> float:
        """
        O(1)
        """
        
        n = len(self.numbers)
        mid = n // 2
        if n % 2 == 1:
            # If odd, return the middle element
            return float(self.numbers[mid])
        else:
            # If even, return the average of the two middle elements
            return (self.numbers[mid - 1] + self.numbers[mid]) / 2
```

---
