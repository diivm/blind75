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
