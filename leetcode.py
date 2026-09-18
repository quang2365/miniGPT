arr = [8,3,10,1,0,2,12,7]
def minimumDeletions(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    max_idx = 0
    min_idx = 0
    nums.reverse()
    print(nums)
    for i,value in enumerate(nums):
        if nums[i] > nums[max_idx]:
            max_idx = i
        if nums[i] < nums[min_idx]:
            min_idx = i
    return nums[min_idx], nums[max_idx]
min, max = minimumDeletions(arr)
print(min, max)