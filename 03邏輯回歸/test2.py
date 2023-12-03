def compute(nums1: list[int], nums2: list[int]):
    myList = nums1 + nums2

    myList.sort()

    total = len(myList)

    if total % 2 == 1:
        return float(myList[total // 2])
    else:
        mid1 = myList[total // 2 - 1]
        mid2 = myList[total // 2]
        return (float(mid1) + float(mid2)) / 2.0
    
nums1 = [1,3]
nums2 = [2]
print(compute(nums1, nums2))