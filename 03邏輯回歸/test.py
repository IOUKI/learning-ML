def bubbleSort(array: list):
    n = len(array)

    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = array[j + 1], array[j]
                swapped = True

            if not swapped:
                break
    
    return array

num1 = [4,3]
num2 = [2,1, 10]
myList = num1 + num2
myList = bubbleSort(myList)
print(myList)

if len(myList) % 2 != 0:
    midIndex = int(len(myList) / 2 - 0.5)
    print(midIndex)
    print(myList[midIndex])
else:
    midIndex1 = int(len(myList) / 2 - 1)
    midIndex2 = int(midIndex1 + 1)
    print(myList[midIndex1], myList[midIndex2])
    print((myList[midIndex1] + myList[midIndex2]) / 2)