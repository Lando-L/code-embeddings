def count_occurences(elements, value):
    count = 0
    for x in elements:
        if x == value:
            count += 1
    return count

def contains(elements, value):
    for x in elements:
        if x == value:
            return True
    return False

def index_of(elements, value):
    index = -1
    for i, x in enumerate(elements):
        if x == value:
            index = i
    return index
