import pandas as pd
from collections import Counter
import numpy as np
from itertools import combinations

# data = """
# T6, 02/08/2024 09 14 19 34 37 41
# T4, 31/07/2024 02 04 12 16 18 42
# CN, 28/07/2024 01 13 21 25 31 39
# T6, 26/07/2024 18 20 28 31 39 45
# T4, 24/07/2024 07 08 20 27 32 34
# CN, 21/07/2024 09 10 23 25 28 38
# T6, 19/07/2024 01 10 20 25 34 35
# T4, 17/07/2024 09 10 11 16 21 30
# CN, 14/07/2024 09 18 33 37 38 43
# T6, 12/07/2024 11 17 25 26 28 29
# T4, 10/07/2024 23 24 36 37 40 45
# CN, 07/07/2024 04 08 22 23 26 45
# T6, 05/07/2024 04 23 33 38 40 44
# T4, 03/07/2024 11 18 24 34 38 43
# CN, 30/06/2024 05 23 25 28 30 43
# T6, 28/06/2024 04 06 16 32 41 44
# T4, 26/06/2024 08 10 29 30 33 40
# CN, 23/06/2024 09 11 19 29 31 44
# T6, 21/06/2024 03 07 11 16 19 35
# T4, 19/06/2024 08 12 17 23 26 27
# CN, 16/06/2024 03 16 17 18 25 37
# """
# read data from file
with open('data/lott645.txt', 'r') as file:
    data = file.read()

def freq_reversed_freq(data):
    numbers = []
    numbers_by_day = []
    stripped_data = data.strip().split('\n')
    for line in stripped_data:
        parts = line.split()
        nums = parts[2:8]
        numbers.extend(map(int, nums))
        nums = list(map(int, nums))
        numbers_by_day.append(nums)
    counter = Counter(numbers)
    frequency = pd.DataFrame(counter.items(), columns=['Number', 'Frequency']).sort_values(by='Frequency', ascending=False)
    frequency.reset_index(drop=True, inplace=True)
    return frequency, numbers_by_day, stripped_data

def random_walk_lottery(numbers_by_day, m, n, frequency):
    # Determine Set A (numbers not appeared in the last m days)
    last_appearances = {num: len(numbers_by_day) for num in range(1, 46)}
    # print(f"last_appearances: {last_appearances}")
    for day, nums in reversed(list(enumerate(numbers_by_day))):
        for num in nums:
            last_appearances[num] = day
    
    set_A = [num for num, days in last_appearances.items() if days >= m]
    # Determine Set B (n most frequently appearing numbers)
    set_B = frequency['Number'].head(n).tolist()

    # Select 6 numbers from the union of sets A and B
    intersection_set = list(set(set_A).intersection(set_B))
    if len(intersection_set) < 6:
        additional_numbers = frequency['Number'].tolist()
        for num in additional_numbers:
            if num not in intersection_set and last_appearances[num] >= m:
                intersection_set.append(num)
                if len(intersection_set) == 6:
                    break
    elif len(intersection_set) > 6:
        intersection_set = sorted(intersection_set, key=lambda x: frequency.loc[frequency['Number'] == x, 'Frequency'].values[0], reverse=True)[:6]
    return intersection_set


frequency, num_by_day, stripped_data = freq_reversed_freq(data)

total_days = len(num_by_day)
start_day = 0
window_size = total_days
window_data = stripped_data[-window_size-start_day:]
window_num_by_day = num_by_day[-window_size-start_day:]
window_frequency, _, _ = freq_reversed_freq("\n".join(window_data))
# I consider numbers that haven't appeared the last time and the most frequent numbers
selected_numbers = random_walk_lottery(window_num_by_day, 1, 20, window_frequency)
print(f"Predicted numbers: {selected_numbers}")
