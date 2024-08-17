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
    stripped_data = data.strip().split('\n')
    for line in stripped_data:
        parts = line.split()
        nums = parts[2:8]
        numbers.extend(map(int, nums))
    counter = Counter(numbers)
    frequency = pd.DataFrame(counter.items(), columns=['Number', 'Frequency']).sort_values(by='Frequency', ascending=False)
    frequency.reset_index(drop=True, inplace=True)
    return frequency
frequency = freq_reversed_freq(data)
print(f"Frequency of appearance of numbers: \n{frequency}")
# Get top 6 most frequent numbers
print(f"Predicted numbers: {frequency['Number'].head(6).tolist()}")