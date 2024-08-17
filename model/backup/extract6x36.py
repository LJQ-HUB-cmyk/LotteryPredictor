import re
import pandas as pd
import datetime

# Function to map Vietnamese days to English
def translate_day(day_vn):
    days = {
        'Thứ 2': 'Mon',
        'Thứ 3': 'Tue',
        'Thứ 4': 'Wed',
        'Thứ 5': 'Thu',
        'Thứ 6': 'Fri',
        'Thứ 7': 'Sat',
        'Chủ nhật': 'Sun'
    }
    return days.get(day_vn, day_vn)

# Read the data from the file
with open('data/lott6x36_raw.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Process the data to extract dates and results
data_points = []
for i in range(0, len(lines), 2):
    date_line = lines[i].strip()
    numbers_line = lines[i+1].strip()
    # Remove extra spaces between numbers
    numbers_line = ' '.join(numbers_line.split())
    print(f"date_line: {date_line}")
    print(f"numbers_line: {numbers_line}")
    
    # Extract date using regex
    date_match = re.search(r'Mở thưởng ngày - (Thứ \d|Chủ nhật) , (\d{2}/\d{2}/\d{4})', date_line)
    if date_match:
        day_vn, date_str = date_match.groups()
        day_en = translate_day(day_vn)
        date_obj = datetime.datetime.strptime(date_str, '%d/%m/%Y')
        formatted_date = f"{day_en}, {date_obj.strftime('%d/%m/%Y')}"
        
        # Format the output string
        formatted_output = f"{formatted_date} {numbers_line}"
        data_points.append(formatted_output)

# Display cleaned data
for data_point in data_points:
    print(data_point)

# Save the cleaned data to a new file
with open('data/xs6x36_cleaned.txt', 'w', encoding='utf-8') as file:
    for data_point in data_points:
        file.write(data_point + '\n')