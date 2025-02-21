import matplotlib
matplotlib.use('tkagg')  # Use 'tkagg' if you have a GUI environment
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import csv
import json
import datetime
import os

# Initialize lists for data storage
human_count = []
violate_count = []
restricted_entry = []
abnormal_activity = []

# Ensure the data file exists before reading
csv_file = 'processed_data/crowd_data.csv'
json_file = 'processed_data/video_data.json'

if not os.path.exists(csv_file):
    print(f"Error: {csv_file} not found. Please ensure the file is generated before running this script.")
    exit()

if not os.path.exists(json_file):
    print(f"Error: {json_file} not found. Please ensure the file is generated before running this script.")
    exit()

# Read CSV file safely
with open(csv_file, 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)  # Skip header row

    for row in reader:
        try:
            human_count.append(int(row[1]))
            violate_count.append(int(row[2]))
            restricted_entry.append(bool(int(row[3])))
            abnormal_activity.append(bool(int(row[4])))
        except (ValueError, IndexError) as e:
            print(f"Warning: Skipping invalid row {row} due to error: {e}")

# Read JSON file safely
with open(json_file, 'r') as file:
    try:
        data = json.load(file)
        data_record_frame = data["DATA_RECORD_FRAME"]
        is_cam = data["IS_CAM"]
        vid_fps = data["VID_FPS"]
        start_time = data["START_TIME"]
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse {json_file}: {e}")
        exit()

# Convert start time to datetime format
try:
    start_time = datetime.datetime.strptime(start_time, "%d/%m/%Y, %H:%M:%S")
except ValueError as e:
    print(f"Error: Invalid start time format in JSON: {e}")
    exit()

# Calculate time steps
time_steps = data_record_frame / vid_fps
data_length = len(human_count)

# Generate time axis
time_axis = []
graph_height = max(human_count) if human_count else 1

fig, ax = plt.subplots()
time = start_time
for i in range(data_length):
    time += datetime.timedelta(seconds=time_steps)
    time_axis.append(time)
    next_time = time + datetime.timedelta(seconds=time_steps)
    rect_width = mdates.date2num(next_time) - mdates.date2num(time)

    if restricted_entry[i]:
        ax.add_patch(patches.Rectangle((mdates.date2num(time), 0), rect_width, graph_height / 10, facecolor='red', fill=True))
    if abnormal_activity[i]:
        ax.add_patch(patches.Rectangle((mdates.date2num(time), 0), rect_width, graph_height / 20, facecolor='blue', fill=True))

# Plot data
violate_line, = plt.plot(time_axis, violate_count, linewidth=3, label="Violation Count")
crowd_line, = plt.plot(time_axis, human_count, linewidth=3, label="Crowd Count")

# Formatting
plt.title("Crowd Data versus Time")
plt.xlabel("Time")
plt.ylabel("Count")

# Add legend
re_legend = patches.Patch(color="red", label="Restricted Entry Detected")
an_legend = patches.Patch(color="blue", label="Abnormal Crowd Activity Detected")
plt.legend(handles=[crowd_line, violate_line, re_legend, an_legend])

# Save and display the plot
plt.savefig('crowd_analysis_plot.png')
plt.show()
