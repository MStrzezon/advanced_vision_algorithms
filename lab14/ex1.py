from matplotlib import pyplot as plt

events_filepath = 'dataset/events.txt'

events = []

with open(events_filepath, 'r') as f:
    while True:
        line = f.readline()
        if not line or float(line.split(' ')[0]) >= 1:
            break
        events.append(line.split(' '))

timestamps = []
x = []
y = []
polarity = []

for event in events:
    timestamps.append(float(event[0]))
    x.append(int(event[1]))
    y.append(int(event[2]))
    polarity.append(int(event[3]))

print('Number of events: ' + str(len(events)))
print('First timestamp: ' + str(timestamps[0]))
print('Last timestamp: ' + str(timestamps[-1]))
print('Maximum values of x, y: ' + str(max(x)) + ', ' + str(max(y)))
print('Minimum values of x, y: ' + str(min(x)) + ', ' + str(min(y)))
print('Number of events with positive polarity: ' + str(polarity.count(1)))
print('Number of events with negative polarity: ' + str(polarity.count(0)))


# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(x, timestamps, y, c=polarity)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Timestamp')
ax.set_zlabel('Y')
ax.set_title('3D Chart for event data')

# Display the plot
plt.show()