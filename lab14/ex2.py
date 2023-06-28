events_filepath = 'dataset/events.txt'

events = []

lineNumber = 1
with open(events_filepath, 'r') as f:
    while True:
        line = f.readline()
        if not line or lineNumber >= 8000:
            break
        events.append(line.split(' '))
        lineNumber += 1


timestamps = []
x = []
y = []
polarity = []

for event in events:
    timestamps.append(float(event[0]))
    x.append(int(event[1]))
    y.append(180 - int(event[2]))
    polarity.append(int(event[3]))


# Create a 3D scatter plot
from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(x, timestamps, y, c=polarity)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Timestamp')
ax.set_zlabel('Y')
ax.set_title('3D Chart for event data')
ax.view_init(elev=0, azim=-90)
# Display the plot
plt.show()

img = plt.imread('dataset/images/frame_00000000.png')
plt.imshow(img, 'gray')
plt.show()

## 2
events = []

with open(events_filepath, 'r') as f:
    while True:
        line = f.readline()
        if not line or float(line.split(' ')[0]) >= 1:
            break
        if float(line.split(' ')[0]) >= 0.5:
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

# trwa minutę
# nie rozumiem o co chodzi w tym zadaniu
# od tego czy obiekt się poruszył
# zmiana piksela, jeśli wartość wzrosła(jaśniejszy) to 1, jeśli spadła to 0 (ciemniejszy)
# do góry