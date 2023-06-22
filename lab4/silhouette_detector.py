import cv2
import numpy as np


def are_any_objects(stats):
    return stats.shape[0] > 1


def if_human_proportions(stat):
    return stat[3] / stat[2] > 2


def is_below(main_stat, sub_stat):
    return 0 < sub_stat[1] - main_stat[1] - main_stat[3] < 100


def is_between_edges(main_stat, sub_stat):
    return sub_stat[0] + sub_stat[2] > main_stat[0] and sub_stat[0] < main_stat[0] + main_stat[2]


def update_right_bottom(right_bottom, sub_stat):
    return ((sub_stat[0] + sub_stat[2]) if (right_bottom[0] < sub_stat[0] + sub_stat[2]) else right_bottom[0]), (
        (sub_stat[1] + sub_stat[3]) if (right_bottom[1] < sub_stat[1] + sub_stat[3]) else right_bottom[1])


def update_left_top(left_top, sub_stat):
    return sub_stat[0] if left_top[0] > sub_stat[0] else left_top[0], left_top[1]


def update_rectangle_tops(left_top, right_bottom, sub_stat):
    return update_left_top(left_top, sub_stat), update_right_bottom(right_bottom, sub_stat)

def get_object_coordinates(stats, stat):
    left_top, right_bottom = (stat[0], stat[1]), (stat[0] + stat[2], stat[1] + stat[3])
    for (j, sub_stat) in enumerate(stats):
        if j == 0:
            continue
        if is_below(stat, sub_stat) and is_between_edges(stat, sub_stat):
            left_top, right_bottom = update_rectangle_tops(left_top, right_bottom, sub_stat)
    return left_top, right_bottom


class SilhouetteDetector:
    def __init__(self, video_name):
        self.original_current_frame = None
        self.current_frame = None
        self.video_capture = cv2.VideoCapture(video_name)

    def convert_frame_to_gray(self):
        self.current_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)

    def threshold_frame(self):
        _, self.current_frame = cv2.threshold(self.current_frame, 230, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def filter_frame(self):
        kernel = np.ones((3, 3), np.uint8)
        self.current_frame = cv2.medianBlur(self.current_frame, 3)
        self.current_frame = cv2.erode(self.current_frame, kernel)
        self.current_frame = cv2.dilate(self.current_frame, kernel)

    def process_frame(self):
        self.convert_frame_to_gray()
        self.threshold_frame()
        self.filter_frame()

    def get_frame_with_detected_objects(self):
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(self.current_frame)

        frame_with_boxes = self.original_current_frame.copy()
        if are_any_objects(stats):
            for (i, stat) in enumerate(stats):
                if i == 0:
                    continue
                if if_human_proportions(stat):
                    left_top, right_bottom = get_object_coordinates(stats, stat)
                    cv2.rectangle(frame_with_boxes, left_top, right_bottom, (255, 0, 0), 2)
        return frame_with_boxes

    def detect(self):
        while True:
            ret, self.current_frame = self.video_capture.read()
            if not ret:
                break
            self.original_current_frame = self.current_frame.copy()
            self.process_frame()
            frame_with_detected_objects = self.get_frame_with_detected_objects()
            cv2.imshow('video', frame_with_detected_objects)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    silhouette_detector = SilhouetteDetector('resources/vid1_IR.avi')
    silhouette_detector.detect()
