import numpy as np
import cv2
import copy
import os
import csv

### Used as config parameters ###
class OutputSpecs:
    def __init__(self, label, size, name=None):
        self.label = label
        self.size = size
        self.name = name if name is not None else label

    def __copy__(self):
        return VideoSaver.OutputSpecs(self.label, self.size, self.name)

class OutputSpecsCollection(object):
    def __init__(self, out_specs):
        if isinstance(out_specs, (list, tuple, np.ndarray)):
            self.collection = out_specs
        elif isinstance(out_specs, dict):
            self.collection = self._init_from_dict(out_specs)
        else:
            raise TypeError("OutputSpecsCollection: out_specs must be either a dict or a list of OutputSpecs. "
                            "Passed argument of type {}".format(type(out_specs)))

    def __getitem__(self, index): # so that Python can iterate over on its own
        return self.collection[index]

    def _init_from_dict(self, out_spec_dict):
        arr = []
        for k, v in out_spec_dict.items():
            arr.append(OutputSpecs(label=k, size=v["size"], name=v["name"]))
        return arr

    def get_by_label(self, label):
        i = self._find_label_index(label)
        if i >=0:  # label found
            return self.collection[i]
        return None  # label not found

    def _find_label_index(self, label):
        try:
            return next(i for i,v in enumerate(self.collection) if v.label == label)
        except StopIteration:  # label not found
            return -1

    def __copy__(self):
        return OutputSpecsCollection(self.collection)
    def __deepcopy__(self):
        return OutputSpecsCollection(copy.deepcopy(self.collection))
### /Used as config parameters ###


class OutputSaver(object):
    def __init__(self, output_path, input_id, enabled=False):
        self.output_path = output_path
        self.input_id = input_id
        self.enabled = enabled

class VideoSaver(OutputSaver):
    _DEFAULT_FPS = 33

    # def __init__(self, labels, output_sizes, output_names, fps, split_after_frames=None, *args, **kwargs):
    #     self.labels = labels # array of VisualizationMode for each out video
    #     self.output_sizes = output_sizes # array of sizes for each out video
    #     self.output_names = output_names # array of string id for each out video
    # input: dict: {label1: {size: (...,...), name: ""}
    def __init__(self, output_specs_col, fps=_DEFAULT_FPS, split_after_frames=None, *args, **kwargs):
        super(VideoSaver, self).__init__(*args, **kwargs)
        self.output_specs_coll = output_specs_col
        self.split_after_frames = split_after_frames
        self.fps = fps
        self.videos = VideoSaver._VideoCollection(self.output_specs_coll, self.output_path, self.fps, self.split_after_frames)
        self._init_videos()
        # self.caps = [None] * len(self.output_specs_col)

    def _init_videos(self):
        if self.enabled:
            for i, v in enumerate(self.videos):
                v.init_video()

    def write(self, label, frame):
        if self.enabled:
            v = self.videos.get_by_label(label)
            v.write(frame)

    def close(self, label):
        if self.enabled:
            v = self.videos.get_by_label(label)
            v.close()

    def close_all(self):
        if self.enabled:
            for v in self.videos:
                v.close()

    ## Private

    class _Video(OutputSpecs):
        def __init__(self, output_spec, output_path, fps, split_after_frames=None):
            self.output_spec = output_spec
            # super(VideoSaver._Video, self).__init__(**output_spec.__dict__) # doesn't work
            # copy parent attributes
            for k, v in output_spec.__dict__.items():
                self.__dict__[k] = copy.deepcopy(v)
            self.output_path = output_path
            self.fps = fps
            self._split_after_frames = split_after_frames
            self._cap = None
            self.n_writted_frames = 0
            if self.is_split_enabled():
                self._video_n = 1
                self._video_frame_i = 0

        def init_video(self):
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            video_out_path = self.output_path + "{}{}.mp4".format(self.name,
                    '_{:04d}'.format(self._video_n)
                            if self.is_split_enabled()
                            else '')
            print(video_out_path)
            self._cap = cv2.VideoWriter(video_out_path,
                                        cv2.VideoWriter_fourcc(*'mp4v'),  # cv2.VideoWriter_fourcc(*'avc1'), #cv2.VideoWriter_fourcc(*'H264'), # CAN'T use on macos unless I manually rebuild opencv
                                        self.fps, self.size, True)

        def write(self, frame):
            self._cap.write(frame)
            self.n_writted_frames += 1
            if self.is_split_enabled():
                self._video_frame_i += 1
                if self._video_frame_i >= self._split_after_frames:
                    # print("Saving video chunk {}...".format(video_out_n))
                    self._video_n += 1
                    self._video_frame_i = 0
                    self.close()
                    self.init_video()

        def close(self):
            if self._cap is not None:
                # close all current videos and create a new one
                self._cap.release()

        def is_split_enabled(self):
            return self._split_after_frames >= 0

        def __copy__(self):
            return VideoSaver._Video(copy.deepcopy(self.output_spec), self.output_path, self.fps, self._split_after_frames)

    class _VideoCollection(OutputSpecsCollection):
        def __init__(self, output_specs, output_path, fps, split_after_frames=None):
            # super(VideoSaver._VideoCollection, self).__init__(out_specs)
            self.collection = []
            for k, v in enumerate(output_specs):
                self.collection.append(VideoSaver._Video(v, output_path, fps, split_after_frames))


class ImageSaver(OutputSaver):
    def __init__(self,  *args, **kwargs):
        super(ImageSaver, self).__init__(*args, **kwargs)


class Display():
    def __init__(self):
        pass

    @staticmethod
    def _show_image(image, window_name="", window_size=None, wait_sec=10):
        if window_size is None:
            window_size = (1280,720)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_size[0], window_size[1])
        cv2.imshow(window_name, image)
        cv2.waitKey(wait_sec)

class CsvSaver(OutputSaver):
    def __init__(self, fields_names, *args, **kwargs):
        super(CsvSaver, self).__init__(*args, **kwargs)
        self.fields_names = fields_names
        if self.enabled:
            self.csv_file = open(self.output_path + '{}_points.csv'.format(self.input_id), mode='w')
            self.csv_writer = csv.DictWriter(self.csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, fieldnames=self.fields_names)
            self.csv_writer.writeheader()

    def write(self, row):
        if self.enabled:
            self.csv_writer.writerow(row)

    def close(self):
        if self.enabled:
            self.csv_file.close()